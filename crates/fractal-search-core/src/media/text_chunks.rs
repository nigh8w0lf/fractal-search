//! Source-preserving document extraction. Structure is a soft preference; byte
//! and actual formatted-token limits are hard bounds and never truncate input.

use super::{
    MediaKind, SegmentSpec, MAX_TEXT_FILE_BYTES, MAX_TEXT_SEGMENTS, TEXT_OVERLAP_BYTES,
    TEXT_SEGMENT_BYTES,
};
use crate::preprocess::{validate_document_input, MAX_SEQUENCE_LENGTH};
use anyhow::{ensure, Context, Result};

/// At most one quarter of a full passage is sacrificed to a structural split.
/// This comes from the application's byte bound, not another model's token cap.
const STRUCTURE_LOOKBACK_BYTES: usize = TEXT_SEGMENT_BYTES / 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BoundaryKind {
    Paragraph,
    Fence,
    Heading,
}

#[derive(Clone, Copy, Debug)]
struct Boundary {
    offset: usize,
    kind: BoundaryKind,
}

struct Structure {
    boundaries: Vec<Boundary>,
    line_starts: Vec<usize>,
}

impl Structure {
    fn scan(text: &str, markdown: bool) -> Self {
        let mut boundaries = Vec::new();
        let mut line_starts = Vec::new();
        let mut offset = 0;
        let mut fence: Option<(u8, usize)> = None;
        let mut previous_line: Option<(usize, &str)> = None;
        for line in text.split_inclusive('\n') {
            line_starts.push(offset);
            let content = line.trim_end_matches(['\r', '\n']);
            let end = offset + line.len();
            let trimmed = content.trim_start_matches(' ');
            let indent = content.len() - trimmed.len();
            if markdown && indent <= 3 {
                if let Some((marker, length)) = fence {
                    let run = trimmed.bytes().take_while(|&byte| byte == marker).count();
                    if run >= length && trimmed[run..].trim().is_empty() {
                        boundaries.push(Boundary {
                            offset: end,
                            kind: BoundaryKind::Fence,
                        });
                        fence = None;
                    }
                    previous_line = None;
                    offset = end;
                    continue;
                }
                if let Some(marker @ (b'`' | b'~')) = trimmed.bytes().next() {
                    let length = trimmed.bytes().take_while(|&byte| byte == marker).count();
                    if length >= 3 && (marker != b'`' || !trimmed[length..].contains('`')) {
                        boundaries.push(Boundary {
                            offset,
                            kind: BoundaryKind::Fence,
                        });
                        fence = Some((marker, length));
                        previous_line = None;
                        offset = end;
                        continue;
                    }
                }
                let hashes = trimmed.bytes().take_while(|&byte| byte == b'#').count();
                if (1..=6).contains(&hashes)
                    && (trimmed.len() == hashes
                        || matches!(trimmed.as_bytes()[hashes], b' ' | b'\t'))
                {
                    boundaries.push(Boundary {
                        offset,
                        kind: BoundaryKind::Heading,
                    });
                } else if is_setext_underline(trimmed) {
                    if let Some((heading_start, previous)) = previous_line {
                        if !previous.trim().is_empty() {
                            boundaries.push(Boundary {
                                offset: heading_start,
                                kind: BoundaryKind::Heading,
                            });
                        }
                    }
                }
            }
            // Markdown-looking lines inside an oversized fence are never
            // promoted to headings/paragraphs. Hard byte cuts still progress.
            if fence.is_none() && content.trim().is_empty() {
                boundaries.push(Boundary {
                    offset: end,
                    kind: BoundaryKind::Paragraph,
                });
            }
            previous_line = if fence.is_none() {
                Some((offset, content))
            } else {
                None
            };
            offset = end;
        }
        // Setext headings can add an earlier offset; sort once for bounded
        // lookback queries rather than repeatedly rescanning the entire file.
        boundaries.sort_unstable_by_key(|boundary| (boundary.offset, boundary.kind));
        Self {
            boundaries,
            line_starts,
        }
    }

    fn preferred_end(&self, start: usize, end: usize) -> usize {
        let minimum = end
            .saturating_sub(STRUCTURE_LOOKBACK_BYTES)
            .max(start + (end - start) / 2)
            .max(start + 1);
        let first = self
            .boundaries
            .partition_point(|boundary| boundary.offset < minimum);
        let last = self
            .boundaries
            .partition_point(|boundary| boundary.offset <= end);
        self.boundaries[first..last]
            .iter()
            .max_by_key(|boundary| (boundary.kind, boundary.offset))
            .map_or(end, |boundary| boundary.offset)
    }

    fn line_at(&self, byte: usize) -> u32 {
        // Inputs are limited to 2 MiB, so the one-based line count fits u32.
        self.line_starts.partition_point(|&start| start <= byte) as u32
    }
}

fn is_setext_underline(line: &str) -> bool {
    let line = line.trim_end_matches([' ', '\t']);
    let Some(marker @ (b'=' | b'-')) = line.bytes().next() else {
        return false;
    };
    line.bytes().all(|byte| byte == marker)
}

/// Extract exact slices of the original text, preserving byte/line provenance.
/// `count_tokens` MUST use the real tokenizer and include document title,
/// formatting, prefix and special tokens with truncation/padding disabled.
/// Only whitespace-only intervals may be omitted, because they cannot embed.
pub fn text_segments_with_token_counter<F>(
    title: &str,
    text: &str,
    markdown: bool,
    mut count_tokens: F,
) -> Result<Vec<SegmentSpec>>
where
    F: FnMut(&str, &str) -> Result<usize>,
{
    ensure!(!text.trim().is_empty(), "Document contains no text");
    ensure!(!text.contains('\0'), "Document contains NUL bytes");
    ensure!(
        text.len() as u64 <= MAX_TEXT_FILE_BYTES,
        "Document exceeds the 2 MiB text limit"
    );
    // Validate the title even when the file begins with blank-only intervals.
    validate_document_input(title, "x")?;
    let structure = Structure::scan(text, markdown);
    let mut segments = Vec::new();
    let mut start = 0;
    let mut covered_end = 0;
    while start < text.len() {
        let mut end = (start + TEXT_SEGMENT_BYTES).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        end = fit_tokens(title, text, start, end, &mut count_tokens)?;
        // Different prefixes can tokenize differently. If the retained overlap
        // consumes the available context, discard overlap and retry from the
        // already-covered frontier instead of emitting duplicate-only chunks.
        if end <= covered_end && start < covered_end {
            start = covered_end;
            continue;
        }
        // Do not shorten a complete final passage just to end at a heading.
        let preferred = if end < text.len() {
            structure.preferred_end(start, end)
        } else {
            end
        };
        if preferred > covered_end && preferred != end {
            // Token counts are not mathematically monotonic under shortening.
            // Recheck the exact final slice after choosing another boundary.
            if fits_tokens(title, &text[start..preferred], &mut count_tokens)? {
                end = preferred;
            }
        }
        ensure!(
            end > start && end > covered_end,
            "Cannot make progress while splitting a UTF-8 document"
        );
        if !text[start..end].trim().is_empty() {
            ensure!(segments.len() < MAX_TEXT_SEGMENTS, "Document exceeds the {MAX_TEXT_SEGMENTS}-segment limit; split the file before indexing");
            segments.push(SegmentSpec {
                index: segments.len() as u32,
                kind: MediaKind::Text,
                start_ms: None,
                end_ms: None,
                text: Some(text[start..end].to_owned()),
                start_byte: Some(start as u64),
                end_byte: Some(end as u64),
                start_line: Some(structure.line_at(start)),
                end_line: Some(structure.line_at(end - 1)),
            });
        }
        covered_end = end;
        if end == text.len() {
            break;
        }
        // Tiny token-limited passages retain at most half their content; the
        // normal byte-bounded case retains the existing <=256-byte overlap.
        let overlap = TEXT_OVERLAP_BYTES.min((end - start) / 2);
        start = end - overlap;
        while !text.is_char_boundary(start) {
            start += 1;
        }
    }
    Ok(segments)
}

fn fit_tokens<F>(
    title: &str,
    text: &str,
    start: usize,
    end: usize,
    count_tokens: &mut F,
) -> Result<usize>
where
    F: FnMut(&str, &str) -> Result<usize>,
{
    let fits = |end: usize, counter: &mut F| fits_tokens(title, &text[start..end], counter);
    if fits(end, count_tokens)? {
        return Ok(end);
    }
    let boundaries: Vec<usize> = text[start..end]
        .char_indices()
        .map(|(offset, character)| start + offset + character.len_utf8())
        .collect();
    // Find a checked fitting prefix, then enlarge it. Binary search is only an
    // efficiency heuristic, never a safety assumption about monotonic counts:
    // every returned endpoint was actually tokenized and verified.
    let mut upper = boundaries.len() - 1;
    let mut lower = 0;
    ensure!(fits(boundaries[0], count_tokens)?, "Document title, formatting and one character exceed the {MAX_SEQUENCE_LENGTH}-token model context; reduce the title or inspect the tokenizer");
    while lower + 1 < upper {
        let middle = lower + (upper - lower) / 2;
        if fits(boundaries[middle], count_tokens)? {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    Ok(boundaries[lower])
}

fn fits_tokens<F>(title: &str, passage: &str, counter: &mut F) -> Result<bool>
where
    F: FnMut(&str, &str) -> Result<usize>,
{
    if passage.trim().is_empty() {
        return Ok(true);
    }
    let count = counter(title, passage)
        .context("Counting formatted document tokens for text extraction")?;
    ensure!(
        count >= 2,
        "Document tokenizer returned an invalid formatted token count"
    );
    Ok(count <= MAX_SEQUENCE_LENGTH)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(text: &str, markdown: bool) -> Vec<SegmentSpec> {
        text_segments_with_token_counter("fixture.md", text, markdown, |title, passage| {
            Ok(title.len() + passage.len() + 16)
        })
        .unwrap()
    }

    fn assert_source_coverage(text: &str, segments: &[SegmentSpec]) {
        let mut covered = vec![false; text.len()];
        let newlines: Vec<usize> = text
            .bytes()
            .enumerate()
            .filter_map(|(offset, byte)| (byte == b'\n').then_some(offset))
            .collect();
        let mut last_end = 0;
        for (index, segment) in segments.iter().enumerate() {
            let start = segment.start_byte.unwrap() as usize;
            let end = segment.end_byte.unwrap() as usize;
            assert_eq!(segment.index, index as u32);
            assert!(end > start && end > last_end && end - start <= TEXT_SEGMENT_BYTES);
            assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
            assert_eq!(segment.text.as_deref(), Some(&text[start..end]));
            if start < last_end {
                assert!(last_end - start <= TEXT_OVERLAP_BYTES);
            }
            assert_eq!(
                segment.start_line.unwrap() as usize,
                newlines.partition_point(|&offset| offset < start) + 1
            );
            assert_eq!(
                segment.end_line.unwrap() as usize,
                newlines.partition_point(|&offset| offset < end - 1) + 1
            );
            covered[start..end].fill(true);
            last_end = end;
        }
        for (offset, character) in text.char_indices() {
            if !character.is_whitespace() {
                assert!(covered[offset..offset + character.len_utf8()]
                    .iter()
                    .all(|&byte| byte));
            }
        }
    }

    #[test]
    fn bounded_lookback_prefers_headings_and_preserves_utf8_offsets() {
        let text = format!(
            "{}\n\n# Second section\r\n{}",
            "a".repeat(3498),
            "é猫🦊\r\n".repeat(700)
        );
        let segments = chunks(&text, true);
        assert_eq!(segments[0].end_byte, Some(3500));
        assert_source_coverage(&text, &segments);
        let too_early = format!("{}\n# Heading\n{}", "a".repeat(100), "b".repeat(6000));
        assert_eq!(
            chunks(&too_early, true)[0].end_byte,
            Some(TEXT_SEGMENT_BYTES as u64)
        );
        let plain = format!("{}\n# Heading\n{}", "a".repeat(3499), "b".repeat(6000));
        assert_eq!(
            chunks(&plain, false)[0].end_byte,
            Some(TEXT_SEGMENT_BYTES as u64)
        );
    }

    #[test]
    fn paragraphs_and_setext_headings_are_structural_boundaries() {
        let paragraph = format!("{}\r\n\r\n{}", "a".repeat(3596), "b".repeat(5000));
        assert_eq!(chunks(&paragraph, false)[0].end_byte, Some(3600));
        assert_source_coverage(&paragraph, &chunks(&paragraph, false));
        let setext = format!(
            "{}\nSection\n=======\n{}",
            "a".repeat(3499),
            "b".repeat(6000)
        );
        assert_eq!(chunks(&setext, true)[0].end_byte, Some(3500));
    }

    #[test]
    fn fences_are_kept_together_when_they_fit_and_long_fences_progress() {
        let opening = format!(
            "{}\n~~~rust\n{}\n~~~\nend",
            "a".repeat(3499),
            "code\n".repeat(400)
        );
        assert_eq!(chunks(&opening, true)[0].end_byte, Some(3500));
        assert_source_coverage(&opening, &chunks(&opening, true));
        let closing = format!("```\n{}\n```\n{}", "x".repeat(3491), "tail ".repeat(1000));
        assert_eq!(chunks(&closing, true)[0].end_byte, Some(3500));
        let long = format!(
            "````rust\n{}\n```\n# still code\n\n{}\n````\n",
            "x".repeat(3100),
            "🦊".repeat(4000)
        );
        let segments = chunks(&long, true);
        assert_eq!(
            segments[0].end_byte.unwrap() as usize,
            (0..=TEXT_SEGMENT_BYTES)
                .rev()
                .find(|&i| long.is_char_boundary(i))
                .unwrap()
        );
        assert_source_coverage(&long, &segments);
    }

    #[test]
    fn token_budget_includes_title_and_controls_chunking_without_truncation() {
        let text = "é猫🦊x\n".repeat(1600);
        let title = "t".repeat(1024);
        // A deterministic expanding-tokenizer fixture makes the real context
        // limit bind before the byte cap. Prefix/special overhead is explicit.
        let count = |title: &str, passage: &str| -> Result<usize> {
            Ok(2 * title.len() + 64 + passage.len() * 3)
        };
        let segments = text_segments_with_token_counter(&title, &text, true, count).unwrap();
        assert!(segments[0].text.as_ref().unwrap().len() < TEXT_SEGMENT_BYTES);
        assert!(segments
            .iter()
            .all(
                |segment| count(&title, segment.text.as_deref().unwrap()).unwrap()
                    <= MAX_SEQUENCE_LENGTH
            ));
        assert_source_coverage(&text, &segments);
    }

    #[test]
    fn shortening_at_a_heading_is_rechecked_for_nonmonotonic_token_counts() {
        let text = format!("{}\n# Heading\n{}", "a".repeat(3499), "tail ".repeat(1000));
        let count = |_: &str, passage: &str| -> Result<usize> {
            Ok(if passage.len() == 3500 {
                MAX_SEQUENCE_LENGTH + 1
            } else {
                passage.len() + 10
            })
        };
        let segments = text_segments_with_token_counter("fixture.md", &text, true, count).unwrap();
        assert_eq!(segments[0].end_byte, Some(TEXT_SEGMENT_BYTES as u64));
        assert!(segments.iter().all(|segment| count(
            "fixture.md",
            segment.text.as_deref().unwrap()
        )
        .unwrap()
            <= MAX_SEQUENCE_LENGTH));
        assert_source_coverage(&text, &segments);
    }

    #[test]
    fn tiny_token_windows_reduce_overlap_and_fail_explicitly_when_impossible() {
        let text = "é猫🦊a".repeat(100);
        let segments = text_segments_with_token_counter("t", &text, false, |_, passage| {
            Ok(MAX_SEQUENCE_LENGTH - 5 + passage.len())
        })
        .unwrap();
        assert_source_coverage(&text, &segments);
        assert!(segments
            .iter()
            .all(|segment| segment.text.as_ref().unwrap().len() <= 5));
        assert!(
            text_segments_with_token_counter("t", "🦊", false, |_, _| Ok(MAX_SEQUENCE_LENGTH + 1))
                .unwrap_err()
                .to_string()
                .contains("one character")
        );
        assert!(
            text_segments_with_token_counter("t", "text", false, |_, _| Ok(0))
                .unwrap_err()
                .to_string()
                .contains("invalid formatted token count")
        );
    }

    #[test]
    fn whitespace_gaps_keep_original_offsets_and_trailing_lines_are_inclusive() {
        let text = format!("{}one\r\ntwo\r\n", " \r\n".repeat(4000));
        let segments = chunks(&text, true);
        assert_source_coverage(&text, &segments);
        let last = segments.last().unwrap();
        assert_eq!(last.end_byte, Some(text.len() as u64));
        assert_eq!(last.end_line, Some(4002));
        assert!(segments
            .iter()
            .all(|segment| !segment.text.as_ref().unwrap().trim().is_empty()));
    }

    #[test]
    fn mixed_unicode_and_structure_have_gap_free_nonblank_coverage() {
        let parts = [
            "a",
            "é",
            "猫",
            "🦊",
            "\r\n",
            "\n\n",
            "# Heading\n",
            "~~~\n",
            " ",
        ];
        let mut random = 0x5eeda11_u64;
        for fixture in 0..24 {
            let mut text = String::from("start\n");
            for _ in 0..3000 {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                text.push_str(parts[(random >> 32) as usize % parts.len()]);
            }
            text.push_str("\nfinish");
            let count = |title: &str, passage: &str| -> Result<usize> {
                Ok(title.len() + passage.len() * 3 + if passage.len() % 7 == 0 { 31 } else { 2 })
            };
            let segments =
                text_segments_with_token_counter("fixture", &text, fixture % 2 == 0, count)
                    .unwrap();
            assert_source_coverage(&text, &segments);
            assert!(segments.iter().all(|segment| count(
                "fixture",
                segment.text.as_deref().unwrap()
            )
            .unwrap()
                <= MAX_SEQUENCE_LENGTH));
        }
    }

    #[test]
    fn exact_caps_errors_and_repeated_runs_are_deterministic() {
        let text = "a".repeat(MAX_TEXT_FILE_BYTES as usize);
        let first = chunks(&text, true);
        assert!(first.len() < MAX_TEXT_SEGMENTS);
        assert_eq!(first, chunks(&text, true));
        assert_source_coverage(&text, &first);
        for input in [
            String::new(),
            " \r\n\t".into(),
            "bad\0text".into(),
            text + "x",
        ] {
            assert!(text_segments_with_token_counter("t", &input, true, |_, _| Ok(10)).is_err());
        }
        assert!(
            text_segments_with_token_counter("t", "text", true, |_, _| anyhow::bail!(
                "broken tokenizer"
            ))
            .unwrap_err()
            .chain()
            .any(|error| error.to_string().contains("broken tokenizer"))
        );
        assert!(
            text_segments_with_token_counter("t", &"a".repeat(3000), false, |_, passage| Ok(
                MAX_SEQUENCE_LENGTH - 1 + passage.len()
            ))
            .unwrap_err()
            .to_string()
            .contains("segment limit")
        );
    }
}
