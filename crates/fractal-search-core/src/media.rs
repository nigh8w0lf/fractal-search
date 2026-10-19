//! Bounded local document and media decoding for catalog indexing.
//!
//! Images keep the existing native decoder. Audio/video require the explicitly
//! installed `ffprobe` and `ffmpeg` executables; no network, playlist demuxer,
//! Python runtime, alternate decoder, or automatic model download is used.
//! Sampled video frames are bounded and downscaled before model-specific patch
//! preparation. Original files are never changed.

use anyhow::{bail, ensure, Context, Result};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

mod text_chunks;
pub use text_chunks::text_segments_with_token_counter;

pub const MAX_TEXT_FILE_BYTES: u64 = 2 * 1024 * 1024;
pub const TEXT_SEGMENT_BYTES: usize = 4096;
pub const TEXT_OVERLAP_BYTES: usize = 256;
pub const MAX_TEXT_SEGMENTS: usize = 1024;
/// Independent of the vector/schema contract: changing extraction requires an
/// explicit reindex, never a destructive catalog migration during open/search.
pub const TEXT_EXTRACTION_PROFILE: &str =
    "structural-v2-4096b-overlap256-lookback1024-formatted8192-offsets-v1";
pub const MAX_MEDIA_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_DURATION_MS: u64 = 2 * 60 * 60 * 1000;
pub const SEGMENT_DURATION_MS: u64 = 30 * 1000;
pub const MAX_TIMED_SEGMENTS: usize = 240;
pub const AUDIO_SAMPLE_RATE: usize = 16_000;
pub const MAX_VIDEO_FRAMES: usize = 30;
pub const MAX_VIDEO_PIXELS: u64 = 1920 * 1080;
pub const MAX_VIDEO_SIDE: u32 = 4096;
const MAX_VIDEO_SOURCE_PIXELS: u64 = 4096 * 4096;
const MAX_VIDEO_SOURCE_SIDE: u32 = 8192;
const MAX_PROBE_BYTES: usize = 256 * 1024;
const MAX_STDERR_BYTES: usize = 64 * 1024;
const MAX_PPM_HEADER_BYTES: usize = 256;
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);
const DECODE_TIMEOUT: Duration = Duration::from_secs(90);
// Intentionally excludes concat, HLS, DASH, image2 and other redirecting or
// playlist demuxers, even if a file is renamed to a supported extension.
const INPUT_FORMATS: &str = "mov,matroska,webm,avi,mpeg,mpegts,ogg,asf,wav,mp3,flac,aac,aiff";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
    Text,
    Audio,
    Video,
}

impl MediaKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Text => "text",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaInfo {
    pub kind: MediaKind,
    pub width: u32,
    pub height: u32,
    pub duration_ms: u64,
    /// Track starts relative to the container start used by ffmpeg input seek.
    pub audio_start_ms: u64,
    pub video_start_ms: u64,
    pub has_audio: bool,
    /// Known audio-track length; None uses container duration as an upper bound.
    pub audio_duration_ms: Option<u64>,
    /// Known visual-track length; None uses container duration as an upper bound.
    pub video_duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentSpec {
    /// Index within this modality, stable if a video's soundtrack is added or removed.
    pub index: u32,
    pub kind: MediaKind,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub text: Option<String>,
    /// Original UTF-8 byte interval, zero-based with an exclusive end.
    pub start_byte: Option<u64>,
    pub end_byte: Option<u64>,
    /// Original source lines, one-based and inclusive (LF and CRLF supported).
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
}

#[derive(Debug)]
pub enum PreparedMedia {
    Text { title: String, text: String },
    Audio { samples: Vec<f32> },
    Video { frames: Vec<RgbImage> },
}

/// Extension-based discovery only. Inspect/decode subsequently validate content.
/// PDF, office files, archives, playlists and URLs are not supported extractors.
pub fn kind_for_path(path: &Path) -> Option<MediaKind> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match extension.as_str() {
        "jpg" | "jpeg" | "png" | "webp" | "gif" | "bmp" | "tif" | "tiff" | "heic" | "heif" => {
            MediaKind::Image
        }
        "txt" | "md" | "markdown" => MediaKind::Text,
        "wav" | "mp3" | "flac" | "m4a" | "aac" | "ogg" | "opus" | "aif" | "aiff" | "wma" => {
            MediaKind::Audio
        }
        "mp4" | "m4v" | "mov" | "mkv" | "webm" | "avi" | "mpeg" | "mpg" | "ogv" | "wmv" | "ts"
        | "mts" | "m2ts" => MediaKind::Video,
        _ => return None,
    })
}

/// `.ts` is shared by TypeScript and MPEG transport streams. Only inspect a
/// bounded prefix; unreadable files remain actionable errors, never exclusions.
/// Binary/corrupt inputs still reach ffprobe and its normal error reporting.
pub fn is_typescript_source(path: &Path) -> Result<bool> {
    if !path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ts"))
    {
        return Ok(false);
    }
    let mut prefix = Vec::new();
    fs::File::open(path)
        .with_context(|| format!("Opening ambiguous .ts file {path:?}"))?
        .take(8192)
        .read_to_end(&mut prefix)
        .with_context(|| format!("Reading ambiguous .ts file {path:?}"))?;
    // TS, M2TS and error-corrected TS packet strides. Require several headers,
    // including a nonreserved adaptation-field-control value.
    for stride in [188, 192, 204] {
        for offset in 0..stride {
            if (0..5).all(|packet| {
                let at = offset + packet * stride;
                prefix.get(at) == Some(&0x47)
                    && prefix.get(at + 3).is_some_and(|byte| byte & 0x30 != 0)
            }) {
                return Ok(false);
            }
        }
    }
    // A prefix can end inside a UTF-8 character; validate its complete portion.
    let text = match std::str::from_utf8(&prefix) {
        Ok(text) => text,
        Err(error) if error.error_len().is_none() => {
            std::str::from_utf8(&prefix[..error.valid_up_to()])?
        }
        Err(_) => return Ok(false),
    };
    Ok(text.chars().all(|c| !c.is_control() || c.is_whitespace()))
}

pub fn inspect(path: &Path) -> Result<MediaInfo> {
    inspect_with_cancel(path, &|| false)
}
pub(crate) fn inspect_with_cancel(path: &Path, cancelled: &impl Fn() -> bool) -> Result<MediaInfo> {
    ensure!(!cancelled(), "Media inspection cancelled");
    let kind =
        kind_for_path(path).with_context(|| format!("Unsupported local media file {path:?}"))?;
    ensure!(!is_typescript_source(path)?, "TypeScript .ts source is not supported for indexing; supported text extensions are .txt, .md and .markdown");
    match kind {
        MediaKind::Image => {
            crate::initialize_image_decoders();
            let image = crate::preprocess::decode_image(path)?;
            Ok(MediaInfo {
                kind,
                width: image.width(),
                height: image.height(),
                duration_ms: 0,
                audio_start_ms: 0,
                video_start_ms: 0,
                has_audio: false,
                audio_duration_ms: None,
                video_duration_ms: None,
            })
        }
        MediaKind::Text => {
            read_text(path)?;
            Ok(MediaInfo {
                kind,
                width: 0,
                height: 0,
                duration_ms: 0,
                audio_start_ms: 0,
                video_start_ms: 0,
                has_audio: false,
                audio_duration_ms: None,
                video_duration_ms: None,
            })
        }
        MediaKind::Audio | MediaKind::Video => probe(path, kind, cancelled).map(|probe| probe.info),
    }
}

pub fn segments(path: &Path, info: &MediaInfo) -> Result<Vec<SegmentSpec>> {
    validate_path_kind(path, info)?;
    match info.kind {
        MediaKind::Text => bail!(
            "Text extraction requires the real model tokenizer; use segments_with_token_counter"
        ),
        MediaKind::Image => Ok(vec![SegmentSpec {
            index: 0,
            kind: MediaKind::Image,
            start_ms: None,
            end_ms: None,
            text: None,
            start_byte: None,
            end_byte: None,
            start_line: None,
            end_line: None,
        }]),
        MediaKind::Audio | MediaKind::Video => timed_segments(info),
    }
}

/// Plan text passages using actual formatted model-input token counts. Other
/// modalities keep their existing model-free segment planning.
pub fn segments_with_token_counter<F>(
    path: &Path,
    info: &MediaInfo,
    count_tokens: F,
) -> Result<Vec<SegmentSpec>>
where
    F: FnMut(&str, &str) -> Result<usize>,
{
    validate_path_kind(path, info)?;
    if info.kind != MediaKind::Text {
        return segments(path, info);
    }
    let title = path
        .file_name()
        .context("Document filename is missing")?
        .to_string_lossy();
    let markdown = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("md") || extension.eq_ignore_ascii_case("markdown")
        });
    text_segments_with_token_counter(&title, &read_text(path)?, markdown, count_tokens)
}

pub fn decode_segment(path: &Path, info: &MediaInfo, spec: &SegmentSpec) -> Result<PreparedMedia> {
    decode_segment_with_cancel(path, info, spec, &|| false)
}
pub(crate) fn decode_segment_with_cancel(
    path: &Path,
    info: &MediaInfo,
    spec: &SegmentSpec,
    cancelled: &impl Fn() -> bool,
) -> Result<PreparedMedia> {
    ensure!(!cancelled(), "Media decoding cancelled");
    validate_path_kind(path, info)?;
    validate_segment(info, spec)?;
    match spec.kind {
        MediaKind::Image => {
            bail!("Image segments must use the native preprocess::decode_image decoder")
        }
        MediaKind::Text => {
            checked_local_path(path, MAX_TEXT_FILE_BYTES)?;
            let title = path
                .file_name()
                .context("Document filename is missing")?
                .to_string_lossy()
                .into_owned();
            let text = spec.text.clone().context("Missing document segment text")?;
            crate::preprocess::validate_document_input(&title, &text)?;
            Ok(PreparedMedia::Text { title, text })
        }
        MediaKind::Audio => {
            let path = checked_local_path(path, MAX_MEDIA_FILE_BYTES)?;
            let mut command = decode_command(&path, spec)?;
            let max_samples = segment_duration(spec)? as usize * AUDIO_SAMPLE_RATE / 1000;
            // Timestamp-based -t alone can admit a whole resampled codec frame.
            // Normalize timestamps and enforce the exact interval in sample space.
            let filter = format!("aresample=16000,atrim=end_sample={max_samples},asetpts=N/SR/TB");
            command.args([
                "-af",
                &filter,
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_f32le",
                "-f",
                "f32le",
                "pipe:1",
            ]);
            let bytes = run_decoder_with_cancel(
                &mut command,
                max_samples * 4,
                DECODE_TIMEOUT,
                "ffmpeg audio decoder",
                cancelled,
            )?;
            Ok(PreparedMedia::Audio {
                samples: parse_audio(&bytes, max_samples)?,
            })
        }
        MediaKind::Video => {
            let (width, height) = video_frame_dimensions(info.width, info.height)?;
            let filter = format!("fps=1:round=up:start_time=0,scale={width}:{height}:flags=area");
            let path = checked_local_path(path, MAX_MEDIA_FILE_BYTES)?;
            let mut command = decode_command(&path, spec)?;
            // A decoded array is supplied to the model, so its processor does not
            // sample again. Keep the pinned max(1, floor(duration * 1fps)) budget.
            let frame_limit =
                ((segment_duration(spec)? / 1000) as usize).clamp(1, MAX_VIDEO_FRAMES);
            command.args([
                "-map",
                "0:V:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                &filter,
                "-frames:v",
                &frame_limit.to_string(),
                "-threads",
                "1",
                "-c:v",
                "ppm",
                "-pix_fmt",
                "rgb24",
                "-f",
                "image2pipe",
                "pipe:1",
            ]);
            let per_frame = usize::try_from(u64::from(width) * u64::from(height) * 3)?;
            let output_limit = frame_limit * (per_frame + MAX_PPM_HEADER_BYTES);
            let bytes = run_decoder_with_cancel(
                &mut command,
                output_limit,
                DECODE_TIMEOUT,
                "ffmpeg video decoder",
                cancelled,
            )?;
            Ok(PreparedMedia::Video {
                frames: parse_frames(&bytes, info, frame_limit)?,
            })
        }
    }
}

fn validate_path_kind(path: &Path, info: &MediaInfo) -> Result<()> {
    ensure!(
        kind_for_path(path) == Some(info.kind),
        "Media kind no longer matches {path:?}; rescan the file"
    );
    Ok(())
}

fn checked_local_path(path: &Path, maximum: u64) -> Result<PathBuf> {
    // Canonical paths are absolute and cannot be interpreted as a command-line
    // option or ffmpeg protocol prefix. Never feed the user-supplied string to -i.
    let path = fs::canonicalize(path).with_context(|| format!("Resolving local file {path:?}"))?;
    let metadata =
        fs::metadata(&path).with_context(|| format!("Inspecting local file {path:?}"))?;
    ensure!(
        metadata.is_file(),
        "Media input {path:?} must be a regular local file"
    );
    ensure!(metadata.len() > 0 && metadata.len() <= maximum,
        "Media input {path:?} must contain 1..={maximum} bytes; split or reduce the file before indexing");
    Ok(path)
}

fn read_text(path: &Path) -> Result<String> {
    let path = checked_local_path(path, MAX_TEXT_FILE_BYTES)?;
    let mut bytes = Vec::new();
    fs::File::open(&path)?
        .take(MAX_TEXT_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("Reading UTF-8 document {path:?}"))?;
    ensure!(
        bytes.len() as u64 <= MAX_TEXT_FILE_BYTES,
        "Document {path:?} grew beyond the 2 MiB limit"
    );
    let text = String::from_utf8(bytes).with_context(|| {
        format!("Document {path:?} must be valid UTF-8; convert it explicitly before indexing")
    })?;
    ensure!(
        !text.trim().is_empty(),
        "Document {path:?} contains no text"
    );
    ensure!(
        !text.contains('\0'),
        "Document {path:?} contains NUL bytes; use a supported plain UTF-8 text extractor"
    );
    Ok(text)
}

fn validate_duration(duration_ms: u64) -> Result<()> {
    ensure!(
        (1..=MAX_DURATION_MS).contains(&duration_ms),
        "Media duration must be positive and at most 2 hours; split the file before indexing"
    );
    Ok(())
}

fn timed_segments(info: &MediaInfo) -> Result<Vec<SegmentSpec>> {
    validate_duration(info.duration_ms)?;
    ensure!(
        matches!(info.kind, MediaKind::Audio | MediaKind::Video),
        "Expected timed media"
    );
    if info.kind == MediaKind::Video {
        validate_video_dimensions(info.width, info.height)?;
    }
    let count = info.duration_ms.div_ceil(SEGMENT_DURATION_MS) as usize;
    ensure!(
        count <= MAX_TIMED_SEGMENTS,
        "Media exceeds the {MAX_TIMED_SEGMENTS}-interval limit"
    );
    let mut segments = Vec::with_capacity(
        count
            * if info.kind == MediaKind::Video && info.has_audio {
                2
            } else {
                1
            },
    );
    for index in 0..count {
        let (primary_start, primary_duration) = track_timing(info, info.kind)?;
        if let Some(spec) =
            planned_timed_segment(info.kind, index as u32, primary_start, primary_duration)?
        {
            segments.push(spec);
        }
        if info.kind == MediaKind::Video && info.has_audio {
            let (audio_start, audio_duration) = track_timing(info, MediaKind::Audio)?;
            if let Some(spec) =
                planned_timed_segment(MediaKind::Audio, index as u32, audio_start, audio_duration)?
            {
                segments.push(spec);
            }
        }
    }
    Ok(segments)
}

fn track_timing(info: &MediaInfo, kind: MediaKind) -> Result<(u64, u64)> {
    let (start, known_duration) = match kind {
        MediaKind::Audio => (info.audio_start_ms, info.audio_duration_ms),
        MediaKind::Video => (info.video_start_ms, info.video_duration_ms),
        _ => bail!("Expected timed media"),
    };
    let available = info
        .duration_ms
        .checked_sub(start)
        .context("Track starts after its container ends")?;
    let duration = known_duration.unwrap_or(available);
    validate_duration(duration)?;
    ensure!(
        duration <= available,
        "Track duration exceeds the media timeline; rescan the file"
    );
    Ok((start, duration))
}

// Move the penultimate/final boundary back for tiny final intervals, preserving
// full, nonoverlapping coverage and <=30-second lengths. Actual ranges are
// persisted. Audio needs >160 samples; a visual tail needs a sampling interval.
// Entire audio clips <=10ms are unsupported; an entire shorter-than-1s video
// keeps its actual duration and must still yield a decodable visual frame.
fn planned_timed_segment(
    kind: MediaKind,
    index: u32,
    offset_ms: u64,
    duration_ms: u64,
) -> Result<Option<SegmentSpec>> {
    validate_duration(duration_ms)?;
    ensure!(
        matches!(kind, MediaKind::Audio | MediaKind::Video),
        "Expected timed media"
    );
    ensure!(
        index < MAX_TIMED_SEGMENTS as u32,
        "Timed segment index exceeds the limit"
    );
    if kind == MediaKind::Audio {
        ensure!(
            duration_ms > 10,
            "Audio must be longer than 10 milliseconds for the pinned audio model"
        );
    }
    let mut start_ms = u64::from(index) * SEGMENT_DURATION_MS;
    if start_ms >= duration_ms {
        return Ok(None);
    }
    let mut end_ms = (start_ms + SEGMENT_DURATION_MS).min(duration_ms);
    let count = duration_ms.div_ceil(SEGMENT_DURATION_MS);
    let last = count - 1;
    let final_length = duration_ms - last * SEGMENT_DURATION_MS;
    let minimum_tail = if kind == MediaKind::Audio { 11 } else { 1000 };
    if count > 1 && final_length < minimum_tail {
        let boundary = duration_ms - minimum_tail;
        if u64::from(index) == last {
            start_ms = boundary;
        }
        if u64::from(index) + 1 == last {
            end_ms = boundary;
        }
    }
    ensure!(
        offset_ms <= MAX_DURATION_MS && duration_ms <= MAX_DURATION_MS - offset_ms,
        "Track timeline exceeds the 2-hour limit"
    );
    Ok(Some(SegmentSpec {
        index,
        kind,
        start_ms: Some(offset_ms + start_ms),
        end_ms: Some(offset_ms + end_ms),
        text: None,
        start_byte: None,
        end_byte: None,
        start_line: None,
        end_line: None,
    }))
}

fn segment_duration(spec: &SegmentSpec) -> Result<u64> {
    let start = spec
        .start_ms
        .context("Timed segment is missing its start timestamp")?;
    let end = spec
        .end_ms
        .context("Timed segment is missing its end timestamp")?;
    ensure!(
        end > start && end - start <= SEGMENT_DURATION_MS,
        "Segment must cover 1..=30000 milliseconds"
    );
    Ok(end - start)
}

fn validate_segment(info: &MediaInfo, spec: &SegmentSpec) -> Result<()> {
    ensure!(
        spec.kind == info.kind
            || (info.kind == MediaKind::Video && info.has_audio && spec.kind == MediaKind::Audio),
        "Segment modality does not belong to this media file"
    );
    match spec.kind {
        MediaKind::Text => {
            ensure!(
                spec.index < MAX_TEXT_SEGMENTS as u32
                    && spec.start_ms.is_none()
                    && spec.end_ms.is_none(),
                "Invalid text segment metadata"
            );
            let text = spec
                .text
                .as_deref()
                .context("Missing document segment text")?;
            ensure!(
                !text.trim().is_empty() && !text.contains('\0') && text.len() <= TEXT_SEGMENT_BYTES,
                "Text segment must contain 1..=4096 UTF-8 bytes"
            );
            let start = spec
                .start_byte
                .context("Text segment is missing its source byte offset")?;
            let end = spec
                .end_byte
                .context("Text segment is missing its source end byte offset")?;
            let start_line = spec
                .start_line
                .context("Text segment is missing its source line offset")?;
            let end_line = spec
                .end_line
                .context("Text segment is missing its source end line offset")?;
            ensure!(
                end > start
                    && end <= MAX_TEXT_FILE_BYTES
                    && end - start == text.len() as u64
                    && start_line > 0
                    && u64::from(start_line) <= start + 1
                    && end_line >= start_line
                    && (end_line - start_line) as usize
                        == text.as_bytes()[..text.len() - 1]
                            .iter()
                            .filter(|&&byte| byte == b'\n')
                            .count(),
                "Invalid text source offsets"
            );
        }
        MediaKind::Audio | MediaKind::Video => {
            validate_duration(info.duration_ms)?;
            segment_duration(spec)?;
            let (track_start, track_duration) = track_timing(info, spec.kind)?;
            ensure!(
                planned_timed_segment(spec.kind, spec.index, track_start, track_duration)?.as_ref()
                    == Some(spec),
                "Invalid timed segment metadata"
            );
        }
        MediaKind::Image => {
            ensure!(
                spec.index == 0
                    && spec.start_ms.is_none()
                    && spec.end_ms.is_none()
                    && spec.text.is_none()
                    && spec.start_byte.is_none()
                    && spec.end_byte.is_none()
                    && spec.start_line.is_none()
                    && spec.end_line.is_none(),
                "Invalid image segment metadata"
            );
        }
    }
    Ok(())
}

fn validate_video_dimensions(width: u32, height: u32) -> Result<()> {
    ensure!(width > 0 && height > 0 && width <= MAX_VIDEO_SOURCE_SIDE && height <= MAX_VIDEO_SOURCE_SIDE
        && u64::from(width) * u64::from(height) <= MAX_VIDEO_SOURCE_PIXELS,
        "Video dimensions {width}x{height} exceed the source decode limit ({MAX_VIDEO_SOURCE_PIXELS} pixels, sides <= {MAX_VIDEO_SOURCE_SIDE}); prepare a smaller video before indexing");
    Ok(())
}

// Preserve aspect ratio to integer-pixel rounding and never upscale. The
// model's own patch sizing still follows this bounded extraction step.
fn video_frame_dimensions(width: u32, height: u32) -> Result<(u32, u32)> {
    validate_video_dimensions(width, height)?;
    let scale = (MAX_VIDEO_PIXELS as f64 / (f64::from(width) * f64::from(height)))
        .sqrt()
        .min(f64::from(MAX_VIDEO_SIDE) / f64::from(width.max(height)))
        .min(1.0);
    Ok((
        (f64::from(width) * scale).floor().max(1.0) as u32,
        (f64::from(height) * scale).floor().max(1.0) as u32,
    ))
}

#[derive(Deserialize)]
struct ProbeOutput {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    format: Option<ProbeFormat>,
}
#[derive(Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    duration: Option<String>,
    start_time: Option<String>,
    #[serde(default)]
    disposition: ProbeDisposition,
    #[serde(default)]
    tags: ProbeTags,
}
#[derive(Default, Deserialize)]
struct ProbeDisposition {
    #[serde(default)]
    attached_pic: u32,
}
#[derive(Default, Deserialize)]
struct ProbeTags {
    #[serde(rename = "DURATION", alias = "duration")]
    duration: Option<String>,
}
#[derive(Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
    start_time: Option<String>,
}
struct ProbedMedia {
    info: MediaInfo,
}

fn input_options(command: &mut Command) {
    command.args([
        "-hide_banner",
        "-v",
        "error",
        "-max_alloc",
        "67108864",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        INPUT_FORMATS,
        "-probesize",
        "8388608",
        "-analyzeduration",
        "10000000",
        "-threads",
        "1",
    ]);
}

fn probe(path: &Path, kind: MediaKind, cancelled: &impl Fn() -> bool) -> Result<ProbedMedia> {
    let path = checked_local_path(path, MAX_MEDIA_FILE_BYTES)?;
    let mut command = Command::new("ffprobe");
    input_options(&mut command);
    // ffprobe does not implement ffmpeg's -nostdin option. run_decoder closes
    // stdin explicitly for both tools, including probe builds with no such flag.
    command.args(["-show_entries", "format=start_time,duration:stream=codec_type,width,height,start_time,duration:stream_disposition=attached_pic:stream_tags=DURATION",
        "-of", "json", "-i"]).arg(&path);
    let output = run_decoder_with_cancel(
        &mut command,
        MAX_PROBE_BYTES,
        PROBE_TIMEOUT,
        "ffprobe media inspector",
        cancelled,
    )?;
    parse_probe(&output, kind).with_context(|| format!("Inspecting media metadata for {path:?}"))
}

fn parse_duration(value: &str) -> Result<u64> {
    let seconds: f64 = value.parse().context("Media duration is not a number")?;
    ensure!(
        seconds.is_finite() && seconds > 0.0 && seconds <= MAX_DURATION_MS as f64 / 1000.0,
        "Media must have a finite, positive duration of at most 2 hours"
    );
    let duration = (seconds * 1000.0).ceil() as u64;
    validate_duration(duration)?;
    Ok(duration)
}

fn parse_timestamp(value: &str) -> Result<f64> {
    let seconds: f64 = value.parse().context("Media timestamp is not a number")?;
    ensure!(
        seconds.is_finite() && seconds.abs() <= 1_000_000_000.0,
        "Media timestamp is not finite or safely representable"
    );
    Ok(seconds)
}

fn duration_tag_end(value: &str) -> Result<f64> {
    let parts: Vec<_> = value.split(':').collect();
    ensure!(parts.len() == 3, "Invalid media stream DURATION tag");
    let hours: u64 = parts[0].parse().context("Invalid stream duration hours")?;
    let minutes: u64 = parts[1]
        .parse()
        .context("Invalid stream duration minutes")?;
    let seconds = parse_timestamp(parts[2])?;
    ensure!(
        hours <= 277_777 && minutes < 60 && (0.0..60.0).contains(&seconds),
        "Invalid media stream DURATION tag"
    );
    Ok(hours as f64 * 3600.0 + minutes as f64 * 60.0 + seconds)
}

fn stream_timing(stream: Option<&ProbeStream>, container_start: f64) -> Result<(u64, Option<u64>)> {
    let Some(stream) = stream else {
        return Ok((0, None));
    };
    let raw_start = stream
        .start_time
        .as_deref()
        .map(parse_timestamp)
        .transpose()?
        .unwrap_or(container_start);
    let relative_start = raw_start - container_start;
    ensure!((0.0..=MAX_DURATION_MS as f64 / 1000.0).contains(&relative_start),
        "Media track start falls outside the supported container timeline; remux the file explicitly");
    let start_ms = (relative_start * 1000.0 + 0.000_000_1).floor() as u64;
    let raw_end = if let Some(value) = &stream.duration {
        let duration = parse_timestamp(value)?;
        ensure!(
            duration > 0.0 && duration <= MAX_DURATION_MS as f64 / 1000.0,
            "Invalid or overlong stream duration"
        );
        Some(raw_start + duration)
    } else if let Some(value) = &stream.tags.duration {
        // Matroska/WebM DURATION tags contain the track's absolute end
        // timestamp, unlike ffprobe stream.duration (a duration). Subtracting
        // start is essential for delayed tracks, otherwise content is omitted.
        Some(duration_tag_end(value)?)
    } else {
        None
    };
    let duration_ms = raw_end.map(|end| -> Result<u64> {
        let relative_end = end - container_start;
        ensure!(relative_end > relative_start && relative_end <= MAX_DURATION_MS as f64 / 1000.0,
            "Media track end falls outside the supported container timeline; remux or split the file explicitly");
        let end_ms = (relative_end * 1000.0 - 0.000_000_1).ceil() as u64;
        let duration = end_ms.checked_sub(start_ms).context("Media track ends before it starts")?;
        validate_duration(duration)?;
        Ok(duration)
    }).transpose()?;
    Ok((start_ms, duration_ms))
}

fn parse_probe(bytes: &[u8], kind: MediaKind) -> Result<ProbedMedia> {
    ensure!(
        bytes.len() <= MAX_PROBE_BYTES,
        "Media metadata exceeded the probe size limit"
    );
    ensure!(
        matches!(kind, MediaKind::Audio | MediaKind::Video),
        "Expected audio or video metadata"
    );
    let output: ProbeOutput =
        serde_json::from_slice(bytes).context("ffprobe returned malformed metadata JSON")?;
    ensure!(
        !output.streams.is_empty() && output.streams.len() <= 32,
        "Media must have 1..=32 streams"
    );
    let audio = output
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("audio"));
    let video = output.streams.iter().find(|stream| {
        stream.codec_type.as_deref() == Some("video") && stream.disposition.attached_pic == 0
    });
    let selected = match kind {
        MediaKind::Audio => audio.context("Audio file has no audio stream")?,
        MediaKind::Video => {
            video.context("Video file has no visual stream (cover art is not a video)")?
        }
        _ => unreachable!(),
    };
    let container_start = output
        .format
        .as_ref()
        .and_then(|format| format.start_time.as_deref())
        .map(parse_timestamp)
        .transpose()?
        .unwrap_or(0.0);
    let (audio_start_ms, audio_duration_ms) = stream_timing(audio, container_start)?;
    let (video_start_ms, video_duration_ms) = stream_timing(video, container_start)?;
    let all_track_ends_known = match kind {
        MediaKind::Audio => audio_duration_ms.is_some(),
        MediaKind::Video => {
            video_duration_ms.is_some() && (audio.is_none() || audio_duration_ms.is_some())
        }
        _ => false,
    };
    let mut duration_ms = 0;
    if !all_track_ends_known {
        // Some containers report duration as an absolute end timestamp when
        // their origin is nonzero. Without actual track ends this is ambiguous.
        ensure!(container_start == 0.0,
            "Nonzero container origin with unknown track duration is ambiguous; remux the file to a zero-based timeline before indexing");
        if let Some(value) = output
            .format
            .as_ref()
            .and_then(|format| format.duration.as_deref())
        {
            duration_ms = parse_duration(value)?;
        }
    }
    let relevant = if kind == MediaKind::Audio {
        vec![(audio_start_ms, audio_duration_ms)]
    } else {
        vec![
            (audio_start_ms, audio_duration_ms),
            (video_start_ms, video_duration_ms),
        ]
    };
    for (start, duration) in relevant {
        if let Some(duration) = duration {
            duration_ms = duration_ms.max(start + duration);
        }
    }
    validate_duration(duration_ms).context("Media has no usable duration metadata")?;
    let (width, height) = if kind == MediaKind::Video {
        let width = selected.width.context("Video width is missing")?;
        let height = selected.height.context("Video height is missing")?;
        validate_video_dimensions(width, height)?;
        (width, height)
    } else {
        (0, 0)
    };
    Ok(ProbedMedia {
        info: MediaInfo {
            kind,
            width,
            height,
            duration_ms,
            audio_start_ms,
            video_start_ms,
            has_audio: audio.is_some(),
            audio_duration_ms,
            video_duration_ms,
        },
    })
}

fn seconds(milliseconds: u64) -> String {
    format!("{}.{:03}", milliseconds / 1000, milliseconds % 1000)
}

fn decode_command(path: &Path, spec: &SegmentSpec) -> Result<Command> {
    ensure!(
        path.is_absolute(),
        "Decoder input must be an absolute local path"
    );
    let mut command = Command::new("ffmpeg");
    // The shared preparation pool owns parallelism; each decoder/filter graph
    // must not create a second machine-wide worker pool.
    command.args(["-nostdin", "-xerror", "-filter_threads", "1"]);
    input_options(&mut command);
    // Accurate input seek is ffmpeg's default for transcoding. Disable automatic
    // orientation so the pinned native no-EXIF-rotation pixel policy is preserved.
    command
        .args([
            "-ss",
            &seconds(spec.start_ms.context("Segment start is missing")?),
            "-noautorotate",
            "-i",
        ])
        .arg(path)
        .args([
            "-t",
            &seconds(segment_duration(spec)?),
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
        ]);
    Ok(command)
}

fn parse_audio(bytes: &[u8], max_samples: usize) -> Result<Vec<f32>> {
    ensure!(bytes.len() > 160 * 4 && bytes.len() % 4 == 0, "Audio decoder returned <=160 or incomplete float32 samples; container duration may overstate this soundtrack, so trim or split it explicitly");
    ensure!(
        bytes.len() / 4 <= max_samples,
        "Audio decoder exceeded the segment sample limit"
    );
    let mut samples = Vec::with_capacity(bytes.len() / 4);
    for sample in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
        ensure!(
            value.is_finite(),
            "Audio decoder returned a non-finite sample"
        );
        samples.push(value);
    }
    Ok(samples)
}

fn ppm_token<'a>(bytes: &'a [u8], cursor: &mut usize, header_start: usize) -> Result<&'a [u8]> {
    while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
        *cursor += 1;
        ensure!(
            *cursor - header_start <= MAX_PPM_HEADER_BYTES,
            "Video frame header is too large"
        );
    }
    let start = *cursor;
    while bytes
        .get(*cursor)
        .is_some_and(|byte| !byte.is_ascii_whitespace())
    {
        *cursor += 1;
        ensure!(
            *cursor - header_start <= MAX_PPM_HEADER_BYTES,
            "Video frame header is too large"
        );
    }
    ensure!(
        *cursor > start && *cursor < bytes.len(),
        "Video decoder returned an incomplete frame header"
    );
    Ok(&bytes[start..*cursor])
}

fn ppm_number(bytes: &[u8]) -> Result<u32> {
    ensure!(
        !bytes.is_empty() && bytes.iter().all(u8::is_ascii_digit),
        "Video frame header contains an invalid number"
    );
    std::str::from_utf8(bytes)?
        .parse()
        .context("Video frame header number is too large")
}

fn parse_frames(bytes: &[u8], info: &MediaInfo, frame_limit: usize) -> Result<Vec<RgbImage>> {
    ensure!(
        (1..=MAX_VIDEO_FRAMES).contains(&frame_limit),
        "Invalid video frame budget"
    );
    let mut cursor = 0;
    let mut frames = Vec::new();
    while cursor < bytes.len() {
        ensure!(
            frames.len() < frame_limit,
            "Video decoder exceeded the frame limit"
        );
        let start = cursor;
        ensure!(
            ppm_token(bytes, &mut cursor, start)? == b"P6",
            "Video decoder must return binary RGB PPM frames"
        );
        let width = ppm_number(ppm_token(bytes, &mut cursor, start)?)?;
        let height = ppm_number(ppm_token(bytes, &mut cursor, start)?)?;
        ensure!(
            ppm_number(ppm_token(bytes, &mut cursor, start)?)? == 255,
            "Video decoder must return 8-bit RGB frames"
        );
        validate_video_dimensions(width, height)?;
        ensure!((width, height) == video_frame_dimensions(info.width, info.height)?,
            "Video dimensions changed during decode ({width}x{height}); prepare a constant-dimension video before indexing");
        // Exactly one separator follows maxval. Consuming all whitespace here
        // would eat valid pixel bytes, including a black/white first pixel.
        ensure!(
            bytes.get(cursor).is_some_and(u8::is_ascii_whitespace),
            "Video frame header is not terminated"
        );
        cursor += 1;
        let length = usize::try_from(u64::from(width) * u64::from(height) * 3)?;
        let end = cursor
            .checked_add(length)
            .context("Video frame size overflow")?;
        let pixels = bytes
            .get(cursor..end)
            .context("Video decoder returned an incomplete RGB frame")?;
        frames.push(
            RgbImage::from_raw(width, height, pixels.to_vec())
                .context("Invalid RGB video frame buffer")?,
        );
        cursor = end;
    }
    ensure!(!frames.is_empty(), "Video decoder returned no visual frames; container duration may overstate this visual track, so trim or split it explicitly");
    Ok(frames)
}

#[cfg(unix)]
fn run_decoder_with_cancel(
    command: &mut Command,
    stdout_limit: usize,
    timeout: Duration,
    label: &str,
    cancelled: &impl Fn() -> bool,
) -> Result<Vec<u8>> {
    use std::os::{fd::AsRawFd, unix::process::CommandExt};
    struct ChildGuard {
        child: std::process::Child,
        complete: bool,
    }
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if !self.complete {
                // Every decoder owns a process group, so descendants cannot keep
                // stdout/stderr alive after timeout. Never signal the app's group.
                unsafe {
                    libc::kill(-(self.child.id() as i32), libc::SIGKILL);
                }
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }
    fn nonblocking(fd: std::os::fd::RawFd) -> Result<()> {
        // F_GETFL/F_SETFL touch only the owned subprocess pipe descriptor.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        ensure!(
            flags >= 0,
            "Cannot inspect decoder pipe: {}",
            std::io::Error::last_os_error()
        );
        ensure!(
            unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } >= 0,
            "Cannot make decoder pipe nonblocking: {}",
            std::io::Error::last_os_error()
        );
        Ok(())
    }
    fn read_pipe(reader: &mut impl Read, output: &mut Vec<u8>, maximum: usize) -> Result<bool> {
        let mut buffer = [0_u8; 16 * 1024];
        // Bounded work per poll prevents a noisy pipe starving the other pipe or
        // the wall-clock timeout. At most one excess byte is ever read/retained.
        let length = buffer
            .len()
            .min(maximum.saturating_sub(output.len()).saturating_add(1));
        match reader.read(&mut buffer[..length]) {
            Ok(0) => Ok(true),
            Ok(count) => {
                ensure!(
                    count <= maximum.saturating_sub(output.len()),
                    "Decoder output exceeded the {maximum}-byte limit"
                );
                let needed = output.len() + count;
                if needed > output.capacity() {
                    let capacity = needed.max(output.capacity().saturating_mul(2)).min(maximum);
                    output.reserve_exact(capacity - output.len());
                }
                output.extend_from_slice(&buffer[..count]);
                Ok(false)
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) =>
            {
                Ok(false)
            }
            Err(error) => Err(error).context("Reading decoder pipe"),
        }
    }
    ensure!(!timeout.is_zero(), "Decoder timeout must be positive");
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    // Bound decoder virtual memory as well as the pipes; malformed codecs can
    // otherwise allocate substantial state before writing their first frame.
    #[cfg(target_os = "linux")]
    unsafe {
        command.pre_exec(|| {
            let mut limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::getrlimit(libc::RLIMIT_AS, &mut limit) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            limit.rlim_cur = limit.rlim_cur.min(2 * 1024 * 1024 * 1024);
            limit.rlim_max = limit.rlim_max.min(2 * 1024 * 1024 * 1024);
            if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command.spawn().map_err(|error| {
        let message = if error.kind() == std::io::ErrorKind::NotFound {
            format!("Cannot start {label}; install ffmpeg and ffprobe and ensure they are on PATH")
        } else {
            format!(
                "Cannot start {label}; check executable permissions and subprocess resource limits"
            )
        };
        anyhow::Error::new(error).context(message)
    })?;
    let mut guard = ChildGuard {
        child,
        complete: false,
    };
    let mut stdout = guard
        .child
        .stdout
        .take()
        .context("Missing decoder stdout pipe")?;
    let mut stderr = guard
        .child
        .stderr
        .take()
        .context("Missing decoder stderr pipe")?;
    nonblocking(stdout.as_raw_fd())?;
    nonblocking(stderr.as_raw_fd())?;
    let mut output = Vec::new();
    let mut errors = Vec::new();
    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut status = None;
    let started = Instant::now();
    loop {
        ensure!(!cancelled(), "{label} cancelled");
        ensure!(
            started.elapsed() < timeout,
            "{label} exceeded its {}-second time limit; split or simplify this media file",
            timeout.as_secs()
        );
        if !stdout_done {
            stdout_done = read_pipe(&mut stdout, &mut output, stdout_limit)
                .with_context(|| format!("{label} stdout"))?;
        }
        if !stderr_done {
            stderr_done = read_pipe(&mut stderr, &mut errors, MAX_STDERR_BYTES)
                .with_context(|| format!("{label} stderr"))?;
        }
        if status.is_none() {
            status = guard
                .child
                .try_wait()
                .with_context(|| format!("Waiting for {label}"))?;
        }
        if let Some(status) = status {
            if stdout_done && stderr_done {
                guard.complete = true;
                let detail: String = String::from_utf8_lossy(&errors)
                    .chars()
                    .filter(|character| {
                        !character.is_control() || matches!(*character, '\n' | '\t')
                    })
                    .collect();
                ensure!(
                    status.success(),
                    "{label} failed ({status}): {}",
                    detail.trim()
                );
                // With -v error, any stderr output is a decode/probe error,
                // even when ffmpeg exits successfully after concealing damage.
                ensure!(
                    errors.is_empty(),
                    "{label} reported a decoding error: {}",
                    detail.trim()
                );
                return Ok(output);
            }
        }
        // 16 KiB/poll still gives ample throughput for the bounded 30-frame
        // segments, while never spinning continuously on an idle decoder.
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[cfg(not(unix))]
fn run_decoder_with_cancel(
    _command: &mut Command,
    _stdout_limit: usize,
    _timeout: Duration,
    _label: &str,
    _cancelled: &impl Fn() -> bool,
) -> Result<Vec<u8>> {
    bail!("Bounded local ffmpeg decoding currently requires a Unix platform")
}

#[cfg(test)]
fn run_decoder(
    command: &mut Command,
    stdout_limit: usize,
    timeout: Duration,
    label: &str,
) -> Result<Vec<u8>> {
    run_decoder_with_cancel(command, stdout_limit, timeout, label, &|| false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn cancelled_decoder_is_reaped_promptly() {
        let started = Instant::now();
        let error = run_decoder_with_cancel(
            Command::new("sh").args(["-c", "sleep 20"]),
            32,
            Duration::from_secs(30),
            "cancel test",
            &|| started.elapsed() > Duration::from_millis(50),
        )
        .unwrap_err();
        assert!(error.to_string().contains("cancelled"));
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    fn info(kind: MediaKind, duration_ms: u64) -> MediaInfo {
        MediaInfo {
            kind,
            width: if kind == MediaKind::Video { 2 } else { 0 },
            height: if kind == MediaKind::Video { 2 } else { 0 },
            duration_ms,
            audio_start_ms: 0,
            video_start_ms: 0,
            has_audio: kind == MediaKind::Audio,
            audio_duration_ms: if kind == MediaKind::Audio {
                Some(duration_ms)
            } else {
                None
            },
            video_duration_ms: if kind == MediaKind::Video {
                Some(duration_ms)
            } else {
                None
            },
        }
    }

    #[test]
    fn discovery_is_explicit_and_case_insensitive() {
        for (name, kind) in [
            ("photo.JPEG", MediaKind::Image),
            ("doc.MARKDOWN", MediaKind::Text),
            ("clip.M4A", MediaKind::Audio),
            ("clip.MP4", MediaKind::Video),
        ] {
            assert_eq!(kind_for_path(Path::new(name)), Some(kind));
            assert_eq!(
                serde_json::to_string(&kind).unwrap(),
                format!("\"{}\"", kind.as_str())
            );
        }
        for name in [
            "private.pdf",
            "office.docx",
            "feed.m3u8",
            "clip.mp4.url",
            "README",
            "archive.zip",
        ] {
            assert_eq!(kind_for_path(Path::new(name)), None);
        }
    }

    #[test]
    fn ambiguous_ts_checks_content_and_propagates_read_errors() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["index.ts", "types.d.ts", "INDEX.TS"] {
            let path = dir.path().join(name);
            fs::write(&path, "export interface Item { title: string }\n").unwrap();
            assert!(is_typescript_source(&path).unwrap());
            assert!(inspect(&path)
                .unwrap_err()
                .to_string()
                .contains("TypeScript"));
        }
        let path = dir.path().join("transport.ts");
        for stride in [188, 192, 204] {
            let mut bytes = vec![0; stride * 6];
            for packet in 0..6 {
                bytes[packet * stride] = 0x47;
                bytes[packet * stride + 3] = 0x10;
            }
            fs::write(&path, bytes).unwrap();
            assert!(!is_typescript_source(&path).unwrap());
        }
        fs::write(&path, [0xff, 0x00, 0x47]).unwrap();
        assert!(!is_typescript_source(&path).unwrap());
        assert!(is_typescript_source(&dir.path().join("missing.ts")).is_err());
    }

    #[test]
    fn video_resize_preserves_small_frames_and_bounds_large_landscape_portrait() {
        for (width, height) in [(2, 2), (1920, 1080), (1080, 1920)] {
            assert_eq!(
                video_frame_dimensions(width, height).unwrap(),
                (width, height)
            );
        }
        for (width, height) in [
            (2880, 1808),
            (2784, 1856),
            (3840, 2160),
            (2160, 3840),
            (8192, 1024),
        ] {
            let (w, h) = video_frame_dimensions(width, height).unwrap();
            assert!(w <= width && h <= height && w <= MAX_VIDEO_SIDE && h <= MAX_VIDEO_SIDE);
            assert!(u64::from(w) * u64::from(h) <= MAX_VIDEO_PIXELS);
            assert!(
                (f64::from(w) / f64::from(h) - f64::from(width) / f64::from(height)).abs() < 0.02
            );
        }
    }

    // Explicit integration check: requires system ffmpeg/ffprobe, no model or
    // user catalog. Missing tools fail visibly when this test is requested.
    #[test]
    #[ignore = "requires ffmpeg and ffprobe"]
    fn native_media_resizes_video_and_bounds_audio_samples() {
        fn generate(path: &Path, input: &str, duration: &str, codec: &str) {
            let output = Command::new("ffmpeg")
                .args([
                    "-v", "error", "-nostdin", "-f", "lavfi", "-i", input, "-t", duration,
                    "-threads", "1", "-c", codec,
                ])
                .arg(path)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let dir = tempfile::tempdir().unwrap();
        let drift = dir.path().join("timestamp-drift.m4a");
        let encoded = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=31",
                "-af",
                "asetpts=0.99*PTS",
                "-c:a",
                "aac",
            ])
            .arg(&drift)
            .output()
            .unwrap();
        assert!(encoded.status.success());
        let metadata = inspect(&drift).unwrap();
        let spec = segments(&drift, &metadata).unwrap().remove(0);
        // Reproduce the original overflow with timestamp-only output trimming.
        let mut legacy = decode_command(&drift, &spec).unwrap();
        legacy.args([
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]);
        let legacy = legacy.output().unwrap();
        assert!(legacy.status.success());
        assert!(legacy.stdout.len() > 480000 * 4);
        let PreparedMedia::Audio { samples } = decode_segment(&drift, &metadata, &spec).unwrap()
        else {
            panic!("expected audio")
        };
        assert_eq!(samples.len(), 480000);
        for (name, input, duration, codec) in [
            ("wide.mp4", "color=c=red:s=2880x1808:r=1", "1", "mpeg4"),
            ("portrait.mp4", "color=c=blue:s=2160x3840:r=1", "1", "mpeg4"),
            (
                "transport.ts",
                "color=c=red:s=64x48:r=25",
                "1",
                "mpeg2video",
            ),
            (
                "short.m4a",
                "sine=frequency=440:sample_rate=44100",
                "9.888",
                "aac",
            ),
            (
                "multi.m4a",
                "sine=frequency=440:sample_rate=48000",
                "30.123",
                "aac",
            ),
            (
                "exact.wav",
                "sine=frequency=440:sample_rate=16000",
                "60.001",
                "pcm_f32le",
            ),
        ] {
            let path = dir.path().join(name);
            generate(&path, input, duration, codec);
            assert!(!is_typescript_source(&path).unwrap());
            let metadata = inspect(&path).unwrap();
            let specs = segments(&path, &metadata).unwrap();
            let mut audio = Vec::new();
            for spec in specs {
                match decode_segment(&path, &metadata, &spec).unwrap() {
                    PreparedMedia::Video { frames } => {
                        assert!(!frames.is_empty());
                        for frame in frames {
                            assert_eq!(
                                frame.dimensions(),
                                video_frame_dimensions(metadata.width, metadata.height).unwrap()
                            );
                        }
                    }
                    PreparedMedia::Audio { samples } => {
                        assert!(samples.len() <= segment_duration(&spec).unwrap() as usize * 16);
                        audio.extend(samples);
                    }
                    _ => panic!("unexpected modality"),
                }
            }
            if name == "exact.wav" {
                // Concatenating the adjusted 30-second boundaries must retain
                // every sample exactly once, not just satisfy per-segment limits.
                let output = Command::new("ffmpeg")
                    .args(["-v", "error", "-i"])
                    .arg(&path)
                    .args(["-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"])
                    .output()
                    .unwrap();
                assert!(output.status.success());
                assert_eq!(audio, parse_audio(&output.stdout, 960016).unwrap());
                assert_eq!(audio.len(), 960016);
            }
        }
    }

    #[test]
    fn text_segments_preserve_utf8_coverage_and_bounded_overlap() {
        let text = "aé中🦊\n".repeat(3000);
        let segments = text_segments(&text).unwrap();
        assert!(segments.len() > 1);
        let mut start = 0;
        for (index, segment) in segments.iter().enumerate() {
            assert_eq!(segment.kind, MediaKind::Text);
            assert_eq!(segment.index, index as u32);
            assert!(segment.start_ms.is_none() && segment.end_ms.is_none());
            let content = segment.text.as_deref().unwrap();
            assert!(!content.is_empty() && content.len() <= TEXT_SEGMENT_BYTES);
            let end = start + content.len();
            assert_eq!(content, &text[start..end]);
            if end == text.len() {
                assert_eq!(index, segments.len() - 1);
                break;
            }
            let mut next = end - TEXT_OVERLAP_BYTES;
            while !text.is_char_boundary(next) {
                next += 1;
            }
            assert!(end - next <= TEXT_OVERLAP_BYTES);
            start = next;
        }
    }

    // Explicit deterministic fake tokenizer for media-only unit tests. Actual
    // indexing always calls the runtime tokenizer through the counted API.
    fn text_segments(text: &str) -> Result<Vec<SegmentSpec>> {
        text_segments_with_token_counter("test.txt", text, false, |title, text| {
            Ok(title.len() + text.len() + 16)
        })
    }

    #[test]
    fn text_limits_and_empty_documents_are_rejected() {
        assert!(text_segments("").is_err());
        assert!(text_segments(" \n\t").is_err());
        let after_blank = text_segments(&(" ".repeat(8192) + "hello")).unwrap();
        assert!(!after_blank.is_empty());
        assert!(after_blank.iter().all(|segment| !segment
            .text
            .as_deref()
            .unwrap()
            .trim()
            .is_empty()));
        assert_eq!(
            text_segments(&"x".repeat(TEXT_SEGMENT_BYTES))
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            text_segments(&"x".repeat(TEXT_SEGMENT_BYTES + 1))
                .unwrap()
                .len(),
            2
        );
        let largest = "x".repeat(MAX_TEXT_FILE_BYTES as usize);
        assert!(text_segments(&largest).unwrap().len() < MAX_TEXT_SEGMENTS);
        assert!(text_segments(&(largest + "x")).is_err());
    }

    #[test]
    fn local_text_requires_regular_bounded_utf8_data() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("note.md");
        fs::write(&path, "Local document 🦊").unwrap();
        let inspected = inspect(&path).unwrap();
        assert!(segments(&path, &inspected)
            .unwrap_err()
            .to_string()
            .contains("real model tokenizer"));
        let segment = segments_with_token_counter(&path, &inspected, |title, text| {
            Ok(title.len() + text.len() + 16)
        })
        .unwrap()
        .remove(0);
        let PreparedMedia::Text { title, text } =
            decode_segment(&path, &inspected, &segment).unwrap()
        else {
            panic!("expected text")
        };
        assert_eq!(title, "note.md");
        assert_eq!(text, "Local document 🦊");
        fs::write(&path, [0xff, 0xfe]).unwrap();
        assert!(inspect(&path).unwrap_err().to_string().contains("UTF-8"));
        fs::write(&path, b"hello\0binary").unwrap();
        assert!(inspect(&path).unwrap_err().to_string().contains("NUL"));
        fs::File::create(&path)
            .unwrap()
            .set_len(MAX_TEXT_FILE_BYTES + 1)
            .unwrap();
        assert!(inspect(&path).is_err());
        assert!(checked_local_path(directory.path(), MAX_TEXT_FILE_BYTES).is_err());
    }

    #[test]
    fn timed_segments_adjust_tiny_tails_without_overlap_and_stop_at_two_hours() {
        let timed = timed_segments(&info(MediaKind::Audio, 60_001)).unwrap();
        assert_eq!(timed.len(), 3);
        assert_eq!(
            (timed[0].start_ms, timed[0].end_ms),
            (Some(0), Some(30_000))
        );
        assert_eq!(
            (timed[1].start_ms, timed[1].end_ms),
            (Some(30_000), Some(59_990))
        );
        assert_eq!(
            (timed[2].start_ms, timed[2].end_ms),
            (Some(59_990), Some(60_001))
        );
        assert_eq!(
            timed_segments(&info(MediaKind::Audio, MAX_DURATION_MS))
                .unwrap()
                .len(),
            240
        );
        assert!(timed_segments(&info(MediaKind::Audio, 0)).is_err());
        assert!(timed_segments(&info(MediaKind::Audio, 10)).is_err());
        assert_eq!(
            timed_segments(&info(MediaKind::Audio, 11)).unwrap()[0].start_ms,
            Some(0)
        );
        let tail = timed_segments(&info(MediaKind::Video, 30_001)).unwrap();
        assert_eq!(
            (tail[1].start_ms, tail[1].end_ms),
            (Some(29_001), Some(30_001))
        );
        assert!(timed_segments(&info(MediaKind::Audio, MAX_DURATION_MS + 1)).is_err());
    }

    #[test]
    fn adjusted_boundaries_preserve_every_millisecond_once() {
        for kind in [MediaKind::Audio, MediaKind::Video] {
            for duration in [
                11,
                999,
                1000,
                29_999,
                30_000,
                30_001,
                30_010,
                30_011,
                30_999,
                31_000,
                60_001,
                MAX_DURATION_MS,
            ] {
                let metadata = info(kind, duration);
                let segments = timed_segments(&metadata).unwrap();
                let mut previous_end = 0;
                for spec in &segments {
                    assert_eq!(spec.start_ms, Some(previous_end));
                    assert!(segment_duration(spec).unwrap() <= SEGMENT_DURATION_MS);
                    if kind == MediaKind::Audio {
                        assert!(segment_duration(spec).unwrap() > 10);
                    }
                    validate_segment(&metadata, spec).unwrap();
                    previous_end = spec.end_ms.unwrap();
                }
                assert_eq!(previous_end, duration);
            }
        }
    }

    #[test]
    fn soundtrack_is_separate_stable_and_clipped_to_known_length() {
        let mut video = info(MediaKind::Video, 61_000);
        let silent = timed_segments(&video).unwrap();
        video.has_audio = true;
        video.audio_duration_ms = Some(45_000);
        let tracks = timed_segments(&video).unwrap();
        assert_eq!(tracks.len(), 5);
        assert_eq!(tracks[0], silent[0]);
        assert_eq!(tracks[1].kind, MediaKind::Audio);
        assert_eq!(tracks[1].index, tracks[0].index);
        assert_eq!(tracks[2], silent[1]);
        assert_eq!(tracks[3].end_ms, Some(45_000));
        assert_eq!(tracks[4], silent[2]);
        video.audio_duration_ms = None;
        assert_eq!(timed_segments(&video).unwrap().len(), 6);
        video.duration_ms = MAX_DURATION_MS;
        video.video_duration_ms = Some(MAX_DURATION_MS);
        assert_eq!(timed_segments(&video).unwrap().len(), 480);
    }

    #[test]
    fn long_soundtracks_continue_after_the_last_visual_segment() {
        let mut metadata = info(MediaKind::Video, 61_000);
        metadata.video_duration_ms = Some(31_000);
        metadata.has_audio = true;
        metadata.audio_duration_ms = Some(61_000);
        let segments = timed_segments(&metadata).unwrap();
        assert_eq!(
            segments
                .iter()
                .filter(|spec| spec.kind == MediaKind::Video)
                .count(),
            2
        );
        assert_eq!(
            segments
                .iter()
                .filter(|spec| spec.kind == MediaKind::Audio)
                .count(),
            3
        );
        let last = segments.last().unwrap();
        assert_eq!(last.kind, MediaKind::Audio);
        assert_eq!(last.end_ms, Some(61_000));
    }

    #[test]
    fn forged_segments_cannot_escape_decoder_budgets() {
        let video = info(MediaKind::Video, 60_000);
        let valid = timed_segments(&video).unwrap().remove(0);
        assert!(validate_segment(&video, &valid).is_ok());
        for invalid in [
            SegmentSpec {
                end_ms: Some(30_001),
                ..valid.clone()
            },
            SegmentSpec {
                end_ms: Some(0),
                ..valid.clone()
            },
            SegmentSpec {
                start_ms: None,
                ..valid.clone()
            },
            SegmentSpec {
                index: 240,
                ..valid.clone()
            },
            SegmentSpec {
                start_ms: Some(1),
                ..valid.clone()
            },
            SegmentSpec {
                kind: MediaKind::Audio,
                ..valid.clone()
            },
            SegmentSpec {
                text: Some("unexpected".into()),
                ..valid.clone()
            },
        ] {
            assert!(validate_segment(&video, &invalid).is_err());
        }
    }

    #[test]
    fn duration_parser_rejects_nonfinite_invalid_and_overlong_values() {
        assert_eq!(parse_duration("0.000001").unwrap(), 1);
        assert_eq!(parse_duration("7200").unwrap(), MAX_DURATION_MS);
        for duration in ["NaN", "inf", "-1", "0", "N/A", "7200.0001", "1e100"] {
            assert!(parse_duration(duration).is_err(), "accepted {duration}");
        }
        assert_eq!(seconds(30_007), "30.007");
    }

    #[test]
    fn probe_metadata_requires_real_streams_and_valid_dimensions() {
        let metadata = br#"{"streams":[{"codec_type":"video","width":64,"height":48,"duration":"61.0"},{"codec_type":"audio","duration":"45.0"}],"format":{"duration":"61.0"}}"#;
        let video = parse_probe(metadata, MediaKind::Video).unwrap().info;
        assert_eq!(video.width, 64);
        assert_eq!(video.duration_ms, 61_000);
        assert!(video.has_audio);
        assert_eq!(video.audio_duration_ms, Some(45_000));
        let tags = br#"{"streams":[{"codec_type":"audio","tags":{"DURATION":"00:01:02.123456"}}],"format":{"duration":"62.124"}}"#;
        assert_eq!(
            parse_probe(tags, MediaKind::Audio)
                .unwrap()
                .info
                .audio_duration_ms,
            Some(62_124)
        );
        let unknown = br#"{"streams":[{"codec_type":"audio"}],"format":{"duration":"60.0"}}"#;
        assert_eq!(
            parse_probe(unknown, MediaKind::Audio)
                .unwrap()
                .info
                .audio_duration_ms,
            None
        );
        for invalid in [
            br#"{}"#.as_slice(),
            br#"{"streams":[{"codec_type":"audio"}]}"#,
            br#"{"streams":[{"codec_type":"video","width":64,"height":48,"disposition":{"attached_pic":1}}],"format":{"duration":"1"}}"#,
            br#"{"streams":[{"codec_type":"video","width":8000,"height":8000}],"format":{"duration":"1"}}"#,
        ] { assert!(parse_probe(invalid, MediaKind::Video).is_err()); }
        assert!(parse_probe(&vec![b' '; MAX_PROBE_BYTES + 1], MediaKind::Audio).is_err());
    }

    #[test]
    fn delayed_tracks_keep_their_complete_file_relative_ranges() {
        let metadata = br#"{"streams":[{"codec_type":"video","width":32,"height":32,"start_time":"0","duration":"40"},{"codec_type":"audio","start_time":"4.936","duration":"30.064"}],"format":{"start_time":"0","duration":"40"}}"#;
        let video = parse_probe(metadata, MediaKind::Video).unwrap().info;
        assert_eq!(video.audio_start_ms, 4936);
        assert_eq!(video.audio_duration_ms, Some(30_064));
        let tracks = timed_segments(&video).unwrap();
        let audio: Vec<_> = tracks
            .iter()
            .filter(|spec| spec.kind == MediaKind::Audio)
            .collect();
        assert_eq!(audio.len(), 2);
        assert_eq!(
            (audio[0].start_ms, audio[0].end_ms),
            (Some(4936), Some(34_936))
        );
        assert_eq!(
            (audio[1].start_ms, audio[1].end_ms),
            (Some(34_936), Some(35_000))
        );
        for spec in tracks {
            validate_segment(&video, &spec).unwrap();
        }
    }

    #[test]
    fn shifted_matroska_tags_are_ends_and_container_origin_is_normalized() {
        let metadata = br#"{"streams":[{"codec_type":"video","width":32,"height":32,"start_time":"10","tags":{"DURATION":"00:00:50.000000000"}},{"codec_type":"audio","start_time":"14.936","tags":{"DURATION":"00:00:45.016000000"}}],"format":{"start_time":"10","duration":"50"}}"#;
        let video = parse_probe(metadata, MediaKind::Video).unwrap().info;
        assert_eq!(video.duration_ms, 40_000);
        assert_eq!(video.video_start_ms, 0);
        assert_eq!(video.video_duration_ms, Some(40_000));
        assert_eq!(video.audio_start_ms, 4936);
        assert_eq!(video.audio_duration_ms, Some(30_080));
        let audio_end = timed_segments(&video)
            .unwrap()
            .into_iter()
            .filter(|spec| spec.kind == MediaKind::Audio)
            .last()
            .unwrap()
            .end_ms;
        assert_eq!(audio_end, Some(35_016));
        let ambiguous = br#"{"streams":[{"codec_type":"audio","start_time":"10"}],"format":{"start_time":"10","duration":"50"}}"#;
        assert!(parse_probe(ambiguous, MediaKind::Audio).is_err());
        let impossible = br#"{"streams":[{"codec_type":"audio","start_time":"5","duration":"10"}],"format":{"start_time":"10","duration":"20"}}"#;
        assert!(parse_probe(impossible, MediaKind::Audio).is_err());
    }

    #[test]
    fn pcm_is_little_endian_finite_bounded_and_nonempty() {
        let mut samples = vec![0.0_f32; 161];
        samples[0] = 0.25;
        samples[1] = -0.5;
        let bytes: Vec<_> = samples
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(parse_audio(&bytes, 161).unwrap(), samples);
        assert!(parse_audio(&[], 161).is_err());
        assert!(parse_audio(&bytes[..bytes.len() - 1], 161).is_err());
        assert!(parse_audio(&bytes, 160).is_err());
        samples[0] = f32::NAN;
        let bytes: Vec<_> = samples
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert!(parse_audio(&bytes, 161).is_err());
    }

    fn frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        let mut frame = format!("P6\n{width} {height}\n255\n").into_bytes();
        frame.extend(vec![value; (width * height * 3) as usize]);
        frame
    }

    #[test]
    fn ppm_parser_preserves_whitespace_pixels_and_bounds_frames() {
        let video = info(MediaKind::Video, 30_000);
        let mut bytes = frame(2, 2, b'\n');
        bytes.extend(frame(2, 2, b' '));
        let frames = parse_frames(&bytes, &video, 2).unwrap();
        assert_eq!(frames.len(), 2);
        assert!(frames[0].as_raw().iter().all(|byte| *byte == b'\n'));
        assert!(frames[1].as_raw().iter().all(|byte| *byte == b' '));
        assert!(parse_frames(&bytes, &video, 1).is_err());
        assert!(parse_frames(&bytes[..bytes.len() - 1], &video, 2).is_err());
        assert!(parse_frames(&[], &video, 1).is_err());
        assert!(parse_frames(&frame(1, 4, 0), &video, 1).is_err());
        assert!(parse_frames(b"P6\n90000 90000\n255\n", &video, 1).is_err());
        assert!(parse_frames(b"P6\n2 2\n65535\n", &video, 1).is_err());
        assert!(parse_frames(&vec![b' '; MAX_PPM_HEADER_BYTES + 1], &video, 1).is_err());
    }

    #[test]
    fn source_dimensions_are_bounded_before_explicit_frame_resize() {
        assert!(validate_video_dimensions(1920, 1080).is_ok());
        assert!(validate_video_dimensions(1080, 1920).is_ok());
        for dimensions in [
            (0, 1),
            (1, 0),
            (8192, 4096),
            (8193, 1),
            (u32::MAX, u32::MAX),
        ] {
            assert!(validate_video_dimensions(dimensions.0, dimensions.1).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn decoder_uses_absolute_single_argument_paths_and_no_network_protocols() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("-clip;echo hacked.mp4");
        fs::write(&path, [1]).unwrap();
        let path = checked_local_path(&path, MAX_MEDIA_FILE_BYTES).unwrap();
        let spec = timed_segments(&info(MediaKind::Video, 1000))
            .unwrap()
            .remove(0);
        let command = decode_command(&path, &spec).unwrap();
        let args: Vec<_> = command.get_args().collect();
        assert!(args.contains(&path.as_os_str()));
        let input = args.iter().position(|argument| *argument == "-i").unwrap();
        assert_eq!(args[input + 1], path.as_os_str());
        let protocol = args
            .iter()
            .position(|argument| *argument == "-protocol_whitelist")
            .unwrap();
        assert_eq!(args[protocol + 1], "file");
        assert!(args.iter().any(|argument| *argument == "-nostdin"));
        for forbidden in ["concat", "hls", "dash", "image2"] {
            assert!(!INPUT_FORMATS.split(',').any(|format| format == forbidden));
        }
    }

    #[cfg(unix)]
    fn shell(script: &str) -> Command {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", script]);
        command
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_captures_both_pipes_and_reports_nonzero_exit() {
        let output = run_decoder(
            &mut shell("printf hello"),
            5,
            Duration::from_secs(2),
            "fixture",
        )
        .unwrap();
        assert_eq!(output, b"hello");
        let concealed = run_decoder(
            &mut shell("printf corrupt >&2; exit 0"),
            32,
            Duration::from_secs(2),
            "fixture",
        )
        .unwrap_err();
        assert!(concealed.to_string().contains("decoding error"));
        let error = run_decoder(
            &mut shell("printf bad >&2; exit 7"),
            32,
            Duration::from_secs(2),
            "fixture",
        )
        .unwrap_err();
        assert!(error.to_string().contains("bad"));
        assert!(error.to_string().contains("7"));
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_output_floods_are_killed_without_unbounded_buffers() {
        let error = run_decoder(
            &mut shell("while :; do printf 12345678; done"),
            32,
            Duration::from_secs(2),
            "fixture",
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("32-byte limit"));
        let error = run_decoder(
            &mut shell("while :; do printf 12345678901234567890123456789012 >&2; done"),
            32,
            Duration::from_secs(2),
            "fixture",
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("65536-byte limit"));
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_timeout_kills_and_waits_even_if_descendant_holds_pipes() {
        let started = Instant::now();
        let error = run_decoder(
            &mut shell("sleep 60 & wait"),
            32,
            Duration::from_millis(50),
            "fixture",
        )
        .unwrap_err();
        assert!(error.to_string().contains("time limit"));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[cfg(unix)]
    #[test]
    fn missing_decoder_dependency_is_actionable() {
        let mut command = Command::new("/nonexistent/fractal-test-decoder");
        let error =
            run_decoder(&mut command, 32, Duration::from_secs(1), "ffmpeg test").unwrap_err();
        assert!(error.to_string().contains("install ffmpeg and ffprobe"));
    }
}
