# Text-only hybrid retrieval

## Scope and behavior

The results panel offers **All media**, **Images**, **Text**, **Audio** and **Video**, alongside the folder selector. All media remains the default. Only a nonempty query with **Text** selected uses hybrid retrieval: native LanceDB BM25 plus exact EmbeddingGemma 2 cosine search, fused by passage identity. All media and the other type filters retain exact cosine ordering. Both legs apply the same folder and text-kind filter before ranking. The type describes an indexed segment: a video's soundtrack appears in Audio, and its visual segment in Video.

An empty query browses the selected type/folder in filename/asset-ID order, without loading a model. Typed queries and filter changes are paired with a request generation; stale results, progress, errors and previews cannot replace a newer scope. Pending changes are coalesced to the newest scope, including service warmup and reconnect.

Results default to passages, so a long file can occupy several slots. Optional **Group Text matches by file** collapses each leg to its strongest passage per document, then fuses document ranks. Up to five additional retrieved passages can be expanded per winning file. All-media/audio/video stay segment-oriented. PDF/Office extraction, OCR, transcription, query expansion, reranking and ColBERT/late interaction are not implemented.

## Indexed content and catalog contract

Schema 4 and indexed embedding profile `embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2` are required. All modalities and queries use the leading 512 dimensions of the unchanged native 768-dimensional FP32 output, followed by L2 normalization. Existing schema-4 catalogs remain readable only when their embedding profile also matches; the previous 768-dimensional profile is incompatible. Use a separate fresh catalog and explicitly reindex. Earlier schemas, SQLite and incompatible embedding profiles are not migrated, deleted or automatically repaired. The catalog dimension and `fractal.embedding_projection = native768-leading512-l2-v1` guard, additional lexical/chunk contract and named FTS index are validated on open, including read-only opens.

Only text rows contain the internal `indexed_text` field. Its `filename-newline-passage-v1` construction is a UTF-8-safe filename title of at most 1,024 bytes, one newline and the complete passage of at most 4,096 bytes (5,121 bytes total). It contains no directory path or embedding prompt labels. The separately versioned structural extraction profile uses at most 256-byte overlap, at most 1,024 passages per file and the real formatted-input 8,192-token cap. Explicit text reindexing upgrades unchanged legacy windows. Browse `textExcerpt` stays a passage-only prefix; search centers it on a matching ICU token in the winning stored passage or falls back to that passage prefix. Both are at most 2,048 UTF-8 bytes. Neither desktop metadata nor the MCP projection returns the full internal lexical field. Search never reads original files.

## Literal multilingual terms

The pinned LanceDB 0.37.1 / Lance 10 tokenizer uses `icu/split`, lowercase and ASCII folding, with stemming and stop-word removal disabled. ICU provides bundled multilingual segmentation without a downloaded language model. Maximum token length is 4,096. Positions are not stored; this is literal term matching with OR semantics, not a phrase or Boolean query language. Quotes, apostrophes and words such as AND/OR are ordinary input, and punctuation may split terms. Byte-exact identifier and substring matching are not promised.

Queries retain the 4,096 UTF-8-byte bound and NUL rejection. A punctuation-only query with no lexical tokens has an empty BM25 list and still runs semantic text search. A tokenizer, index or query failure is an error, never a silent vector-only fallback. Search text is passed through the native Match-query API rather than interpolated into SQL.

## Ranking and scores

One pinned table snapshot serves both candidate lists. Each leg retrieves `min(1000, max(100, 4 * result_limit))` candidates. The vector leg uses exact cosine with prefiltering and no approximate/fast-search shortcut. Each list is sorted by its native score with asset-ID tie-breaking, then assigned one-based ranks. Equal-weight Reciprocal Rank Fusion adds `1 / (60 + rank)` for every list containing a passage. Results are deduplicated by opaque asset ID, sorted by descending fused score with asset-ID tie-breaking, then truncated to the requested result limit. Candidate limits and output limits are separate. Grouped Text searches widen each leg when duplicate passages leave too few files, up to the same 1,000-candidate cap. After strongest-per-file collapse, each leg receives document ranks before RRF; additional passages never add document score. Candidate exhaustion can still limit document recall.

`similarity` always means actual cosine, including candidates found only through BM25. `hybridScore` is optional and emitted only for text-hybrid search (omitted for browse/semantic results). It is a rank-fusion value, not a probability, percent, cosine similarity or cross-query relevance calibration. The UI labels it **Hybrid score** and preserves backend order. Search responses expose `searchMode: "hybrid"` for Text and `"semantic"` otherwise. Browsing has no search score.

These candidate depths and RRF k=60 are reproducible initial settings, not tuned relevance claims. Single-vector passage retrieval and BM25 do not require token-level multi-vector models. Compare dense-only, BM25-only and hybrid on labeled exact-name/identifier and paraphrase queries before claiming gains. Report Recall@100, nDCG@6 or MRR, and p50/p95 latency at representative passage counts; synthetic correctness tests do not establish real-world quality or speed.

## Index lifecycle and resource limits

A fresh catalog creates its named FTS index even when empty. Existing catalogs must contain a compatible index; read-only MCP opens never create, repair or optimize it. Complete-file replacements keep text, vectors and metadata atomic. Fresh unindexed fragments are searchable by native Match queries, so interrupted scans retain searchable completed files without waiting for maintenance.

The desktop atomically replaces the named FTS index at eligible scan/checkpoint boundaries, not per query or per file. Small corpora (at most 4,096 text rows) rebuild after text changes; larger corpora wait for at least max(256, 10% of text rows) changed rows, a 24-hour age threshold observed at a checkpoint, or restart recovery. It reuses the explicit 128 MiB builder buffer and two-worker settings. Image/source-only writes and image-only deletions do not trigger this rebuild. Old pinned snapshots retain their prior index version. Previous table/index versions are intentionally retained for readers, so frequent changes to a large text corpus can increase disk use; there is no automatic version or index-file cleanup. It does not use `OptimizeAction::All`, compact/prune old versions or alter the concurrent-reader retention policy. A maintenance failure is reported after committed rows remain durable; it is not reported as a failed atomic file transaction.

Each actual full-index rebuild is O(corpus size). Thresholding reduces rebuild frequency, not the cost of a replacement; unindexed fragments may trade cheaper indexing for slower queries until maintenance runs. The configured buffer is not a hard process-memory cap. The pinned upstream incremental index-only optimization API reloads runtime builder defaults and cannot apply the same memory/worker overrides; Fractal uses full replacement to retain the configured builder settings. Large-library latency/memory measurements remain necessary before treating this initial policy as a production envelope.

Text/index generations and the last published index identity are persisted in a small write-ahead sidecar. Intent is fsynced before a text-row commit; indexed generation advances only after atomic index replacement. A crash cannot erase the maintenance obligation. Reopening marks pending work for repair even on an unchanged scan/checkpoint. Compatible-profile schema-4 catalogs without this optional sidecar conservatively establish it once. Read-only MCP never writes it. A failed intent or index build remains visible and retains complete searchable rows.

## Interfaces and validation

Desktop `browse`/`search` accept optional `mediaKind: "image" | "text" | "audio" | "video" | null` together with the existing `sourceId`; omit or pass null for All media. MCP `browse_assets`/`search_assets` accept the same optional type filter without exposing folder paths or source IDs. Unknown fields and invalid enum values (including `"all"` and `"Text"`) are errors. Existing query/frame/page/result bounds remain in force. Text-hybrid search still needs the pinned text model; metadata browsing does not.

Core tests use synthetic vectors and temporary catalogs to verify FTS lifecycle, literal tokenization, full-passage/filename recall, source/type filtering, deterministic fusion, replacement/removal and unchanged all-media behavior. The backend/MCP unit tests verify strict optional filter arguments. Run from the repository root:

```bash
python3 quickshell-app/tests/smoke_hybrid_text_ui.py
python3 quickshell-app/tests/smoke_qmd_ui.py
python3 quickshell-app/tests/smoke_multimodal_ui.py
python3 quickshell-app/tests/smoke_multimodal.py target/debug/fractal-search target/debug/fractal-search-mcp /absolute/path/to/pinned/models
```

The first test runs the actual Controller JavaScript functions in Node with a synthetic socket/list model; it covers filter/query races, warmup/reconnect, score ordering and stale previews without models. Static layout checks are not QML rendering. The second requires Quickshell and the host's theme imports. The last uses real inference and synthetic temporary media, checks filtered browsing/search and preserves read-only MCP catalog hashes after originals are removed. Native narrow-window, keyboard/mouse, popup, loading/error and render checks must be run separately on a suitable isolated desktop. Do not infer they passed from the Node checks.

## Research sources

- [Original RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
- [Analysis of hybrid fusion functions](https://arxiv.org/abs/2210.11934)
- [LanceDB full-text search](https://docs.lancedb.com/search/full-text-search)
- [LanceDB hybrid search](https://docs.lancedb.com/search/hybrid-search)

Exact-version behavior was checked against locally resolved LanceDB 0.37.1 and Lance/Lance Index 10.0.0 sources. The application performs fusion itself: that LanceDB version's built-in hybrid path normalizes distance metadata and uses zero-based RRF denominators, which would violate the score contract above. Earlier broader document/ColBERT investigation remains in [HYBRID-SEARCH-INVESTIGATION.mkd](HYBRID-SEARCH-INVESTIGATION.mkd); it is not a claim those features are shipped.

## QMD-inspired extension

The default ungrouped ranking remains the baseline ablation. See [QMD-SEARCH-IMPROVEMENTS.md](QMD-SEARCH-IMPROVEMENTS.md) for optional per-file Text fusion, match-centered bounded snippets, independent text extraction/versioned reuse, shared connections and persisted thresholded FTS maintenance. This does not add LLM reranking or expansion. The FTS optimization is restart-aware and thresholded full replacement because pinned selective append does not expose the required builder resource controls.
