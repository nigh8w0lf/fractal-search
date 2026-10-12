# QMD-inspired search improvements

Implemented against Fractal `317e39a9111d34964280854fe1ad9d68063aa35d`, informed by QMD `93d211f9ef4a869a9aed0d075ca767dda552627f`. This is a Rust/ONNX/Lance adaptation; it adds no LLM reranker or query expansion. FP16/Q8 remain separate work. Model assets, FP32 normalized 768-dimensional embeddings, image budget 280, query formatting and pooling stay pinned.

## Retrieval evaluation

See [QMD-EVALUATION.md](QMD-EVALUATION.md) for the labeled synthetic corpus and real BM25/dense/RRF ablations. Conventional document and passage metrics are distinct from ONNX numerical parity. No actual relevance/latency improvement is claimed until the real-model harness runs; a synthetic test vector is only a correctness fixture. Quality gates are explicit per-category policies, not benchmark results.

## Cheap, content-aware rescans

A scan opens one fresh pinned source snapshot and reads projected metadata in 1,024-row keyset pages, capped at 250,000 segments. The Store shares its connection while each public read operation obtains a fresh snapshot. There is no per-file Lance list/open/schema-validation round trip.

The private `index-state-v1/files/` cache binds each complete file's SHA-256, observed size/mtime, extraction profile and passage locations to a fingerprint of the exact committed rows, including a fresh per-file commit nonce stored in the private opaque modified field. Original raw timestamps remain in the sidecar and legacy values remain readable. Trusted old unchanged stat values avoid file reads. A changed stat or timestamp within two seconds of now (including future times) requires a streaming content hash. Touching a known unchanged file updates the private stat cache without inference or replacement of its rows. New or changed source bytes are hashed with bounded 64 KiB buffers. Changed files still undergo post-decode checks; timestamps classified as recent at admission continue to require byte verification before publication even after a long inference. Explicit verification scans also retain this final byte check. Oversized inputs are rejected before hashing and scan cancellation is checked during the streaming read.

Stat matching is a bounded heuristic, not a tamper detector. The explicit `verifyContent: true` indexing option hashes stable files too, detecting intentionally restored old mtimes. First scans of older catalogs lack a content fingerprint and may need one content-processing pass before the touch fast path is available. Hash/cache failures are visible. Scan cancellation and incomplete traversals never prune unvisited files; extraction/inference failures keep the last complete file generation.

## Text extraction and explicit upgrade

The structural text extraction profile is separate from the model and lexical profiles. Text remains at most 2 MiB/file, 4,096 UTF-8 bytes/passage, 1,024 passages/file and at most 256 bytes overlap. A bounded lookback prefers Markdown headings, paragraph breaks and fence-safe boundaries. The actual model tokenizer counts each formatted document with its title and BOS/EOS against the 8,192-token context. No estimated token count is used in production. Hard UTF-8 splitting guarantees progress; all non-whitespace source content is covered, with original byte and one-based line locations retained. Whitespace-only intervals cannot produce meaningful embeddings and may be omitted.

Schema-4 catalogs remain readable. There is no table replacement or automatic destructive migration. Unchanged legacy text may retain its previous byte-window passages; run an explicit indexing command with `reindexText: true` to rebuild each file atomically with the new profile. A changed extraction profile in a known sidecar requires this explicit opt-in. Cached old searchable rows remain available until each new complete file commits.

The private desktop worker accepts, for example:

```json
{"id":1,"command":{"type":"index","directory":"/absolute/docs","reindexText":true,"verifyContent":true}}
```

This is a trusted desktop command, not an MCP operation. Do not use a user's live catalog for tests.

## Complete-input embedding reuse

`index-state-v1/vectors/` stores validated normalized vectors keyed by a full SHA-256 over the exact formatted input (including title), model profile, complete pinned asset manifest (including tokenizer hashes), runtime implementation identity, precision, dimension/normalization/pooling, tokenizer preprocessing, modality and extraction profile. An unchanged passage inside an edited file can reuse its vector. A changed title, runtime profile, extraction contract or formatted bytes cannot reuse incompatible vectors. Custom embedders must explicitly opt in with an implementation identity; absence disables caching.

The model, extraction and lexical contracts are distinct. Lexical maintenance does not re-embed text. Entries publish atomically and are validated on read; a corrupted cache is an actionable error rather than a silent fallback. Cache writes happen before file publication; failures preserve the previous complete generation. The optional file sidecar is published after the authoritative Lance commit, so a crash can lose reuse or locations but cannot expose an incomplete file. The next scan detects a mismatched generation and rebuilds derived state safely. Cache eviction/space quotas are not implemented; measure growth before use on very large changing corpora.

## Snippets and optional file grouping

Search snippets use the winning stored private passage and the same ICU/lowercase/ASCII-folding lexical token semantics, centering a UTF-8-safe excerpt around a match. A hit beyond the previous first 2,048-byte display prefix becomes visible. Semantic-only matches fall back to their actual retrieved passage. Output is still bounded to 2,048 bytes; browse excerpts, opaque MCP IDs, original-path privacy and full-body restrictions remain unchanged.

`groupTextByFile` defaults to false. When enabled for explicit Text searches, each retrieval leg keeps the strongest passage per `(source_id, path)`, then RRF fuses document ranks. Passage scores are never summed to reward document length. Candidate windows widen within the existing 1,000-per-leg cap. Up to five additional retrieved passages can expand under a winning file. Ties are deterministic. Candidate exhaustion remains possible, so this does not promise exact top documents beyond that cap. All-media/audio/video retrieval remains segment-oriented.

## Safe idle session release

The runtime owns a serial admission/inference/unload boundary and a real idle timer. Admitted active and queued work prevent disposal; lazy loading is single-flight. Completed operations start the idle period. Text, vision and audio policies are independent; default text is 900 seconds and media 300 seconds. `KeepWarm` disables a modality's timeout. The desktop saves `textIdleSeconds` and `mediaIdleSeconds` (0 means KeepWarm, maximum 86,400) and applies policy changes without a model reload. MCP uses its own runtime's safe default timers.

Status diagnostics report loaded modalities, idle time, load/release counts and load/reload durations. Tokenizer/configuration remain resident and cheap; model asset hashes are verified again before reloading released sessions. Dropping ONNX sessions does not prove a particular RSS or VRAM reduction; measure device and allocator behavior on the target hardware. This branch does not incorporate the separate local output-selection/CUDA no-spin/image development optimizations; reconcile `runtime.rs` and build-profile changes carefully later.

## Restart-aware bounded lexical maintenance

Lance's pinned selective index-append API has no FTS memory/worker overrides. Calling it would abandon the existing explicit 128 MiB/two-worker guarantee. This implementation therefore uses write-ahead persisted data/index generations and thresholded **full index replacement**, not claimed incremental posting updates. Small catalogs rebuild promptly; larger catalogs accumulate a bounded change/age obligation. Unindexed fragments remain searchable through Lance, and a restart recovers pending maintenance even when the source scan finds no changed files. Existing schema-4 catalogs conservatively establish maintenance state once.

Replacement preserves the prior index until the new one publishes atomically. It retains the existing builder resource settings and pinned concurrent readers. No `OptimizeAction::All`, destructive index drop, SQLite port, automatic vacuum or old-version cleanup is introduced. Policy simulation tests establish rebuild frequency only. A separate real native benchmark on 5,000 synthetic records and 20 small edits verifies index lag, crash recovery and concurrent readers; see [validation results](QMD-VALIDATION.md). Real user-corpus latency/memory and semantic quality remain unmeasured.

## Validation and reconciliation

Run the root Rust checks and applicable scoped tests from AGENTS.md, the new deterministic evaluation tests, and the real evaluation harness with an existing verified model cache. GPU release/latency requires CUDA hardware; native QML interaction needs the isolated Omarchy compositor workflow. Distinguish pass, failed and unrun checks in the branch handoff.

No model precision options, LLM stages, catalog deletion, deployment, merge or update of the user's installed plugin are part of this branch. Main and the user's uncommitted local quick wins remain untouched.

See [QMD-VALIDATION.md](QMD-VALIDATION.md) for the dated checks, synthetic native maintenance results and explicit blocked stages.
