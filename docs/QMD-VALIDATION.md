# QMD improvement validation

Historical validation of the QMD extension before the 512-dimensional indexed-profile change; these results do not validate that later change. Rerun checks against the current tree, and do not reuse its measured catalog sizes as 512D measurements.

Validation date: 2026-10-08. Base: `317e39a9111d34964280854fe1ad9d68063aa35d`.

## Environment and scope

Cloud Linux x86-64, Rust 1.91.0, ONNX Runtime 1.22.0 CPU, libheif 1.20.2, libjpeg-turbo 3.1.2, protoc 3.21.12, installed FFmpeg/ffprobe. Dependencies used the locked Cargo resolution. Validation-only Cargo configuration disabled optimization/debug information to fit the cloud; production source profiles were unchanged. No user computer, library, installation, local uncommitted work or deployment was touched.

## Completed checks

- `cargo fmt --all --check`
- CPU and CUDA-feature `cargo check --workspace --all-targets --locked` (CUDA adds `--features cuda`), including both new evaluation/lifecycle examples
- Available CPU workspace tests: 143 passed (119 core, 19 backend, 3 MCP, 2 picker); four opt-in tests ignored by the ordinary run and one private-socket test explicitly filtered because the sandbox denies UnixListener binding
- Explicit real FFmpeg/ffprobe audio/video/soundtrack regression passed
- Explicit native synthetic FTS maintenance benchmark passed, including current readers, pinned historical readers and unchanged-restart maintenance recovery
- 33 Python evaluation metric/fixture/protocol tests, including adversarial truncated-ranking rejection
- `smoke_hybrid_text_ui.py` and `smoke_qmd_ui.py`: actual extracted Controller JavaScript plus static UI checks, including grouping bounds, escaped/Unicode excerpts, stale scopes, pagination, idle status refresh and confirmed rebuild behavior
- Whitespace diff checks and scoped source review; all confirmed findings fixed and regression-tested

The available CUDA-feature workspace suite also passed the same 143 tests, with the same explicit socket filter and opt-in ignores. Compile/test feature integration on this CPU host does not establish actual CUDA inference.

## Native FTS measurement

Command:

```sh
cargo test --workspace --locked benchmark_small_edit_maintenance_with_readers_and_restart -- --ignored --nocapture
```

The fixture contains 5,000 short synthetic text records with synthetic normalized vectors and 20 one-file replacements. Initialization is excluded. Each arm retains a pinned historical reader while a second reader checks current results. These are **unoptimized cloud, synthetic-corpus measurements**, not semantic quality or production hardware benchmarks.

| Measurement | Immediate full replacement | Thresholded full replacement |
| --- | ---: | ---: |
| Rebuilds during 20 edits | 20 | 0 |
| Update + checkpoint p50 | 196.75 ms | 97.70 ms |
| Update + checkpoint p95 | 212.65 ms | 117.95 ms |
| Two current-reader queries p50 | 51.93 ms | 53.38 ms |
| Two current-reader queries p95 | 72.12 ms | 71.03 ms |
| Data/indexed generation before restart | 21 / 21 | 21 / 1 |
| Indexed/unindexed rows before restart | 5,000 / 0 | 4,999 / 1 |
| Retained catalog bytes before recovery | 16,203,339 | 15,722,066 |
| Restart/checkpoint recovery | 5.02 ms | 115.01 ms |

The deferred arm retained 20 change intents even though the same physical row was edited each time. Restart repaired that obligation. Process-lifetime peak RSS was about 390 MB after the first arm and 426 MB after the second; the peak is cumulative in one process, so it is **not** an independent per-policy memory comparison. The 128 MiB FTS builder buffer is not an overall RSS cap. This measures thresholding's tradeoff; it does not implement incremental posting updates or prove large-library query latency.

## Blocked or unrun checks

- Real EmbeddingGemma retrieval evaluation and real idle unload/reload parity: the official pinned Hugging Face download redirects large weights to `us.aws.cdn.hf.co`, where the cloud proxy returns tunnel 403. Supported escalation did not succeed. Only config.json and audio_encoder.onnx were downloaded and SHA-256 verified (209,608 bytes total); no weights were available
- The real evaluation preflight exits 2 and reports `quality: null`, `latency: null`, with missing assets identified. No substitute vectors or passing relevance gates were reported
- One local Unix-socket test: sandbox `UnixListener::bind` returns EPERM. Escalated runner fails before launch with an environment `bwrap` setup error. The test remains intact; it was filtered only for the available-suite run
- Native Quickshell rendering, actual narrow-window/popup/keyboard interactions and qmllint: Quickshell and the actual Qt lint executable are absent. JS/static smoke does not replace native QA
- CUDA inference, actual VRAM reclamation, real model reload latency and real-corpus ranking/quality are unmeasured

## Reproduce on a suitable development machine

Use a temporary data directory and the existing verified model cache. Do not reindex a live library merely to validate this branch.

```sh
cargo fmt --all --check
cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo check --workspace --all-targets --locked --features cuda
cargo test --workspace --locked --features cuda
cargo build --workspace --examples --locked
python3 scripts/evaluate-qmd.py --help
cargo run -p fractal-search-core --example validate_qmd_lifecycle --locked -- /absolute/path/to/pinned/models
python3 quickshell-app/tests/smoke_hybrid_text_ui.py
python3 quickshell-app/tests/smoke_qmd_ui.py
```

Follow [QMD-EVALUATION.md](QMD-EVALUATION.md) for the exact native-adapter path and report options. Follow [AGENTS.md](../AGENTS.md) for native UI and CUDA workflows. Model precision/output-selection/no-spin/image-build quick wins in the separate local checkout must be reconciled deliberately, especially `runtime.rs`; they are not included or claimed as new work here.
