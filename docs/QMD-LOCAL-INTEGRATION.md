# QMD and 512D local source integration

Integrated on 2026-10-12 in `integrate/qmd-512-20261012`. This is a source integration only: no binaries were built, no tests or inference ran, no installation or service restart occurred, and no user catalog/model cache was changed.

## Source ancestry and recovery

The preserved local base is `dc1e78b12e71c5edecdf741b9edbba559c4d0b31`, which contains the previously uncommitted dependency-layout, GIO opener, packaging and test/documentation work. The earlier native inference optimizations and runtime pruning are in its parent `0e3d4d1e0d38ef64bd73bb47b130f472c1af904a`.

- Durable checkpoint: `checkpoint/pre-qmd-20261012-dc1e78b`; `main` was left at the original base.
- Additional recoverable Git bundle: `/tmp/fractal-pre-qmd-dc1e78b.bundle`. Temporary files are not durable across reboot; the checkpoint branch is the primary recovery reference.
- QMD source: `cf09d084ffd07bdd81fb8b4f79b91b5eab7427e9`, merged in `5e4a4da`.
- 512D source: `1cf14085d47c203015af9d76fceb60aea631fe60`, including implementation `6548728`, merged in `c83e3bc`.

QMD merged without textual conflicts. The 512D merge conflicted only in `docs/MODELS.md`, `quickshell-app/CUDA.md` and `quickshell-app/README.md`. Resolution retained the native optimization/benchmark guidance, adopted the 512D indexed contract, preserved all three media graphs, and explicitly labeled prior 768D measurements historical.

## Integrated behavior and exclusions

Structural/token-budget text chunks, byte/line locations, content verification and persistent passage-vector reuse, thresholded/recoverable FTS maintenance, matched snippets, optional grouped Text results, runtime diagnostics and configurable idle session release are retained from QMD.

Text retrieval still uses literal ICU BM25 plus semantic embeddings and equal-weight one-based RRF with k=60. Other media retain cosine ordering. SQLite, token-vector storage/ColBERT/MaxSim, LLM reranking and query expansion were not integrated. Existing image, text, audio and video support remains single-vector-per-segment.

The unchanged pinned FP32 graph emits 768 values. Every indexed/query vector takes the leading 512 and L2-normalizes them; the indexed profile is `embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2`. Native 768 parity tooling remains separate. Old 768D catalogs must remain untouched and fail incompatibility checks; deploying this profile requires a separate fresh catalog and explicit reindexing. No automatic conversion or reindexing was performed. Smaller stored vectors do not reduce graph inference work or model VRAM.

## Preservation review

The root/dependent Cargo manifests and lockfile, split picker/MCP packages, complete vendored LanceDB patch, packaging/install/pruning scripts, GIO opener/actions, synthetic indexing benchmark, resummon test and pinned model manifest were preserved. Key files were compared byte-for-byte with the checkpoint. Runtime source retained `OutputSelector` for only `sentence_embedding`, CUDA intra/inter-op no-spin settings and TF32 disabling. The development core optimization and reduced-debug/incremental profiles remain in the unchanged root manifest.

## Static checks only

- `cargo fmt --all --check` — formatting only, no compilation.
- `git diff --check checkpoint/pre-qmd-20261012-dc1e78b HEAD`.
- Python `ast.parse` over 16 changed Python files — syntax only; modules/tests were not executed.
- `bash -n scripts/package-fractal-search.sh quickshell-app/packaging/install.sh quickshell-app/packaging/prune-runtimes.sh`.
- `qmllint` with the host `qs.Commons` import over SearchPanel, Controller, SettingsPage, AcceleratorSection, GroupedTextMatches, ModelIdlePolicy and ModelRuntimeSection. This is static QML analysis, not rendering/interactions.
- Conflict-marker, ancestry and source-preservation review.

No Cargo build/check/test/clippy, Rust or Python test suite, model inference, native UI validation, dependency installation, model download or deployment was run for this integration. These remain explicitly unverified until the user authorizes coordinated validation. Published historical check counts do not validate this combined tree.

## Prior evidence and future validation limits

The earlier native cf09 768D synthetic baseline measured 202 files/212 passages and 24 queries, with eight provisional hybrid quality-gate failures: paraphrase ordering and a graded folder-filter case. That was the first baseline, not a measured regression against a pre-QMD build. The evaluator used ungrouped passage fusion with canonical-document collapse, not native grouped-file fusion; folder-travel had an exact RRF tie resolved by opaque UUID. Do not infer relevance improvement from numerical projection/parity checks or tune solely to these four queries per category.

CPU idle unload/reload previously preserved vectors exactly. A supplemental CUDA before/after-reload probe exceeded the copied CPU tolerance; the tolerance was not calibrated for CUDA and within-session variation was not controlled. It was not a CPU-versus-CUDA comparison. A follow-up GPU variance/idle-memory diagnostic and narrow native chunk/cache checks were interrupted when the execution connection/app was lost, and must not be described as passed. Crash cause was not established.

Future validation must be coordinated as a single heavy job, use temporary synthetic catalogs and the existing verified model cache, keep the installed app/user data untouched, and distinguish 768-to-512 quality comparisons from source-version comparisons and grouping/candidate changes. No future check was launched by this integration.
