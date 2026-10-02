# Fractal Search contributor guidance

Read `README.md` and `quickshell-app/AGENTS.md` before editing. Keep changes focused and preserve unrelated work. Do not use silent fallbacks: invalid configuration, missing dependencies, inference failures and incompatible data must produce actionable errors.

## Architecture

- `quickshell-app/ui/`: Qt Quick/QML panel hosted inside Omarchy Shell.
- `quickshell-app/backend/`: Rust service, desktop-portal folder chooser and stdio MCP server.
- `crates/fractal-search-core/`: native decoding, patch preprocessing, tokenizer, resident FG-CLIP2 ONNX sessions and SQLite/sqlite-vec catalog.
- `quickshell-app/packaging/` and `scripts/package-fractal-search.sh`: installation and versioned runtime bundles.

The root Cargo workspace owns both crates. Keep dependencies inside this repository or in declared Cargo dependencies; do not introduce references to sibling checkouts. The production runtime has no Python dependency. Python scripts under `scripts/` are optional offline model-validation tooling.

## Development

Run from the root:

```bash
cargo fmt --all --check
cargo check --workspace --locked
cargo test --workspace --locked
# When changing GPU behavior or feature integration:
cargo check --workspace --locked --features cuda
cargo test --workspace --locked --features cuda
```

Build/install with `./start-quickshell-search.sh --dev`. Use `--cuda --provider cuda` for a GPU installation; never infer a provider silently. See `quickshell-app/CUDA.md` for runtime libraries, real GPU smoke and model parity tests. Package without installing using `scripts/package-fractal-search.sh --output /absolute/new/directory`.

Use the narrowest meaningful tests, then the relevant workspace checks. Native libraries, host Unix sockets, GPU devices and compositor access may require an appropriate host environment. Report environmental failures rather than weakening checks. Real-inference tests require the pinned model cache; do not download large models merely to run metadata tests.

## Data and interfaces

- Keep catalogs, model caches, thumbnails, previews and CUDA libraries outside the source tree. Tests use temporary catalogs and synthetic images, never the user's library.
- Preserve the embedding profile, dimension and preprocessing contract. Schema/profile changes require coordinated migrations and explicit existing-data handling.
- The trusted desktop socket may index folders and open originals. The separate MCP server is read-only, exposes opaque handles, excludes original paths/bytes and never repairs caches from originals. See `quickshell-app/MCP.md`.
- Keep stdout machine-readable for worker and MCP protocols; diagnostics belong on stderr. Bound input, output, queues, image decoding and queries. Reject unknown fields.
- Metadata operations must not initialize models. Text search loads only the text session; vision stays lazy until indexing.
- Cancellation finishes the active image, commits/checkpoints the catalog and drains before releasing the worker. Hiding or reloading the panel must not stop indexing.
- Keep errors, progress and skipped-file reports visible. Decoder errors may skip individual files with an explicit report; model, traversal, database and cache-write failures remain fatal.

## Change discipline

Inspect `git status` before editing. Do not discard unrelated changes or format unrelated files. Change lockfiles only when dependency resolution or workspace membership changes. Keep documentation aligned with commands and model/runtime behavior.

Do not commit build outputs, temporary fixtures, caches, logs, databases, model weights, secrets or downloaded runtime libraries. Preserve model attribution. Do not publish or choose a project license without authorization.

Run keyboard/mouse automation only in an isolated compositor. Shared-session inspection must be read-only or use scoped plugin IPC. Follow the Omarchy skill for user desktop configuration, preserve existing bindings, and install the UI/runtime together through the product installer.

Before handing off, review the diff, run relevant checks, state skipped checks and their reasons, and confirm data, errors and progress remain intact.
