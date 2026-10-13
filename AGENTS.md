# Fractal Search contributor guidance

Read [README.md](README.md) first. These two root documents provide the current product and engineering orientation. Before editing `quickshell-app/`, also read its [scoped instructions](quickshell-app/AGENTS.md); consult specialized docs only for the area being changed. Keep changes focused and preserve unrelated work.

**No silent fallbacks.** Invalid configuration, missing dependencies, inference failures and incompatible data must produce actionable errors. Never silently substitute providers, decoders, models or path-derived identities.

## Start here

1. Confirm the working directory and inspect `git status --short` and the current branch. This is a self-contained project; do not modify a sibling checkout or introduce dependencies on one.
2. Read the source relevant to the task using the README source map. Current source is authoritative; dated validation sections in deeper docs describe past checks, not guarantees for today's checkout.
3. Distinguish source, installed plugin and live data. Editing source does not update the running UI. Tests must use temporary data, never the user's library.
4. Implement and run the narrowest meaningful checks. For a UI change, verify rendering and actual interactions, including a narrow window and error/loading states. Package/install when the task calls for updating the running app.
5. Review the diff, report what changed and what was tested, and identify environmental limitations. Do not claim unrun checks.

## Architecture and ownership

The root Cargo workspace contains `crates/fractal-search-core`, `quickshell-app/backend` (`fractal-search`), `quickshell-app/folder-picker` (`fractal-folder-picker`) and `quickshell-app/mcp` (`fractal-search-mcp`). It builds `fractal-search`, `fractal-folder-picker` and `fractal-search-mcp`. Production inference is native ONNX Runtime; Python scripts are development tooling only.

- `SearchPanel.qml` owns launcher/results/settings geometry, focus, shortcuts and lifecycle. `Controller.qml` owns desktop socket requests, state, selected assets and operation progress.
- `SettingsPage.qml` has **Folders**, **Preferences** and **Performance** tabs. Keep folder watching (Keep folders up to date) under Folders. Keep the tabs visible while content scrolls; preserve keyboard tab navigation. Model paths, runtime diagnostics and service management belong under Performance.
- Reuse `ResultActions.qml` for **Open file** and **Show in folder**, including list-only/narrow results. Reuse `SearchConfirmationDialog.qml` for removal and Ctrl+Q/Stop service; Cancel is the default focus. Do not add a Browse all photos action. Current layouts are list-and-preview and list-only, not a grid.
- Shared QML controls and `SearchTheme` use Omarchy Color/Style/font tokens. Register new components in `ui/qmldir`; packaging copies QML components automatically. Avoid duplicating styled controls or embedding runtime details in normal search flows.
- `backend/src/daemon.rs` owns the resident socket service and worker process; `main.rs` owns command admission, settings and synchronous work. `models.rs`, `watch.rs` and `diagnostics.rs` handle setup, watching and dependency checks.
- Core `catalog.rs` owns asset IDs, caches and indexing; `store.rs` owns LanceDB metadata, vectors and snapshot queries. `runtime.rs` / `preprocess.rs` / `embeddings.rs` own model inference and preprocessing. `agent.rs` provides the read-only catalog API used by the MCP binary and its transport module.

## UI and lifecycle invariants

- The installed QML runs inside the existing Omarchy Shell process. Never call `Qt.quit()` there or restart/kill the shell to close or update this plugin. Only the excluded developer harness `ui/shell.qml` may own a ShellRoot and quit its own process.
- Keep one launcher panel with compact search, results and settings states. Animate the painted panel inside a fixed native surface; keep input/masks aligned and wait for a correctly sized submitted buffer before revealing a remapped panel. Avoid per-frame native window resizing.
- Release focus for external viewers and native pickers, then restore promptly on summon. Retest repeated viewer close/resummon with both hide-after-opening settings; delays here are a known regression risk.
- Use `PortalFolderDialog.qml` and the native Rust picker. Do not instantiate QtQuick.Dialogs FolderDialog in the shared shell: its GTK/GVfs integration can abort the host. Cancellation must close the portal request; errors stay visible.
- Parent item popups to the persistent canvas, close them before hiding, and restore search focus afterwards. Keep confirmations bounded and long content scrollable.
- Hide/reload detaches UI without stopping indexing. Ctrl+Q asks to drain and stop the backend; the explicit plugin quit method is immediate admission to the same safe shutdown. Finish in-flight inference and release the worker safely. Still images commit individually; incomplete multi-segment files are discarded atomically, preserving their previous complete rows.
- Completed notices can be dismissed, but skipped-file reports remain under Folders until the next scan and new errors must reappear. Never erase active progress to make the UI look idle.

## Data, inference and interface rules

- Store databases, caches, model weights and CUDA libraries outside source control. Use temporary catalogs and synthetic media in tests. Never delete/reindex the user's catalog for validation.
- Desktop catalog initialization may create a logged empty catalog at startup only when its directory is absent or completely empty. Never replace partial/corrupt/incompatible storage or recreate it during an active worker session; deletion requires stopping and reopening the service. The read-only MCP server must not initialize it.
- Preserve the normalized 512-dimensional, single-vector-per-segment indexed EmbeddingGemma 2 FP32 contract. Read the unchanged native 768-dimensional `sentence_embedding`, keep the leading 512 values and L2-normalize every modality and query. Retain explicit full-768 native parity tools only; do not add a user dimension selector. Apply the search-query prefix, respect the 8,192-token model context and retain the 4,096 UTF-8-byte app query bound. Image preprocessing uses 16-pixel patches, pooling kernel 3 and at most 280 soft tokens (2,520 padded patches). Video uses 1 fps visual segments up to 30 frames/30 seconds and 140 soft tokens per frame; available soundtracks additionally produce audio segments. Text passages are at most 4,096 UTF-8 bytes from `.txt`/`.md`/`.markdown` files up to 2 MiB, with overlap. Audio is mono 16 kHz in up to 30-second segments. Audio/video input is capped at 2 GiB/two hours, with explicit ffmpeg/ffprobe dependencies. Do not add PDF/Office or transcription support implicitly. Keep the pinned native-graph manifest unchanged and distinct from indexed `MODEL_PROFILE` (`embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2`). Schema remains 4; old 768-dimensional catalogs must fail with clear fresh-catalog/reindex guidance, remain untouched and never mix with indexed 512-dimensional vectors. Use a separate fresh catalog and explicit reindexing, with no automatic conversion, deletion or fallback. This reduces raw vector storage from 3,072 to 2,048 bytes, not model inference work or VRAM.
- Image batch size is a saved Performance setting (1–8, default 1), captured when the worker starts. Saving prompts for restart; Later retains the active size. Group only consecutive images using the pinned preprocessing budget, never retry failed inference with a smaller batch. Cancellation waits for in-flight inference and commits the current still image; incomplete multi-segment files are discarded while their previous complete rows remain. Never prune unvisited entries.
- Initialize native image decoder hooks before decoding. Report per-file read/decode/change failures and continue healthy files, preserving previous indexed entries for failed files. Traversal, model, database and cache-write failures remain fatal. Cancellation must not prune unvisited entries.
- CUDA is explicit opt-in at build and runtime. CPU builds remain usable without NVIDIA libraries; selected CUDA failures must never retry inference on CPU. Preserve provider/device/library configuration on updates. Diagnostics run without loading models or opening the catalog and distinguish dependency checks from verified inference.
- Schema 4 adds internal filename + full-passage lexical text (at most 5,121 UTF-8 bytes), never exposed in desktop/MCP metadata; published excerpts stay at 2,048 bytes. Require fresh catalogs; no backfill/migration. Only explicit Text searches fuse BM25 + exact semantic ranks with one-based RRF k=60; All media/other types keep cosine ordering. Preserve the separate nullable `hybridScore`, actual-cosine `similarity`, bounded candidates, literal ICU tokenizer contract and prefilters on both legs. Never re-sort hybrid results by cosine or write from read-only MCP. See [hybrid retrieval](docs/HYBRID-TEXT-SEARCH.md), including maintenance resource caveats.
- Metadata operations do not initialize models. Installed service startup warms text only; vision and audio stay lazy. Retain queries typed during warmup and surface failures across reconnects. Prepare search at login uses systemd's actual enabled state.
- The private same-user desktop socket can index and open originals. The separate MCP server is read-only: no catalog creation/migration, original path metadata/file bytes, writes or thumbnail repair from originals. Bounded cached text excerpts (2,048 UTF-8 bytes) are permitted metadata. Nonvisual results have no thumbnails; video uses only cached frame thumbnails. Preserve opaque `asset_<UUID>` identities and cache confinement.
- Reserve worker/MCP stdout for their machine-readable protocols; logs belong on stderr. Bound requests, frames, queues, pages, images and thumbnails; reject unknown fields. Read `quickshell-app/MCP.md` before changing the agent boundary.

## Validation commands

Run commands from the repository root. Choose checks by changed area; documentation-only changes need link/path/command review and `git diff --check`, not inference runs.

For Rust changes:

```bash
cargo fmt --all --check
cargo check --workspace --locked
cargo test --workspace --locked
# Also for GPU/provider/feature integration:
cargo check --workspace --locked --features cuda
cargo test --workspace --locked --features cuda
```

Build binaries for integration checks with `cargo build -p fractal-search -p fractal-folder-picker -p fractal-search-mcp --bins --locked` (add `--features cuda` when needed). CPU and CUDA builds share target paths; confirm capabilities with `target/debug/fractal-search providers` and `target/debug/fractal-search-mcp --providers` before packaging. Use an existing verified model cache for real inference, not a new download just for testing.

For QML lint, provide the host's `qs.Commons` import without modifying its installation:

```bash
SEARCH_LINT_DIR=$(mktemp -d /tmp/fractal-search-lint.XXXXXX)
mkdir -p "$SEARCH_LINT_DIR/qs"
ln -s /usr/share/omarchy/shell/Commons "$SEARCH_LINT_DIR/qs/Commons"
ln -s /usr/share/omarchy/shell/Ui "$SEARCH_LINT_DIR/qs/Ui"
/usr/lib/qt6/bin/qmllint -I "$SEARCH_LINT_DIR" quickshell-app/ui/SearchPanel.qml quickshell-app/ui/SettingsPage.qml
# Include any other changed QML components in the same invocation.
rm -r -- "$SEARCH_LINT_DIR"
```

Adjust the installed Omarchy/Qt paths if this host differs; missing imports must be investigated, not suppressed. Lint does not replace a native UI check.

Model-free dimension/reference-contract checks: `python3 scripts/test_embeddinggemma2_contract.py` and `PYTHONPATH=scripts python3 -m unittest discover -s scripts/qmd_eval -p 'test_*.py' -v`. These validate projection math and fixture/adapter contracts, not real inference or retrieval quality.

Representative integration commands (replace `/absolute/path/to/models` with the existing cache):

```bash
python3 quickshell-app/tests/smoke_plugin.py target/debug/fractal-search /absolute/path/to/models
python3 quickshell-app/tests/smoke_mcp.py target/debug/fractal-search target/debug/fractal-search-mcp /absolute/path/to/models
python3 quickshell-app/tests/smoke_resummon.py target/debug/fractal-search /absolute/path/to/models
python3 quickshell-app/tests/smoke_accelerator_ui.py
python3 quickshell-app/tests/smoke_diagnostics_ui.py
python3 quickshell-app/tests/smoke_multimodal_ui.py
python3 quickshell-app/tests/smoke_hybrid_text_ui.py
python3 quickshell-app/tests/smoke_qmd_ui.py
python3 quickshell-app/tests/smoke_multimodal.py target/debug/fractal-search target/debug/fractal-search-mcp /absolute/path/to/models
python3 quickshell-app/tests/smoke_installer.py /absolute/path/to/prebuilt/bundle
python3 quickshell-app/tests/smoke_dependency_layout.py
python3 quickshell-app/tests/smoke_file_opener.py
```

The resummon check requires Hyprland, hyprctl, Quickshell with QtTest, imv, grim, dbus-daemon, ffmpeg/ffprobe and host graphics/socket access. It waits for configured output geometry in a private nested compositor/session bus, indexes synthetic images/text/audio/video, checks actual media/folder selectors and repeated typed hybrid queries, settings tabs/toggles and narrow layout, exercises Open file and a FileManager1 command fixture, and repeats viewer close/resummon with both checkbox states. Its fixture validates reveal dispatch, not a real file manager. See CUDA.md for real GPU checks and the scoped README for private-bus native-picker tests.

Run simulated keyboard/mouse input only in the isolated compositor. Shared-desktop inspection must be read-only or scoped plugin IPC. Keep `FRACTAL_SEARCH_UI_TEST=1` confined to test processes. Never kill the desktop shell or inject global input to finish a test. If sandbox permissions block sockets/devices, report the environmental failure or use the available approval mechanism; do not weaken production code.

## Packaging and handoff

- Use `scripts/package-fractal-search.sh` and the bundle's `install.sh`, or `./start-quickshell-search.sh --dev` to build/install/summon. For CUDA updates include `--cuda` and preserve the selected provider/library path. `--no-build --summon` only opens an existing installation when no install options are supplied.
- Successful installs prune obsolete runtime bundles, retaining the current/previous builds and runtimes in use; keep pruning after the rollback boundary and test with temporary directories.
- Install UI and backend together; Omarchy's generic git-plugin installer does not build/install the backend. The installer generates RuntimeConfig and versioned paths. Do not patch only the installed QML as the final deliverable.
- Preserve catalog/model caches, login registration, runtime provider settings and personal bindings. Direct installation stops the existing service; preserve its previous active state and check rollback failures. The repository launcher summons afterwards. Inspect logs and plugin `ping` after installation.
- Use the Omarchy skill for desktop customization. Do not change shortcuts without authorization. Do not edit `/usr/share/omarchy/`; user-owned plugins hot reload. A slow plugin load may need `OMARCHY_SHELL_IPC_TIMEOUT=15s` for the installer, not a desktop restart.
- Keep shared Rust dependency versions in root `[workspace.dependencies]`; keep the picker independent of core/storage/MCP and preserve the manifest-only `vendor/lancedb` patch and its provenance; preserve explicit features and pinned inference/catalog contracts. See [dependency review](docs/DEPENDENCIES.md) before changing build-footprint settings.
- Inspect the final diff for unrelated work, generated files and secrets. Change lockfiles only when dependency resolution changes. Do not commit build outputs, model assets, logs, runtime data or generated design mockups. Preserve attribution; publishing/licensing needs separate authorization.
- Keep both root onboarding docs and affected scoped docs aligned when behavior, source ownership, commands or workflows change. Report exact checks and material limitations; dated test counts must never stand in for current validation.

- Accelerator status must distinguish configured, available and active providers. Grey out GPU when requirements fail; do not label a disconnected CUDA service as active CPU. Keep diagnostics collapsed by default, errors visible, and the full report copyable for sharing with an agent. Automatic preflight must not load models or switch providers.

## QMD-inspired indexing and runtime extension

Read [QMD-SEARCH-IMPROVEMENTS.md](docs/QMD-SEARCH-IMPROVEMENTS.md) for the non-destructive schema-4 extension: private generation-bound content/vector caches, separate extraction profile and explicit text reindex, real formatted-input token budgets, match-centered bounded snippets, optional Text document grouping, serial idle session ownership and restart-aware thresholded FTS replacement. Keep lexical/model/extraction contracts distinct. Never describe the bounded full-replacement fallback as native incremental posting maintenance or report synthetic quality/VRAM results as measurements. Run the labeled evaluation harness separately from numerical embedding parity.
