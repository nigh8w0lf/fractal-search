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

The root Cargo workspace contains `crates/fractal-search-core` and `quickshell-app/backend` (package `fractal-search`). It builds `fractal-search`, `fractal-folder-picker` and `fractal-search-mcp`. Production inference is native ONNX Runtime; Python scripts are development tooling only.

- `SearchPanel.qml` owns launcher/results/settings geometry, focus, shortcuts and lifecycle. `Controller.qml` owns desktop socket requests, state, selected assets and operation progress.
- `SettingsPage.qml` has **Folders**, **Preferences** and **Performance** tabs. Keep the tabs visible while content scrolls; preserve keyboard tab navigation. Model paths, runtime diagnostics and service management belong under Performance.
- Reuse `ResultActions.qml` for **Open image** and **Show in folder**, including list-only/narrow results. Reuse `SearchConfirmationDialog.qml` for removal and Ctrl+Q/Stop service; Cancel is the default focus. Do not add a Browse all photos action. Current layouts are list-and-preview and list-only, not a grid.
- Shared QML controls and `SearchTheme` use Omarchy Color/Style/font tokens. Register new components in `ui/qmldir`; packaging copies QML components automatically. Avoid duplicating styled controls or embedding runtime details in normal search flows.
- `backend/src/daemon.rs` owns the resident socket service and worker process; `main.rs` owns command admission, settings and synchronous work. `models.rs`, `watch.rs` and `diagnostics.rs` handle setup, watching and dependency checks.
- Core `catalog.rs` owns SQLite/sqlite-vec, asset IDs, caches and indexing. `runtime.rs` / `preprocess.rs` / `embeddings.rs` own model inference and preprocessing. `agent.rs` provides the read-only catalog API used by the MCP binary and its transport module.

## UI and lifecycle invariants

- The installed QML runs inside the existing Omarchy Shell process. Never call `Qt.quit()` there or restart/kill the shell to close or update this plugin. Only the excluded developer harness `ui/shell.qml` may own a ShellRoot and quit its own process.
- Keep one launcher panel with compact search, results and settings states. Animate the painted panel inside a fixed native surface; keep input/masks aligned and wait for a correctly sized submitted buffer before revealing a remapped panel. Avoid per-frame native window resizing.
- Release focus for external viewers and native pickers, then restore promptly on summon. Retest repeated viewer close/resummon with both hide-after-opening settings; delays here are a known regression risk.
- Use `PortalFolderDialog.qml` and the native Rust picker. Do not instantiate QtQuick.Dialogs FolderDialog in the shared shell: its GTK/GVfs integration can abort the host. Cancellation must close the portal request; errors stay visible.
- Parent item popups to the persistent canvas, close them before hiding, and restore search focus afterwards. Keep confirmations bounded and long content scrollable.
- Hide/reload detaches UI without stopping indexing. Ctrl+Q asks to drain and stop the backend; the explicit plugin quit method is immediate admission to the same safe shutdown. Finish the active image, commit/checkpoint and release the worker before exit.
- Completed notices can be dismissed, but skipped-file reports remain under Folders until the next scan and new errors must reappear. Never erase active progress to make the UI look idle.

## Data, inference and interface rules

- Store databases, caches, model weights and CUDA libraries outside source control. Use temporary catalogs and synthetic images in tests. Never delete/reindex the user's catalog for validation.
- Preserve the normalized 768-dimensional, single-vector-per-image FG-CLIP2 contract and short 64-token text mode. The pinned manifest and `MODEL_PROFILE` must agree with runtime/preprocessing and catalog queries. Profile/schema changes require coordinated migration or explicit reindexing; incompatible vectors must not mix.
- Initialize native image decoder hooks before decoding. Report per-file read/decode/change failures and continue healthy files, preserving previous indexed entries for failed files. Traversal, model, database and cache-write failures remain fatal. Cancellation must not prune unvisited entries.
- CUDA is explicit opt-in at build and runtime. CPU builds remain usable without NVIDIA libraries; selected CUDA failures must never retry inference on CPU. Preserve provider/device/library configuration on updates. Diagnostics run without loading models or opening the catalog and distinguish dependency checks from verified inference.
- Metadata operations do not initialize models. Installed service startup warms text only; vision stays lazy. Retain queries typed during warmup and surface failures across reconnects. Prepare search at login uses systemd's actual enabled state.
- The private same-user desktop socket can index and open originals. The separate MCP server is read-only: no catalog creation/migration, original paths/bytes, writes or thumbnail repair from originals. Preserve opaque `asset_<UUID>` identities and cache confinement.
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

Build binaries for integration checks with `cargo build -p fractal-search --bins --locked` (add `--features cuda` when needed). CPU and CUDA builds share target paths; confirm capabilities with `target/debug/fractal-search providers` and `target/debug/fractal-search-mcp --providers` before packaging. Use an existing verified model cache for real inference, not a new download just for testing.

For QML lint, provide the host's `qs.Commons` import without modifying its installation:

```bash
SEARCH_LINT_DIR=$(mktemp -d /tmp/fractal-search-lint.XXXXXX)
mkdir -p "$SEARCH_LINT_DIR/qs"
ln -s /usr/share/omarchy/shell/Commons "$SEARCH_LINT_DIR/qs/Commons"
/usr/lib/qt6/bin/qmllint -I "$SEARCH_LINT_DIR" quickshell-app/ui/SearchPanel.qml quickshell-app/ui/SettingsPage.qml
# Include any other changed QML components in the same invocation.
rm -r -- "$SEARCH_LINT_DIR"
```

Adjust the installed Omarchy/Qt paths if this host differs; missing imports must be investigated, not suppressed. Lint does not replace a native UI check.

Representative integration commands (replace `/absolute/path/to/models` with the existing cache):

```bash
python3 quickshell-app/tests/smoke_plugin.py target/debug/fractal-search /absolute/path/to/models
python3 quickshell-app/tests/smoke_mcp.py target/debug/fractal-search target/debug/fractal-search-mcp /absolute/path/to/models
python3 quickshell-app/tests/smoke_resummon.py target/debug/fractal-search /absolute/path/to/models
python3 quickshell-app/tests/smoke_accelerator_ui.py
python3 quickshell-app/tests/smoke_diagnostics_ui.py
python3 quickshell-app/tests/smoke_installer.py /absolute/path/to/prebuilt/bundle
```

The resummon check requires Hyprland, Quickshell with QtTest, imv, grim, dbus-daemon and host graphics/socket access. It creates a private nested compositor/session bus, indexes synthetic images, checks settings tabs/toggles and narrow layout, exercises Open image and a FileManager1 command fixture, and repeats viewer close/resummon with both checkbox states. Its fixture validates reveal dispatch, not a real file manager. See CUDA.md for real GPU checks and the scoped README for private-bus native-picker tests.

Run simulated keyboard/mouse input only in the isolated compositor. Shared-desktop inspection must be read-only or scoped plugin IPC. Keep `FRACTAL_SEARCH_UI_TEST=1` confined to test processes. Never kill the desktop shell or inject global input to finish a test. If sandbox permissions block sockets/devices, report the environmental failure or use the available approval mechanism; do not weaken production code.

## Packaging and handoff

- Use `scripts/package-fractal-search.sh` and the bundle's `install.sh`, or `./start-quickshell-search.sh --dev` to build/install/summon. For CUDA updates include `--cuda` and preserve the selected provider/library path. `--no-build --summon` only opens an existing installation when no install options are supplied.
- Install UI and backend together; Omarchy's generic git-plugin installer does not build/install the backend. The installer generates RuntimeConfig and versioned paths. Do not patch only the installed QML as the final deliverable.
- Preserve catalog/model caches, login registration, runtime provider settings and personal bindings. Direct installation stops the existing service; preserve its previous active state and check rollback failures. The repository launcher summons afterwards. Inspect logs and plugin `ping` after installation.
- Use the Omarchy skill for desktop customization. Do not change shortcuts without authorization. Do not edit `/usr/share/omarchy/`; user-owned plugins hot reload. A slow plugin load may need `OMARCHY_SHELL_IPC_TIMEOUT=15s` for the installer, not a desktop restart.
- Inspect the final diff for unrelated work, generated files and secrets. Change lockfiles only when dependency resolution changes. Do not commit build outputs, model assets, logs, runtime data or generated design mockups. Preserve attribution; publishing/licensing needs separate authorization.
- Keep both root onboarding docs and affected scoped docs aligned when behavior, source ownership, commands or workflows change. Report exact checks and material limitations; dated test counts must never stand in for current validation.

- Accelerator status must distinguish configured, available and active providers. Grey out GPU when requirements fail; do not label a disconnected CUDA service as active CPU. Keep diagnostics collapsed by default, errors visible, and the full report copyable for sharing with an agent. Automatic preflight must not load models or switch providers.
