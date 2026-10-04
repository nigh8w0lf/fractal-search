# Fractal Search

A local image-search app for Omarchy Shell. Add photo folders, describe an image, and open a matching original in your viewer or file manager. The Quickshell panel uses native Rust FG-CLIP2 inference and a LanceDB catalog. CPU and explicitly selected NVIDIA CUDA inference are supported. A separate read-only MCP server makes the indexed library available to agent clients.

This repository is the complete standalone project: both Cargo workspace members, UI, installer and validation tools live here. The production runtime has no Python dependency. New contributors should read this file and [AGENTS.md](AGENTS.md); together they cover the product, architecture, workflow and constraints. Scoped instructions in [quickshell-app/AGENTS.md](quickshell-app/AGENTS.md) also apply when editing that directory.

## Current interface

- **Search:** a compact description field with a Settings gear icon and keyboard hints. Typing searches after a short pause; Enter submits immediately. There is no Browse all photos button. The existing empty-query Enter action still lists indexed images.
- **Results:** a thumbnail list and selected-image preview, friendly folder names, an All folders filter, and explicit **Open image** / **Show in folder** buttons. Preferences offers list-and-preview or list-only layout; narrow windows use the list with actions beneath it. A photo grid is not implemented.
- **Settings → Folders:** add folders, see image counts, remove a folder from the index, rescan folders, keep folders up to date and review skipped files. Removing a folder preserves originals and asks for confirmation.
- **Settings → Preferences:** prepare search at login, hide after opening a photo, maximum results (default 6) and result layout.
- **Settings → Performance:** model setup, a compact Accelerator section, shortcut setup and Stop service. Accelerator checks GPU requirements automatically when shown, displays actual CPU/GPU activity, and greys out an unavailable GPU. Details expands the report; Copy diagnostics prepares it to share with your agent. Availability does not mean active GPU inference and never changes the configured provider.

Up/Down selects results, Enter opens the selection, Ctrl+Enter shows it in the file manager, and Ctrl+, opens Settings. Escape goes back or hides the panel. Ctrl+Q asks to stop the service; Cancel is focused by default, using the same confirmation component as folder removal. Left/Right switches a focused settings tab. All panels follow Omarchy's theme and font tokens.

Hiding or reloading the UI keeps the backend and indexing alive. Stopping the service drains active work and saves the catalog; summoning starts it again. **Prepare search at login** controls the actual systemd user-unit enabled state and preloads only text inference. Vision loads lazily for indexing.

## Build, install and open

From this repository's root:

```bash
# CPU development build, installation and summon
./start-quickshell-search.sh --dev

# CUDA-capable build with explicit GPU selection
./start-quickshell-search.sh --dev --cuda --provider cuda --device-id 0
# Add --cuda-lib-dir /absolute/persistent/path/to/lib if libraries are installed privately.
```

Omit `--dev` for a release build. First-run **Set up model**, also available under Settings → Performance, downloads and verifies the pinned assets (about 1.14 GB). **Choose folder** selects an existing verified model cache. Do not download models just to run metadata checks.

Open an existing installation without building or touching the checkout:

```bash
omarchy-shell shell summon fractal.search '{}'
# Equivalent repository convenience command:
./start-quickshell-search.sh --no-build --summon
```

Use the installed summon command for a desktop shortcut. Installation preserves existing shortcuts; optional `--bind-close` explicitly installs managed Super+W close routing. Do not restart the desktop shell to update this plugin.

Development needs Rust, a C/C++ toolchain, pkg-config and native image libraries. The runtime needs Omarchy Shell/Quickshell, curl, jq, libheif, libjpeg-turbo, gcc-libs and a desktop portal providing FileChooser. Show in folder uses `gdbus` and a file manager implementing FileManager1.ShowItems. The Rust build selects system libjpeg-turbo on Linux. Python is used only by development/validation tools.

CUDA bundles include the ONNX Runtime provider libraries; NVIDIA driver, CUDA 12 runtime and cuDNN 9 libraries are installed separately. Use `--cuda` when updating a CUDA installation; a CPU bundle refuses to replace a CUDA configuration unless you explicitly switch to `--provider cpu`. Startup/inference failures remain visible and never retry the model on CPU. See [CUDA.md](quickshell-app/CUDA.md) for exact dependencies and validation.

To package without installing:

```bash
scripts/package-fractal-search.sh --dev --output /absolute/new/bundle-directory
# For an already built CUDA debug runtime:
scripts/package-fractal-search.sh --dev --no-build --cuda --output /absolute/new/cuda-bundle-directory
```

Output directories must not already exist. `--no-build` packages all three existing binaries and checks their CPU/CUDA capability; it does not rebuild them. Install with the bundle's `install.sh`. UI and backend must be installed together. Updates drain the backend before swapping files; preserve its previous active state when using the installer directly. The repository launcher summons it afterwards.

## Architecture and source map

```text
Omarchy Shell → SearchPanel / Controller → private Unix socket
                                          ↓
                               fractal-search daemon
                                          ↓ worker NDJSON
                               native inference + catalog

MCP client → fractal-search-mcp (stdio) → existing catalog, read-only
```

The desktop socket is trusted same-user access: it may index folders and open originals. The MCP process is a different permission boundary; it exposes opaque identities, metadata and bounded thumbnails, never original paths or bytes. It can read alongside the desktop worker. Metadata does not load models; semantic MCP search starts its own text session lazily. Configure clients with the installed stable `~/.config/omarchy/plugins/fractal.search/fractal-search-mcp` launcher; use an absolute path. See [MCP.md](quickshell-app/MCP.md).

| Source | Responsibility |
| --- | --- |
| `quickshell-app/ui/SearchPanel.qml` | Panel geometry, focus, search/results, shortcuts and plugin lifecycle. |
| `quickshell-app/ui/Controller.qml` | Socket protocol, application state, search/index operations and previews. |
| `quickshell-app/ui/SettingsPage.qml` | Folders, Preferences and Performance tabs. |
| `quickshell-app/ui/ResultActions.qml` | Shared Open image / Show in folder controls for preview and narrow/list layouts. |
| `quickshell-app/ui/SearchConfirmationDialog.qml` | Shared folder-removal and stop-service confirmation. |
| `quickshell-app/ui/AcceleratorSection.qml` | Compact CPU/GPU activity and capability status with expandable, copyable diagnostics. |
| `quickshell-app/ui/RuntimeConfig.qml`, `RuntimeDiagnostics.qml`, `ServiceAutostart.qml` | Runtime paths, dependency diagnostics and login registration. The installer generates the installed RuntimeConfig. |
| `quickshell-app/backend/src/main.rs`, `daemon.rs` | Worker commands/settings and resident socket service/lifecycle. |
| `quickshell-app/backend/src/models.rs`, `watch.rs`, `diagnostics.rs` | Verified downloads, folder watching and CUDA preflight. |
| `quickshell-app/backend/src/bin/` | Native portal folder picker and stdio MCP server/transport. |
| `crates/fractal-search-core/src/` | Catalog, agent read API, decoding/preprocessing, tokenization and ONNX runtime. |
| `quickshell-app/packaging/`, `scripts/package-fractal-search.sh` | Versioned bundles, installation, uninstall and optional close binding. |
| `quickshell-app/tests/`, `scripts/` | Temporary-catalog smoke checks and optional reference-model tooling. |

FG-CLIP2 produces a normalized 768-dimensional vector for each image/query; the catalog stores one vector per image, not multi-vector retrieval. Text uses short mode (64 tokens). Model assets and hashes are pinned in `crates/fractal-search-core/models/fgclip2.json`; the profile constant in `src/lib.rs` guards compatibility. Changing embeddings/preprocessing requires a coordinated profile change and reindexing. See [model details](docs/MODELS.md).

## Runtime locations and troubleshooting

Defaults below follow XDG configuration/data/cache roots:

| Location | Contents |
| --- | --- |
| `~/.config/omarchy/plugins/fractal.search/` | Installed plugin and stable MCP launcher. |
| `~/.config/systemd/user/fractal-search.service` | User service. |
| `~/.local/share/fractal-search/` | `catalog.lancedb/`, `settings.json`, thumbnails, previews and service lock. |
| `~/.local/share/fractal-search/runtime/<version>/` | Versioned runtime bundles. |
| `~/.cache/fractal-search/models/fgclip2-33f3b9d4/` | Verified model assets. |
| `$XDG_RUNTIME_DIR/fractal-search/backend.sock` | Private desktop socket. |

Provider/device/private-library settings are preserved in the installed `.fractal-search-runtime.json`. UI preferences and model location persist in the data directory; login registration lives in systemd. Uninstall preserves the catalog and model cache. Never use these live paths for tests.

```bash
systemctl --user status fractal-search.service
journalctl --user -u fractal-search.service -n 80 --no-pager
omarchy-shell shell call fractal.search ping ''
```

A source edit is not an installed update: rebuild/package/install to change the running plugin. If installation hits the shell IPC timeout, inspect the error and service state; `OMARCHY_SHELL_IPC_TIMEOUT=15s /path/to/bundle/install.sh` allows a longer load/rescan window. Do not hide errors or restart the whole desktop as a workaround.

## Validation and further reading

[AGENTS.md](AGENTS.md) contains runnable checks and task-specific validation guidance. The native UI smoke uses a private nested compositor, temporary catalogs and synthetic photos. Shared-desktop keyboard/mouse automation is prohibited.

- [Detailed plugin usage and historical validation](quickshell-app/README.md)
- [CUDA runtime requirements and real-inference validation](quickshell-app/CUDA.md)
- [MCP protocol, client configuration and privacy limits](quickshell-app/MCP.md)
- [Model contract and reference fixtures](docs/MODELS.md)

This is a local development project. Model weights and generated bundles are not checked in. Public release/signing and redistribution review remain separate work; preserve [bundle notices](quickshell-app/packaging/NOTICE).

## LanceDB catalog format

LanceDB stores folder records, image metadata and normalized vectors in `catalog.lancedb/`. Each image update commits metadata and its vector together; folder removal commits the source and its images together. Read-only MCP queries pin a table snapshot. Search uses exact cosine distance with folder filtering before ranking.

This development format has no SQLite migration. Use a fresh `--data-dir /absolute/new/catalog` and index the desired folders. The model profile and opaque asset identities are unchanged. Lance versions are retained for concurrent readers; automatic compaction/version cleanup is not implemented yet.

Builds require Rust 1.91 or newer and the Protocol Buffers compiler (`protoc`). LanceDB is pinned to 0.37.1 with cloud/remote features disabled; 0.38.0 and 0.39.0 have an upstream local-only compilation error in their job error handling.
