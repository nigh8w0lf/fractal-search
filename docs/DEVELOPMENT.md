# Developing Fractal Search

[Project overview](../README.md) · [All documentation](README.md)

Read [contributor guidance](../AGENTS.md) before making changes, and the [scoped guidance](../quickshell-app/AGENTS.md) before editing `quickshell-app/`. Commands below run from the repository root. Source paths in the tables are relative to that root.

## Build and install from source

For development from this repository's root:

```bash
# Fetch verified third-party covered source once for distributable bundles
python3 scripts/fetch-release-sources.py

# CPU development build, installation and summon
./start-quickshell-search.sh --dev

# CUDA-capable build with explicit GPU selection
./start-quickshell-search.sh --dev --cuda --provider cuda --device-id 0
# Add --cuda-lib-dir /absolute/persistent/path/to/lib if libraries are installed privately.
```

Omit `--dev` for a release build. First-run **Set up model**, also available under Settings → Performance, downloads and verifies the pinned assets (about 1.50 GB). **Choose folder** selects an existing verified model cache. Do not download models just to run metadata checks.

Development builds optimize the native core's image/preprocessing loops at level 3 while retaining the app's development profile and level-2 dependency optimization. Workspace code keeps level-1 debug information, dependencies omit debug information, and incremental compilation remains enabled for faster rebuilds. Shared dependency versions are centralized in the root manifest; see the [dependency review](DEPENDENCIES.md) for the graph and tradeoffs. CPU/CUDA inference fetches only the sentence embedding; CUDA's CPU worker pools sleep while waiting for work instead of spinning. These execution optimizations retain the indexed vector contract. The [synthetic indexing benchmark](../quickshell-app/README.md#synthetic-indexing-benchmark) records scan timings, CPU usage and search-score parity between binaries.

For a targeted development build, select the package you changed:

```bash
cargo build -p fractal-folder-picker --locked
cargo build -p fractal-search --locked
cargo build -p fractal-search-mcp --locked
# Build all three CUDA-bundle binaries together:
cargo build -p fractal-search -p fractal-folder-picker -p fractal-search-mcp --bins --locked --features fractal-search/cuda,fractal-search-mcp/cuda
```

The picker does not compile the inference/catalog or MCP stacks. Daemon and MCP
CUDA features are forwarded independently; the bundle script selects both and
checks their capabilities. The local LanceDB manifest patch keeps ICU tokenization
and the existing catalog format; it introduces no migration or reindexing.

Development needs Rust, a C/C++ toolchain, pkg-config and native image libraries. The runtime needs Omarchy Shell/Quickshell, curl, jq, libheif, libjpeg-turbo, gcc-libs, GLib's `gio`, ffmpeg and ffprobe and a desktop portal providing FileChooser. Show in folder uses `gdbus` and a file manager implementing FileManager1.ShowItems. The Rust build selects system libjpeg-turbo on Linux. Python is used only by development/validation tools.

Native inference uses pinned ONNX Runtime 1.24.2 (`ort` rc.12). CUDA bundles include its matching provider libraries; NVIDIA driver, CUDA 12 runtime and cuDNN 9 libraries are installed separately. Use `--cuda` when updating a CUDA installation; a CPU bundle refuses to replace a CUDA configuration unless you explicitly switch to `--provider cpu`. Startup/inference failures remain visible and never retry the model on CPU. See [CUDA.md](../quickshell-app/CUDA.md) for exact dependencies and validation.

CUDA sessions grow their memory arenas to the requested size and return unused scratch buffers after each model call. This prevents earlier image/audio/video work from retaining unnecessary GPU capacity across the three resident graphs. Weights remain loaded according to the existing idle policy. Allocation failures still stop the scan; no model, precision, batch-size or provider fallback is used.

To package without installing:

```bash
scripts/package-fractal-search.sh --dev --output /absolute/new/bundle-directory
# For an already built CUDA debug runtime:
scripts/package-fractal-search.sh --dev --no-build --cuda --output /absolute/new/cuda-bundle-directory
```

Output directories must not already exist. `--no-build` packages all three existing binaries and checks their CPU/CUDA capability; it does not rebuild them. Install with the bundle's `install.sh`. UI and backend must be installed together. Updates drain the backend before swapping files; preserve its previous active state when using the installer directly. The repository launcher summons it afterwards.

## Test first-run setup from local code

Commit the setup changes locally, then remove any existing managed installation
using its runtime's `uninstall.sh` (see the user guide). Catalogs and models remain.
Install the committed source without publishing it:

```bash
omarchy plugin add /absolute/path/to/fractal-search --enable
omarchy-shell shell toggle fractal.search
```

The bootstrap accepts an absolute local Git checkout whose origin is the official
HTTPS repository and whose HEAD matches the installed clone. Uncommitted changes
are not cloned; a dirty installed clone or changed source HEAD is rejected.
This tests first-run installation of the pinned release binaries, not a build of
local Rust/UI changes. For privately installed CUDA libraries, enter the absolute
library folder in the setup panel (equivalent to the bundle installer's `--cuda-lib-dir` option).

## Architecture and source map

```text
Omarchy Shell → SearchPanel / Controller → private Unix socket
                                          ↓
                               fractal-search daemon
                                          ↓ worker NDJSON
                               native inference + catalog

MCP client → fractal-search-mcp (stdio) → existing catalog, read-only
```

The desktop socket is trusted same-user access: it may index folders and open originals. The MCP process is a different permission boundary; it exposes opaque identities, metadata and bounded thumbnails, never original path metadata or file bytes. Metadata can include bounded cached text excerpts; thumbnails disclose visual content. It can read alongside the desktop worker. Metadata does not load models; semantic MCP search starts its own text session lazily. Configure clients with the installed stable `~/.config/omarchy/plugins/fractal.search/fractal-search-mcp` launcher; use an absolute path. See [MCP.md](../quickshell-app/MCP.md).

| Source | Responsibility |
| --- | --- |
| `quickshell-app/ui/SetupPanel.qml`, `quickshell-app/packaging/first-run.sh` | Source-clone first-run screen and detached, pinned release bootstrap. Bundles rebase their entry to SearchPanel and do not run the bootstrap. |
| `quickshell-app/ui/SearchPanel.qml` | Panel geometry, focus, search/results, shortcuts and plugin lifecycle. |
| `quickshell-app/ui/Controller.qml` | Socket protocol, application state, search/index operations and previews. |
| `quickshell-app/ui/SettingsPage.qml` | Folders, Preferences and Performance tabs. |
| `quickshell-app/ui/AgentConnectionSettings.qml` | Read-only MCP discovery and copyable installed client configuration under Preferences. |
| `quickshell-app/ui/MediaPreview.qml`, `SearchSlider.qml` | Manual local audio/video playback, seek/volume controls and playback lifecycle. |
| `quickshell-app/ui/ResultActions.qml` | Shared Open file / Show in folder controls for preview and narrow/list layouts. |
| `quickshell-app/ui/DesktopFileOpener.qml` | GIO desktop launch, bounded diagnostics and visible opener failures. |
| `quickshell-app/ui/SearchConfirmationDialog.qml` | Shared folder-removal and stop-service confirmation. |
| `quickshell-app/ui/AcceleratorSection.qml` | Compact CPU/GPU activity and capability status with expandable, copyable diagnostics. |
| `quickshell-app/ui/RuntimeConfig.qml`, `RuntimeDiagnostics.qml`, `ServiceAutostart.qml` | Runtime paths, dependency diagnostics and login registration. The installer generates the installed RuntimeConfig. |
| `crates/fractal-search-core/src/preparation_pool.rs`, `catalog/preparation.rs` | Shared bounded CPU preparation, lookahead and native input construction. |
| `quickshell-app/backend/src/main.rs`, `daemon.rs` | Worker commands/settings and resident socket service/lifecycle. |
| `quickshell-app/backend/src/models.rs`, `watch.rs`, `diagnostics.rs` | Verified downloads, folder watching and CUDA preflight. |
| `quickshell-app/folder-picker/src/main.rs` | Independent native portal folder picker. |
| `quickshell-app/mcp/src/main.rs`, `transport.rs` | Independent stdio MCP server and bounded transport. |
| `vendor/lancedb/` | Pinned LanceDB 0.37.1 source with a manifest-only patch disabling unused Jieba/Lindera tokenizers. |
| `crates/fractal-search-core/src/` | Catalog, agent read API, decoding/preprocessing, tokenization and ONNX runtime. |
| `quickshell-app/packaging/`, `scripts/package-fractal-search.sh` | Versioned bundles, installation, uninstall and optional close binding. |
| `quickshell-app/tests/`, `scripts/` | Temporary-catalog smoke checks and optional reference-model tooling. |

EmbeddingGemma 2 natively produces a normalized 768-dimensional vector. The app projects every image, text passage, audio segment, video visual segment and query to its leading 512 dimensions, then L2-normalizes the result. Each searchable segment has one indexed vector in that shared 512-dimensional space. This is segment retrieval, not token-level multi-vector retrieval. The native runtime uses FP16 shared/text, vision and audio graphs with float32 interfaces, takes `sentence_embedding` from the backbone, and applies the search-query task prefix only to queries. The model context is 8,192 tokens; app queries remain limited to 4,096 UTF-8 bytes. Images use the pinned 280-soft-token preprocessing budget. Model assets and hashes are pinned in `crates/fractal-search-core/models/embeddinggemma2.json`; the separate native-graph and indexed-profile constants in `src/lib.rs` guard their respective contracts.

The indexed profile is `embeddinggemma2-daa72c51-fp16-multimodal-mrl512-v3`; the FP16 graph emits native 768-dimensional float32 outputs. Use a fresh, separate schema-4 catalog and explicitly reindex supported files. Old FP32 catalogs (768D or 512D) are incompatible, preserved on disk and never automatically converted or deleted. There is no 768/256 dimension selector. Raw FP32 vectors occupy 2,048 bytes instead of 3,072 bytes (one-third less); this projection alone does not reduce native graph computation. Separately, FP16 halves most weight storage; memory and latency depend on the provider and workload. See [model details](MODELS.md).

## Indexing execution

Indexing uses two shared CPU preparation workers for images, text, audio and video. Decoding, document tokenization and native input preparation can overlap inference, with bounded file/segment lookahead and queued work. Image/video thumbnail encoding also runs in the preparation workers; cache publication stays ordered. Single-image tensor batches reuse their buffers, and multi-segment files use bounded columnar write batches and indexed segment matching to reduce allocations and repeated work. Inference and catalog writes remain serial; the existing image batch setting still applies. Unchanged files retain their cheap scan path. No reindex or database reset is needed for this scheduling change. See [pool limits and validation](INDEXING-WORKERS.md).

## LanceDB catalog format

LanceDB stores folder records, typed segment metadata and normalized vectors in `catalog.lancedb/`. Each file update commits its complete segment set and vectors atomically; folder removal commits the source and all its segments together. Read-only MCP queries pin a table snapshot. All media and non-text type filters use exact cosine distance; Text mode fuses BM25 and semantic passage ranks with equal-weight RRF (k=60). Folder/type filters apply before both candidate lists. Empty-query browsing stays model-free. Hybrid scores are labeled separately from actual cosine. See [text-only hybrid retrieval](HYBRID-TEXT-SEARCH.md) for tokenizer, candidate, privacy and index-maintenance details.

This development format has no SQLite or legacy-model migration. Use a fresh `--data-dir /absolute/new/catalog` and index the desired folders. The `embeddinggemma2-daa72c51-fp16-multimodal-mrl512-v3` profile and schema 4 reject other catalog contracts, including the previous 768-dimensional profile even at schema 4; asset identities retain the opaque `asset_<UUID>` format. Lance versions are retained for concurrent readers; automatic compaction/version cleanup is not implemented yet.

Builds require Rust 1.91 or newer and the Protocol Buffers compiler (`protoc`). LanceDB is pinned to 0.37.1 with cloud/remote features disabled; 0.38.0 and 0.39.0 have an upstream local-only compilation error in their job error handling.

## Media in source control

Image, audio and video files are excluded from Git. Keep screen captures, design images and library media outside the tracked source. The tiny synthetic corrupt-JPEG regression fixture is stored as hexadecimal text and reconstructed by tests in temporary directories, so fresh clones do not require any image assets.

## Validation

Use the task-specific commands in [AGENTS.md](../AGENTS.md#validation-commands). Tests use temporary catalogs and synthetic media; keyboard/mouse automation runs only in an isolated compositor. A source edit does not update the installed app: package and install UI and backend together.

See the [documentation index](README.md) for inference, retrieval and dated validation reports. Historical results do not validate the current checkout. For distributable CPU/CUDA builds, attribution and publication, follow [release preparation](RELEASING.md).

## First-run bootstrap validation

`python3 quickshell-app/tests/smoke_first_run.py` exercises synthetic release archives, checksum rejection, repository/local-change refusal, installer rollback and interrupted-job reporting. `python3 quickshell-app/tests/smoke_setup_ui.py --native` checks the setup panel in a private compositor with a fixture installer. These do not install into the live desktop. The bootstrap pins v0.1.2 archive hashes; updating that pin requires verifying both published assets and their GitHub build attestations. Its worker runs as a transient user unit outside the replaced plugin directory. The installer retains a source recovery copy and preserves the existing data/model directories.

These checks use a fixture installer/user-unit launcher and a private native UI; they do not establish that a complete live Omarchy multi-plugin reload is crash-free. The previously observed shell reload crash remains an independent release-readiness concern. Marketplace re-review is required before removing the manual-setup designation.
