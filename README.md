# Fractal Search

A local multimodal-search app for Omarchy Shell. Add folders containing images, text, audio and video, describe what you need, and open a matching file in its default desktop app or file manager. The Quickshell panel uses native Rust EmbeddingGemma 2 inference and a LanceDB catalog. CPU and explicitly selected NVIDIA CUDA inference are supported. A separate read-only MCP server makes the indexed library available to agent clients.

This repository is the complete standalone project: all four Cargo workspace members, UI, installer and validation tools live here. The production runtime has no Python dependency. New contributors should read this file and [AGENTS.md](AGENTS.md); together they cover the product, architecture, workflow and constraints. Scoped instructions in [quickshell-app/AGENTS.md](quickshell-app/AGENTS.md) also apply when editing that directory.

## Current interface

- **Search:** a compact description field with a Settings gear icon and keyboard hints. Typing searches after a short pause; Enter submits immediately. There is no Browse all photos button. The existing empty-query Enter action still lists indexed segments.
- **Results:** a thumbnail list and selected-result preview, friendly folder names, All folders and All file types / Images / Text / Audio / Video filters, and explicit **Open file** / **Show in folder** buttons. Preferences offers list-and-preview or list-only layout; narrow windows use the list with actions beneath it. A photo grid is not implemented.
- **Settings → Folders:** add folders, see file and segment counts, remove a folder from the index, rescan folders, keep folders up to date and review indexing issues. Removing a folder preserves originals and asks for confirmation.
- **Settings → Preferences:** prepare search at login, hide after opening a file, maximum results (default 6) and result layout.
- **Settings → Performance:** model setup, a compact Accelerator section, image batch size, shortcut setup and Stop service. Accelerator checks GPU requirements automatically when shown, displays actual CPU/GPU activity, and greys out an unavailable GPU. Details expands the report; Copy diagnostics prepares it to share with your agent. Availability does not mean active GPU inference and never changes the configured provider.

Up/Down selects results, Enter opens the selection, Ctrl+Enter shows it in the file manager, and Ctrl+, opens Settings. Escape goes back or hides the panel. Ctrl+Q asks to stop the service; Cancel is focused by default, using the same confirmation component as folder removal. Left/Right switches a focused settings tab. All panels follow Omarchy’s theme and font tokens, including popup border colors, gradients and widths. Shortcut hints are clickable, with consistent spacing and visible hover/keyboard focus states.

Hiding or reloading the UI keeps the backend and indexing alive. Stopping the service drains active work and saves the catalog; summoning starts it again. **Prepare search at login** controls the actual systemd user-unit enabled state and preloads only text inference. Vision and audio encoders load lazily for the media being indexed. Idle ONNX sessions are released safely after configurable timeouts (default 15 minutes for text, 5 minutes for media); Keep warm is available.

**Image batch size** accepts 1–8 (default 1) for CPU and GPU indexing. Larger batches use more memory. Saving a change prompts **Restart now** or **Later**; the running service keeps its current size until restarted. Restart now safely drains indexing and reconnects the panel. Consecutive images share a batch; skipped/unchanged files and the final partial batch can produce smaller batches. Inference errors stop the scan and drain preparation workers; they never trigger retries at a smaller size or on another provider. Non-finite model outputs are fatal and must be diagnosed rather than indexed or silently skipped.

Indexing uses two shared CPU preparation workers for images, text, audio and video. Decoding, document tokenization and native input preparation can overlap inference, with bounded file/segment lookahead and queued work. Image/video thumbnail encoding also runs in the preparation workers; cache publication stays ordered. Single-image tensor batches reuse their buffers, and multi-segment files use bounded columnar write batches and indexed segment matching to reduce allocations and repeated work. Inference and catalog writes remain serial; the existing image batch setting still applies. Unchanged files retain their cheap scan path. No reindex or database reset is needed for this scheduling change. See [pool limits and validation](docs/INDEXING-WORKERS.md).

## Supported media and result segments

All media and queries share the pinned normalized 512-dimensional EmbeddingGemma 2 retrieval space. The pinned FP16 model emits 768 dimensions; the app keeps the leading 512 values and L2-normalizes them before indexing or search. Results represent searchable segments; one file can produce several results. Optional Text file grouping keeps the strongest document result and expandable additional passages. Counts distinguish `fileCount` (distinct source files) from `assetCount` (searchable segments).

- Images: JPEG, PNG, WebP, GIF, BMP, TIFF, HEIC and HEIF retain one embedding per still image and the 280-soft-token image budget
- Text: UTF-8 `.txt`, `.md` and `.markdown` files, at most 2 MiB, split into overlapping passages of at most 4,096 UTF-8 bytes with at most 256-byte overlap at Unicode boundaries, structural Markdown/paragraph boundaries and real formatted-input token checks, up to 1,024 passages/file. Explicit text reindexing upgrades unchanged legacy passages. This is plain-text passage indexing; Text mode adds BM25 + semantic hybrid retrieval; PDF, Office and OCR are not implemented
- Audio: decoded with FFmpeg to mono 16 kHz samples and embedded in segments up to 30 seconds. These are native audio embeddings; no transcription is generated
- Video: sampled at 1 frame per second into visual segments up to 30 seconds / 30 frames, using 140 soft tokens per frame. An available soundtrack is additionally indexed as separate audio segments with its own vectors and timestamps. Visual and audio matches can therefore refer to the same video file

Audio/video files are limited to 2 GiB and two hours. Video sources may contain up to 16,777,216 pixels with sides up to 8,192 pixels. Sampled frames are downscaled, preserving aspect ratio to pixel rounding and never upscaling, to at most 2,073,600 pixels and a 4,096-pixel side before model preprocessing. Originals remain unchanged. Ambiguous `.ts` files are checked for transport-stream headers and text content: TypeScript source is excluded, while binary inputs still undergo normal decoder validation. Audio is resampled to mono 16 kHz and trimmed to the segment’s exact sample budget (at most 480,000 samples), with normalized output timestamps. The system `ffmpeg` and `ffprobe` commands are explicit audio/video dependencies; missing tools, unsupported codecs and decode failures are reported. FFmpeg is not bundled, and dependency diagnostics do not prove codec support or successful inference. Decoder commands are confined to local files and explicit supported demuxers, with bounded output and timeouts.

A video soundtrack decode failure is a visible partial success: all visual segments and every valid audio segment commit together, including valid audio after a failed interval. Only failed audio intervals are omitted; stale rows for those intervals are removed. **Indexing issues** reports the skipped count and intervals (with bounded display text and full decoder errors in the backend log). A generation-bound retry marker ensures the next rescan retries incomplete audio, including after service restart; unchanged visual and valid audio vectors are reused. Segments are decoded again to verify them, so this is embedding reuse rather than decode caching. Cancellation, source changes, visual decode failures, native input preparation, model/inference, database and cache-write failures do not use this exception. Standalone audio decode failures still skip the file and preserve its previous complete rows. A partial video counts as both updated and a file with an issue; progress advances once. No catalog reset is needed.

Rows show media kind, passage number or time range, and cached text excerpts (at most 2,048 UTF-8 bytes). Images have bounded previews; video shows a cached poster until you press Play. Audio and video play inline in the preview pane, starting at the selected match’s timestamp and continuing through the original file. Controls provide play/pause, seeking, elapsed/total time, mute and volume. Playback stops and releases the file when selection changes, the pane is hidden (including narrow/list-only layouts), Settings opens, or the panel closes. Loading/codec errors stay visible; playback never starts automatically. Qt Multimedia and its FFmpeg backend are required (`qt6-multimedia` and `qt6-multimedia-ffmpeg` on Omarchy). Text/audio do not request image previews. **Open file** opens the whole original with its default desktop app; it does not request seeking to the matched segment. **Show in folder** reveals the original. Files stay in place.

Opening uses GLib's `gio open`, which resolves the local file's MIME association
and honors terminal-based desktop entries. Fractal Search does not override
desktop defaults. Missing opener/application errors and launch timeouts stay
visible; the panel hides or releases focus only after GIO accepts the launch.
An application can still fail after accepting the request.

## Build, install and open

**Manual setup is required.** Use the source build below or a prebuilt bundle's `install.sh`. Omarchy's generic `omarchy plugin add` command only clones the repository; it does not build the native binaries, create the user service or generate runtime paths. Do not enable a raw clone as a working installation.

From this repository's root:

```bash
# CPU development build, installation and summon
./start-quickshell-search.sh --dev

# CUDA-capable build with explicit GPU selection
./start-quickshell-search.sh --dev --cuda --provider cuda --device-id 0
# Add --cuda-lib-dir /absolute/persistent/path/to/lib if libraries are installed privately.
```

Omit `--dev` for a release build. First-run **Set up model**, also available under Settings → Performance, downloads and verifies the pinned assets (about 1.50 GB). **Choose folder** selects an existing verified model cache. Do not download models just to run metadata checks.

Development builds optimize the native core's image/preprocessing loops at level 3 while retaining the app's development profile and level-2 dependency optimization. Workspace code keeps level-1 debug information, dependencies omit debug information, and incremental compilation remains enabled for faster rebuilds. Shared dependency versions are centralized in the root manifest; see the [dependency review](docs/DEPENDENCIES.md) for the graph and tradeoffs. CPU/CUDA inference fetches only the sentence embedding; CUDA's CPU worker pools sleep while waiting for work instead of spinning. These execution optimizations retain the indexed vector contract. The [synthetic indexing benchmark](quickshell-app/README.md#synthetic-indexing-benchmark) records scan timings, CPU usage and search-score parity between binaries.

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

Open an existing installation without building or touching the checkout:

```bash
omarchy-shell shell summon fractal.search '{}'
# Equivalent repository convenience command:
./start-quickshell-search.sh --no-build --summon
```

Use the installed summon command for a desktop shortcut. Installation preserves existing shortcuts; optional `--bind-close` explicitly installs managed Super+W close routing. Do not restart the desktop shell to update this plugin.

Development needs Rust, a C/C++ toolchain, pkg-config and native image libraries. The runtime needs Omarchy Shell/Quickshell, curl, jq, libheif, libjpeg-turbo, gcc-libs, GLib's `gio`, ffmpeg and ffprobe and a desktop portal providing FileChooser. Show in folder uses `gdbus` and a file manager implementing FileManager1.ShowItems. The Rust build selects system libjpeg-turbo on Linux. Python is used only by development/validation tools.

Native inference uses pinned ONNX Runtime 1.24.2 (`ort` rc.12). CUDA bundles include its matching provider libraries; NVIDIA driver, CUDA 12 runtime and cuDNN 9 libraries are installed separately. Use `--cuda` when updating a CUDA installation; a CPU bundle refuses to replace a CUDA configuration unless you explicitly switch to `--provider cpu`. Startup/inference failures remain visible and never retry the model on CPU. See [CUDA.md](quickshell-app/CUDA.md) for exact dependencies and validation.

CUDA sessions grow their memory arenas to the requested size and return unused scratch buffers after each model call. This prevents earlier image/audio/video work from retaining unnecessary GPU capacity across the three resident graphs. Weights remain loaded according to the existing idle policy. Allocation failures still stop the scan; no model, precision, batch-size or provider fallback is used.

To package without installing:

```bash
scripts/package-fractal-search.sh --dev --output /absolute/new/bundle-directory
# For an already built CUDA debug runtime:
scripts/package-fractal-search.sh --dev --no-build --cuda --output /absolute/new/cuda-bundle-directory
```

Output directories must not already exist. `--no-build` packages all three existing binaries and checks their CPU/CUDA capability; it does not rebuild them. Install with the bundle's `install.sh`. UI and backend must be installed together. Updates drain the backend before swapping files; preserve its previous active state when using the installer directly. The repository launcher summons it afterwards.

After a successful install, obsolete runtime builds are removed automatically. The current and immediately previous builds are retained, including across reinstalls; builds still running or containing the configured CUDA library directory are retained until a later update. Failed installs do not prune builds. Cleanup is limited to recognized runtime bundle directories and preserves catalogs, models, shared CUDA libraries and unrelated files. Cleanup errors are reported without rolling back the successful installation.

## Removal and permissions

For complete application removal, run the installed runtime's uninstaller **before** using Omarchy's generic plugin removal:

```bash
SEARCH_PLUGIN_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins/fractal.search"
SEARCH_INSTALLED_RUNTIME=$(cat "$SEARCH_PLUGIN_DIR/.fractal-search-install")
"$SEARCH_INSTALLED_RUNTIME/uninstall.sh"
```

This stops and disables the user service, removes the panel and managed service unit, and removes the optional managed close binding. Catalogs, models and runtime bundles remain on disk. `omarchy plugin disable fractal.search` hides the panel only; indexing and the service continue. `omarchy plugin remove` also removes the panel's installation marker, so it is not the application's uninstaller. Updates use the new bundle's installer rather than `omarchy plugin update`.

The app runs as your user and requires no root service. It reads selected folders, writes its catalog/caches under the XDG data/cache directories, and launches original files only through desktop actions. Installation writes the user plugin, a systemd user unit and versioned runtime files; optional shortcut changes require an explicit installer option or UI action. System dependencies must be installed separately. Model setup downloads pinned assets from Hugging Face over HTTPS and verifies their sizes and SHA-256 hashes; indexing and search run locally. The same-user desktop socket can read originals; the separate read-only MCP boundary is described below.

## Marketplace publication

The repository has one root [manifest](manifest.json). Packaging rebases its panel path for the bundle; installation generates the selected runtime paths. Request the marketplace's **manual-setup** designation before approval so the listing directs users to these instructions instead of offering the generic install command. Suggested category: **Productivity**; tags: **ai**, **launcher**, **media**. The preview is optional; media remains excluded from this repository.

Follow the [current publishing guide](https://plugins.omarchy.org/publish.html) and [submission requirements](https://github.com/omacom/omarchy-plugin-marketplace/blob/main/SUBMISSION.md). The installer and user-service management require maintainer review under the [automated security baseline](https://github.com/omacom/omarchy-plugin-marketplace/blob/main/SECURITY.md#automated-security-baseline). Local validation is not marketplace approval: submit a committed, tested snapshot and request `manual-setup` before the maintainer applies `approved-and-verified`. Do not claim standard one-click installation support.

Fractal Search is [MIT licensed](LICENSE), copyright 2026 nigh8w0lf (Ashwin Cheekati). Vendored LanceDB retains its [Apache-2.0 license](vendor/lancedb/LICENSE) and [patch provenance](vendor/lancedb/PATCHES.md). Dependencies and downloaded models retain their own licenses. Publishing prebuilt binaries additionally requires a complete third-party notice/redistribution review; the current development bundle's [NOTICE](quickshell-app/packaging/NOTICE) is an overview, not a complete binary license inventory.

## Architecture and source map

```text
Omarchy Shell → SearchPanel / Controller → private Unix socket
                                          ↓
                               fractal-search daemon
                                          ↓ worker NDJSON
                               native inference + catalog

MCP client → fractal-search-mcp (stdio) → existing catalog, read-only
```

The desktop socket is trusted same-user access: it may index folders and open originals. The MCP process is a different permission boundary; it exposes opaque identities, metadata and bounded thumbnails, never original path metadata or file bytes. Metadata can include bounded cached text excerpts; thumbnails disclose visual content. It can read alongside the desktop worker. Metadata does not load models; semantic MCP search starts its own text session lazily. Configure clients with the installed stable `~/.config/omarchy/plugins/fractal.search/fractal-search-mcp` launcher; use an absolute path. See [MCP.md](quickshell-app/MCP.md).

| Source | Responsibility |
| --- | --- |
| `quickshell-app/ui/SearchPanel.qml` | Panel geometry, focus, search/results, shortcuts and plugin lifecycle. |
| `quickshell-app/ui/Controller.qml` | Socket protocol, application state, search/index operations and previews. |
| `quickshell-app/ui/SettingsPage.qml` | Folders, Preferences and Performance tabs. |
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

The indexed profile is `embeddinggemma2-daa72c51-fp16-multimodal-mrl512-v3`; the FP16 graph emits native 768-dimensional float32 outputs. Use a fresh, separate schema-4 catalog and explicitly reindex supported files. Old FP32 catalogs (768D or 512D) are incompatible, preserved on disk and never automatically converted or deleted. There is no 768/256 dimension selector. Raw FP32 vectors occupy 2,048 bytes instead of 3,072 bytes (one-third less); this projection alone does not reduce native graph computation. Separately, FP16 halves most weight storage; memory and latency depend on the provider and workload. See [model details](docs/MODELS.md).

## Runtime locations and troubleshooting

Defaults below follow XDG configuration/data/cache roots:

| Location | Contents |
| --- | --- |
| `~/.config/omarchy/plugins/fractal.search/` | Installed plugin and stable MCP launcher. |
| `~/.config/systemd/user/fractal-search.service` | User service. |
| `~/.local/share/fractal-search/` | `catalog.lancedb/`, `settings.json`, thumbnails, previews and service lock. |
| `~/.local/share/fractal-search/runtime/<version>/` | Versioned runtime bundles. |
| `~/.cache/fractal-search/models/embeddinggemma2-daa72c51-fp16/` | Verified model assets. |
| `$XDG_RUNTIME_DIR/fractal-search/backend.sock` | Private desktop socket. |

Provider/device/private-library settings are preserved in the installed `.fractal-search-runtime.json`. UI preferences and model location persist in the data directory; login registration lives in systemd. Uninstall preserves the catalog and model cache. Never use these live paths for tests.

At startup, the desktop backend creates an empty catalog if `catalog.lancedb/` is absent or completely empty, and logs its creation. If you delete the catalog while the service is running, stop and reopen Fractal Search to recreate it, then add folders and index them again. Settings and model caches are retained. A partially deleted, corrupt or incompatible catalog produces an error instead of being replaced; the read-only MCP server never creates a catalog.

```bash
systemctl --user status fractal-search.service
journalctl --user -u fractal-search.service -n 80 --no-pager
omarchy-shell shell call fractal.search ping ''
```

A source edit is not an installed update: rebuild/package/install to change the running plugin. If installation hits the shell IPC timeout, inspect the error and service state; `OMARCHY_SHELL_IPC_TIMEOUT=15s /path/to/bundle/install.sh` allows a longer load/rescan window. Do not hide errors or restart the whole desktop as a workaround.

## Validation and further reading

[AGENTS.md](AGENTS.md) contains runnable checks and task-specific validation guidance. The native UI smoke uses a private nested compositor, temporary catalogs and synthetic media to check filters, typed queries, settings, safe restart and viewer resummoning. Shared-desktop keyboard/mouse automation is prohibited.

- [Detailed plugin usage and historical validation](quickshell-app/README.md)
- [CUDA runtime requirements and real-inference validation](quickshell-app/CUDA.md)
- [MCP protocol, client configuration and privacy limits](quickshell-app/MCP.md)
- [Model contract and reference fixtures](docs/MODELS.md)
- [FP16 graph validation and deployment boundary](docs/FP16-VALIDATION.md)
- [Earlier 512D default validation](docs/EMBEDDING-512-VALIDATION.md)
- [Text-only hybrid retrieval and validation](docs/HYBRID-TEXT-SEARCH.md)
- [QMD-inspired improvements, indexing/reuse, idle policies and limitations](docs/QMD-SEARCH-IMPROVEMENTS.md)
- [Labeled retrieval evaluation](docs/QMD-EVALUATION.md)
- [QMD validation results and blocked checks](docs/QMD-VALIDATION.md)
- [Local QMD/512 integration checkpoint and unrun validation](docs/QMD-LOCAL-INTEGRATION.md)

Model weights and generated bundles are not checked in. Repository licensing and manual installation are documented above; public binary release/signing and redistribution review remain separate work.

## LanceDB catalog format

LanceDB stores folder records, typed segment metadata and normalized vectors in `catalog.lancedb/`. Each file update commits its complete segment set and vectors atomically; folder removal commits the source and all its segments together. Read-only MCP queries pin a table snapshot. All media and non-text type filters use exact cosine distance; Text mode fuses BM25 and semantic passage ranks with equal-weight RRF (k=60). Folder/type filters apply before both candidate lists. Empty-query browsing stays model-free. Hybrid scores are labeled separately from actual cosine. See [text-only hybrid retrieval](docs/HYBRID-TEXT-SEARCH.md) for tokenizer, candidate, privacy and index-maintenance details.

This development format has no SQLite or legacy-model migration. Use a fresh `--data-dir /absolute/new/catalog` and index the desired folders. The `embeddinggemma2-daa72c51-fp16-multimodal-mrl512-v3` profile and schema 4 reject other catalog contracts, including the previous 768-dimensional profile even at schema 4; asset identities retain the opaque `asset_<UUID>` format. Lance versions are retained for concurrent readers; automatic compaction/version cleanup is not implemented yet.

Builds require Rust 1.91 or newer and the Protocol Buffers compiler (`protoc`). LanceDB is pinned to 0.37.1 with cloud/remote features disabled; 0.38.0 and 0.39.0 have an upstream local-only compilation error in their job error handling.

## Media in source control

Image, audio and video files are excluded from Git. Keep screen captures, design images and library media outside the tracked source. The tiny synthetic corrupt-JPEG regression fixture is stored as hexadecimal text and reconstructed by tests in temporary directories, so fresh clones do not require any image assets.
