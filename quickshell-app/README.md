# Fractal Search

An Omarchy Shell plugin with a Quickshell search panel and a bundled native Rust EmbeddingGemma 2 ONNX service. The panel runs inside the existing desktop shell; a systemd user service owns the model worker independently of UI reloads. Add a folder, index supported image/text/audio/video content, and search with a description. It generates thumbnails and bounded previews; originals stay in their selected folders. Image decoding, preprocessing and model inference run in the native service; audio/video decoding uses explicit system ffmpeg/ffprobe commands.

`crates/fractal-search-core/` implements native decoding, bounded multimodal preprocessing, the pinned 8,192-token tokenizer contract and the LanceDB segment catalog. The workspace builds the plugin backend, native folder chooser and read-only MCP server together.

## Search and settings

Type a description to search. Results use a compact list and a larger selected-result preview; **Open file** opens the whole original in its default app and **Show in folder** selects it in your file manager. Both actions remain available in the list-only layout and narrow windows. Arrow keys select results, Enter opens the selected file, Ctrl+Enter reveals it, and Escape goes back. There is no Browse all photos button.

Settings has three tabs:

- **Folders:** add search folders, see their file and segment counts, rescan, keep folders up to date, review skipped files, or remove a folder from the index. Removal always asks for confirmation and preserves originals.
- **Preferences:** prepare search at login, hide after opening a file, and choose the result count and layout.
- **Performance:** set up the search engine, inspect GPU requirements, copy diagnostics, choose image batch size, configure a launcher shortcut, and stop the service safely.

The tabs remain visible while their contents scroll. Left/Right switches a focused tab. All panels use the active Omarchy theme; confirmations share the same component and default to Cancel.

## Supported media and result segments

All media and queries share the pinned normalized 512-dimensional EmbeddingGemma 2 retrieval space. The unchanged FP32 model emits 768 dimensions; the app keeps the leading 512 values and L2-normalizes them before indexing or search. Results represent searchable segments; one file can produce several results. Counts distinguish `fileCount` (distinct source files) from `assetCount` (searchable segments).

- Images: JPEG, PNG, WebP, GIF, BMP, TIFF, HEIC and HEIF retain one embedding per still image and the 280-soft-token image budget
- Text: UTF-8 `.txt`, `.md` and `.markdown` files, at most 2 MiB, split into overlapping passages of at most 4,096 UTF-8 bytes with at most 256-byte overlap at Unicode boundaries, structural boundaries and actual formatted-input token checks, up to 1,024 passages/file. Explicit text rebuild upgrades unchanged legacy passages. This is plain-text passage indexing; Text mode adds BM25 + semantic hybrid retrieval; PDF, Office and OCR are not implemented
- Audio: decoded with FFmpeg to mono 16 kHz samples and embedded in segments up to 30 seconds. These are native audio embeddings; no transcription is generated
- Video: sampled at 1 frame per second into visual segments up to 30 seconds / 30 frames, using 140 soft tokens per frame. An available soundtrack is additionally indexed as separate audio segments with its own vectors and timestamps. Visual and audio matches can therefore refer to the same video file

Audio/video files are limited to 2 GiB and two hours. Video decoding rejects frames over 2,073,600 pixels or a 4,096-pixel side instead of silently scaling them. The system `ffmpeg` and `ffprobe` commands are explicit audio/video dependencies; missing tools, unsupported codecs and decode failures are reported. FFmpeg is not bundled, and dependency diagnostics do not prove codec support or successful inference. Decoder commands are confined to local files and explicit supported demuxers, with bounded output and timeouts.

Choose **All media**, **Images**, **Text**, **Audio** or **Video** alongside the folder filter. **Text** combines literal BM25 terms with semantic passage retrieval using RRF (k=60); the other modes retain cosine order. Empty-query browsing filters without loading a model. Hybrid scores have a separate label and are never percentages. The full passage plus filename is indexed internally while returned excerpts stay bounded. See the [hybrid contract](../docs/HYBRID-TEXT-SEARCH.md) for multilingual tokenization, candidate limits, fresh-schema requirements and index-maintenance resource caveats.

Rows show media kind, passage number or time range, and cached text excerpts (at most 2,048 UTF-8 bytes). Images have bounded previews; videos show an existing cached frame thumbnail. Text/audio do not request image previews. **Open file** opens the whole original with its default desktop app; it does not request seeking to the matched segment. **Show in folder** reveals the original. Files stay in place.

## Install and launch

For a prebuilt bundle, run its `install.sh`, then:

```bash
omarchy-shell shell toggle fractal.search
```

The installer installs both the `fractal.search` plugin and its native backend. Users need no compiler, Python or venv. Runtime requirements are Omarchy Shell/Quickshell, curl, jq, libheif, libjpeg-turbo, gcc-libs, ffmpeg and ffprobe, plus xdg-desktop-portal with a FileChooser backend (xdg-desktop-portal-gtk on Omarchy). Missing dependencies and portal failures produce visible errors. ONNX Runtime is linked into the binary. Tested with Quickshell 0.3.1 and Hyprland 0.56.2; the visual-mask protocol requires Hyprland 0.47 or newer.

For development from this repository:

```bash
./start-quickshell-search.sh --dev
```

This builds a bundle, installs the plugin/backend and opens it. `--no-build --summon` opens the installed plugin without rebuilding or reinstalling; existing Super+I bindings using that launcher continue to work. Omit `--dev` for a release build. Developers require Rust, pkg-config and a C/C++ toolchain; native libraries are selected through pkg-config. The root workspace uses system TurboJPEG and does not require CMake for that dependency. LanceDB requires Rust 1.91 or newer and `protoc`. The catalog stores metadata and vectors in `catalog.lancedb/`. Use a fresh data directory and reindex for this development format; see the root README.

Use **Set up model** on first open to download the pinned assets (~2.96 GB) with visible progress. Completed assets are checked by size/SHA-256 and reused. A failed download or corrupt asset is reported, never substituted. **Stop** cancels a download and stops the service, retaining verified files. Settings → Performance → **Choose folder** selects an existing pinned cache. Models are never stored under `/tmp` or bundled into source control.

`--model-dir PATH` and `--threads N` can configure a development installation. CPU is the default; the thread default is half the available logical CPUs, rounded up and capped at six. Providers are never silently selected. The installed service prepares the text-search model in the background on startup; the vision and audio models load only when indexing needs them. Loaded models remain resident while the backend runs. Standalone developer commands retain lazy loading unless passed `--warm-search`.

### CUDA acceleration and diagnostics

Performance also offers **Image batch size**, from 1 to 8 with a default of 1, for both CPU and CUDA indexing. Saving it prompts **Restart now** or **Later**; the current worker keeps its active size until restart, and a pending change remains visible. Restart now safely drains indexing, preserves completed files and reconnects without closing the desktop shell. Larger batches use more memory; consecutive images share inference with the same fixed preprocessing budget, preserving each image's embedding and the catalog profile. Skipped/unchanged files and batch tails can produce smaller batches. Inference failures stay fatal without batch-size or provider retries.

Build a CUDA-capable bundle and explicitly select NVIDIA inference:

```bash
./start-quickshell-search.sh --dev --cuda --provider cuda --device-id 0
# Add --cuda-lib-dir /absolute/path/to/libraries for a private runtime installation.
```

CUDA bundles include ONNX Runtime's CUDA provider libraries. They require a compatible NVIDIA driver, CUDA 12 runtime libraries and cuDNN 9; the CUDA compiler/toolkit is not needed for inference. The required library major versions must match even when newer CUDA packages are installed. See [ONNX Runtime's requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements). NVIDIA libraries are not included in the bundle.

**Settings → Performance → Accelerator** lists the detected GPU/driver, build support, required shared libraries, loader errors and setup steps. The compact row shows CPU/GPU status and greys out GPU when requirements are unmet. Requirements are checked automatically when this section is first shown. **Details** expands the report; **Copy diagnostics** copies it for sharing with your agent. Refresh works independently of the model worker, including when it cannot start. Availability is distinct from active inference; no provider is changed automatically. This is a dependency preflight; model initialization and inference still determine whether the accelerator works. The equivalent command is `fractal-search diagnostics --provider cuda --device-id 0`; it does not open the catalog or load models. `fractal-search providers` reports compiled provider support. See [CUDA.md](CUDA.md) for reproducible GPU validation and runtime versions.

The installer preserves the provider, device and private library directory across updates. Use `--cuda` when rebuilding a CUDA installation; a CPU-only bundle refuses to replace an active CUDA configuration. Explicitly choose `--provider cpu` to switch back. The installed service, diagnostics and stable MCP wrapper share the same loader path, including the bundled ONNX provider directory. Do not point persistent installations at temporary library directories. CUDA startup/inference errors remain visible and never retry the model on CPU. ONNX Runtime can still assign shape/control operations to CPU within a CUDA session; the GPU validation example checks that its major matrix/attention operations run on CUDA.

### Read-only MCP server

The bundle also installs `~/.config/omarchy/plugins/fractal.search/fractal-search-mcp`, a stable stdio command for agent clients. It exposes status, filtered browsing, semantic and text-hybrid search, metadata and bounded thumbnails from the Quickshell catalog. It can run beside the indexing service, opens the existing database read-only, and excludes original paths and original file bytes. See [MCP.md](MCP.md) for configuration, tool limits and CPU/CUDA selection. The panel's trusted desktop socket and the read-only MCP server have distinct permissions and protocols.

### Packaging

Create a prebuilt bundle without installing it:

```bash
scripts/package-fractal-search.sh --output /absolute/path/to/fractal-search-bundle
# Or --dev --no-build to package the existing debug binary.
```

The bundle includes QML, native backend, installer/uninstaller, pinned model manifest, runtime notes and SHA-256 checksums. It is a local development artifact; public publication/signing and a complete redistribution-license review remain separate release work. Omarchy's generic plugin installer clones QML repositories and does not run installation hooks: use this product's bundle installer to obtain the backend too.

UI files: `$XDG_CONFIG_HOME/omarchy/plugins/fractal.search`. Backend/unit: `$XDG_DATA_HOME/fractal-search/runtime/<bundle-version>` and `$XDG_CONFIG_HOME/systemd/user/fractal-search.service`. The unit starts on demand; installation does not enable it at login. Updates drain the existing backend before swapping files, use versioned QML paths to avoid stale host component caches, and wait for the new panel to finish loading. Logs are available through `journalctl --user -u fractal-search.service`.

Esc goes back within settings/results and hides the search bar, leaving indexing/watching running. With the optional close binding installed, Super+W hides the whole focused panel or cancels its focused folder picker; other application windows close normally. Ctrl+Q and Settings → Performance → Stop service ask for confirmation, with Cancel focused by default. Confirming drains the backend and hides the panel; the desktop shell continues running. Summoning starts the backend again. UI reloads reconnect to active work and restore progress/skipped-file reports. Disabling a plugin removes its UI; stop background work with `systemctl --user stop fractal-search.service`, or run the installed bundle's `uninstall.sh` to disable/remove both. Uninstall retains the catalog and model cache.

## Using it

The compact search bar shows Search and Settings actions, alongside Super+W **Close** when the desktop binding is installed. Ctrl+Q remains a keyboard shortcut for the Stop service confirmation and is shown with the Stop service button in Settings. Esc still works from the keyboard and appears as **Back** in results/settings.

**Settings → Preferences → Prepare search at login** preloads text search in the background before the panel is opened. Opening Fractal always starts a stopped service and prepares search; this setting enables its systemd user unit at the graphical login instead. Turning it off affects the next login and does not stop the current service. The checkbox reads the actual enabled state, persists across updates, and reports systemd failures visibly. Installation leaves it off unless already enabled; uninstall disables its registration. The unit drains and stops with the graphical session. No system-wide boot service or automatic model download is installed. Missing models remain in first-run setup; corrupt cached models produce a visible warmup error.

Text-only background preparation measured roughly 0.95–0.97 GiB worker RSS plus 24 MiB daemon RSS on the development machine, or about 1 GB in total. Indexing additionally loads the vision model; the login option does not preload that model. These measurements exclude the shared desktop UI and filesystem page cache.

1. The app opens as a search bar with a hotkeys footer (680 × 95 logical pixels on a sufficiently large screen). Start typing a description to search after a 180 ms pause, or press Enter to search immediately. Enter with an empty query browses indexed segments. The same panel expands over 160 ms to reveal results and an inline preview; Escape collapses it back to the bar and footer. The footer stays visible in every view and shows the available actions as lightly outlined buttons with separate shortcut keycaps. Use the keyboard shortcut or click the button; hovering highlights the action and shows a hand cursor. Unavailable actions are dimmed, and the footer wraps on narrow screens.
2. Open **Settings** with the gear icon or Ctrl+, then select **Folders → + Add folder** and choose a directory in the native picker. Return with the header back arrow or Esc; Ctrl+, is only available outside Settings. Supported image formats are JPEG, PNG, WebP, GIF, BMP, TIFF, HEIC and HEIF; text/audio/video policies are listed above. Actual content determines decoding, including WebP files with a `.jpg` suffix. Symlinks are not followed.
3. Results show a vertically navigable media/file/folder list, with type and passage/time metadata, text excerpts or visual previews. Up/Down changes the selection; Enter opens the whole original in its default desktop app; Ctrl+Enter reveals it in the file manager. Double-clicking a row also opens the original. Previews preserve aspect ratio, are bounded to a maximum side of 1,600 pixels, and do not upscale small images. Image cache repair may read originals in the trusted desktop service; video previews only display the cached frame thumbnail.
4. Use **All folders** to restrict results to one indexed folder. Settings → Preferences controls **Maximum results** (1–100, default 6) and **Result layout** (list with preview or list only). Narrow windows use the list to keep filenames and navigation usable. Catalog browsing has a **Load more** button.
5. Settings → Folders lists indexed folders with their file and segment counts. **×** removes a folder from the index after confirmation; original files stay in place. **Rescan folders** scans the configured folders in sequence. Unchanged files keep their vectors and stable IDs. Missing image thumbnails are repaired without reembedding; video previews use cached frame thumbnails. Modified files are updated, and removed files leave the catalog only after a completed, uncancelled traversal. Overlapping sources are rejected. A reindex batch pauses on skipped files so their report remains available.
6. **Settings → Folders → Keep folders up to date** enables automatic updates while the service is running. **Hide after opening a file** dismisses the panel after a successful desktop-open request; the service stays resident. Escape goes back from settings/results, then dismisses the summon prompt. Ctrl+Q opens the Stop service confirmation. The explicit plugin `quit` call stops the backend cleanly after active work finishes without a UI confirmation.

Omarchy owns the plugin lifecycle and exposes these commands:

```bash
omarchy-shell shell summon fractal.search '{}'
omarchy-shell shell toggle fractal.search
omarchy-shell shell summon fractal.search '{"query":"a red bicycle"}'
omarchy-shell shell hide fractal.search
omarchy-shell shell call fractal.search quit ''
```

Settings shows the toggle command for a desktop shortcut. The repository launcher preserves the previously configured Super+I binding. The installer leaves keybindings alone by default; `./start-quickshell-search.sh --dev --bind-close` (or a prebuilt bundle's `./install.sh --bind-close`) explicitly integrates Super+W through a marked block in personal `hypr/bindings.lua`. It backs up that file, refuses an unrelated personal Super+W override, reloads/validates Hyprland, and preserves the managed block across updates. Uninstall removes only that block. The close router checks actual panel focus, targets the original application address outside Fractal, and reports IPC failures visibly instead of closing the application underneath. The footer shows Super+W only when that binding is installed. The status bar is unchanged.

Corrupt, unsupported, unreadable or changing supported files are skipped individually, with a visible failure count and a **Skipped files** report containing their paths and decoder errors. Healthy files later in the directory continue indexing. A failed rescan preserves that file's complete last indexed segment set; repairing it and rescanning retries it. Reported TurboJPEG corruption warnings remain errors, with no decoder fallback. Directory traversal, model inference, database and cache-write errors still stop the scan and remain visible. Cancellation never prunes unvisited entries.

Use **Dismiss** to acknowledge a completed notice or error and return to the compact search bar. The last skipped-file report remains available in **Settings → Folders → Skipped files** until the next scan; new failures show a fresh banner. Active indexing and shutdown cannot be dismissed this way. Report/error dialogs are parented to the persistent canvas so they follow Quickshell’s recreated native window after hide/resummon. They stay inside the panel, support mouse and Escape, and return focus to search when closed. Clicking outside the panel hides it without clearing its diagnostics.

Indexing progresses by complete files and segments. Progress stays visible, and **Stop** finishes in-flight inference and preserves completed files. Incomplete multi-segment replacements are discarded, retaining the previous complete rows. Images use the fixed 280-soft-token budget; queries use the search-query prefix and the 8,192-token model context with a 4,096 UTF-8-byte app bound. Inputs exceeding the model context fail rather than being silently truncated. Decoding rejects images above 100 megapixels, sides above 65,536 pixels, files above 128 MiB, and unsupported high-bit-depth input.

The plugin uses Omarchy’s shared `Color` and `Style` singletons, including popup colors, fonts and spacing scale. Theme changes update the existing host bindings. A centered Quickshell layer-shell panel owns its geometry without reserving desktop space or changing window rules. Results expand to 980 × 640 and settings to 520 × 740 logical pixels, capped to the current screen with a 16-pixel margin. During each 160 ms transition, the painted panel animates inside a fixed transparent rendering surface (up to 980 × 740), so the compositor does not resize or reposition it a second time. The visible panel and mouse input region fit the compact bar when idle; transparent surroundings are excluded from the input region. Clicking outside dismisses the panel and releases desktop input. Matching input and compositor-visible regions keep the surrounding area click-through and prevent pale resize edges. The idle launcher remains 680 × 95 logical pixels on a sufficiently large screen. Results fade in as the panel grows, and errors/setup/indexing messages expand it enough to remain visible. Settings reserve a scrollbar gutter so controls stay clear of the thumb. Native folder pickers and successful external open/reveal requests release keyboard focus and lower the panel behind desktop windows. Summoning it or closing a picker restores its focus. Hiding or reloading the panel detaches the UI without stopping indexing or the shell. Explicit Stop service drains in-flight inference, saves completed files, discards incomplete multi-segment replacements and releases the worker.

The panel inherits Omarchy’s Qt controls style. It uses Hyprland’s popup focus grab with on-demand layer focus. Native folder selection uses the desktop FileChooser portal through the bundled `fractal-folder-picker` executable. The desktop selects its dialog implementation in a separate process; our picker never loads GTK/GVfs into the shared shell. Missing/old portals, helper failures and malformed responses fail visibly, with no alternate picker fallback. The helper does not load models or open the catalog. Hiding/reloading the panel cancels its request; closing the picker restores panel focus. The launcher does not force an X11/Wayland backend or GPU.

The card appears after Qt has submitted its first correctly sized buffer. The native rendering surface stays fixed across summon, results, settings and report transitions; only the painted panel and its masks animate. This avoids provisional-size clipping and competing native/QML geometry animations. Dialogs use in-panel popups with one panel transition instead of an additional popup animation.

## Data and model compatibility

Default runtime data lives in `$XDG_DATA_HOME/fractal-search/` (normally `~/.local/share/fractal-search/`): `catalog.lancedb/`, `settings.json`, `service.lock`, `thumbnails/` and `previews/`. A process lock prevents two services opening the same catalog. Do not select this directory or one of its ancestors as an image source. Models default to `$XDG_CACHE_HOME/fractal-search/models/embeddinggemma2-daa72c51/`.

The desktop backend creates and logs an empty catalog at startup when `catalog.lancedb/` is missing or completely empty. After deleting a catalog from a running service, stop and reopen Fractal Search, then add and index folders again; settings and model caches remain intact. Partial deletion, corruption and incompatible schemas remain visible errors. The read-only MCP server never creates or repairs a catalog.

The normalized 512-dimensional catalog records profile `embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2`. The unchanged native FP32 graph still emits 768 dimensions; all modalities and queries use leading-512 truncation followed by L2 normalization. The app does not expose a dimension selector. Schema 4 adds a full-passage lexical contract and named FTS index alongside the FP32 shared/text, vision and audio exports, image/video preprocessing budgets, text passage formatting and audio feature extraction. The new indexed profile requires a separate fresh catalog and explicit reindexing. Existing 768-dimensional schema-4 catalogs fail visibly and are never converted, overwritten or deleted automatically. The same pinned model assets are reused. Raw vectors use 2,048 rather than 3,072 bytes each; this does not reduce model inference cost or VRAM. Model assets and licensing are described in the [model manifest](../crates/fractal-search-core/models/embeddinggemma2.json) and [model guide](../docs/MODELS.md).

## Desktop transport

Uncached semantic searches emit `searchProgress` events with the request `id` and `payload.stage`. A cold text model emits `starting` while its assets are verified and loaded, then `searching` before inference. Warm searches emit only `searching`; cached result pages need neither stage. With `--warm-search`, `ready.payload.textModelWarming` announces background preparation, followed by `warmupFinished` with `{ok, error}`. Status includes `textModelLoaded`; reconnect snapshots retain warmup failures. The compact placeholder displays “Starting search service…” during service/model startup without expanding for startup alone. Typed queries wait for readiness. The UI displays “Searching your library…” during query search and “Loading results…” during catalog browsing. Startup failures remain visible as errors. The gear icon opens the tabbed settings view; Ctrl+, remains available outside Settings and appears in the compact footer.

`target/debug/fractal-search serve` reads newline-delimited JSON from stdin and emits JSON on stdout; diagnostics are on stderr. One worker owns catalog and models. This is a **trusted local desktop API**, including writes and original filesystem paths, separate from the read-only `fractal-search-mcp` privacy boundary. The plugin uses `fractal-search daemon` over a private Unix socket, restricted to the current user. There is no network listener and it is not an MCP server. The daemon keeps a worker alive across UI disconnects, translates IDs per session and restores active progress/results on reconnection. A second simultaneous UI is rejected explicitly.

```json
{"id":1,"command":{"type":"index","directory":"/absolute/path/to/media"}}
{"id":2,"command":{"type":"search","query":"a red bicycle","offset":0,"limit":50}}
{"id":3,"command":{"type":"shutdown"}}
```

Responses have `id`, `ok` and `result` or `error`. Events have `event` and `payload`; `indexProgress` also includes the indexing request ID. DTO fields use camelCase. Commands are `status`, `browse`, `search`, `index`, `preview`, `originalUrl`, `reveal`, `setModelDirectory`, `updateSettings`, `removeSource`, `theme`, `installModels`, `cancel`, `shutdown`. Directory arguments accept absolute local paths or local `file:` URLs. `status` is also a one-shot CLI mode that does not load models.


`status.catalog.modelProfile` identifies the indexed 512D contract, `embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2`. It is distinct from the unchanged native 768D graph contract.

`status.settings` contains `topMatches` (1–100, default 6), `resultLayout` (`listPreview` or `list`), `watchChanges` (default false), `closeAfterOpening` (default true), and `imageBatchSize` (1–8, default 1). `status.activeImageBatchSize` reports the size captured at worker startup. `status.settings` also includes `groupTextByFile` (false), `textIdleSeconds` (900) and `mediaIdleSeconds` (300); 0 keeps that model class warm. `updateSettings` requires the original five fields, defaults omitted extension fields for compatibility, and returns full status. Idle changes apply live; a batch-size change is saved without changing the running worker. These preferences persist alongside the configured model directory. `browse` and `search` accept optional `sourceId` (null means all folders) and `mediaKind` (`image`, `text`, `audio`, `video`, or null for All media). Nonempty Text searches use BM25 + semantic RRF; other scopes keep cosine ordering. `search` accepts `maxResults` (1–1,000); the UI supplies its top-match preference. `removeSource` requires a numeric `sourceId` and returns `{removed: segmentCount}`. `reveal` requires `assetId` and uses the desktop file-manager interface to select the original; unsupported reveal operations fail visibly.

When enabled, recursive folder watching coalesces changes and scans after a brief debounce. Background scans emit `indexStarted` with `{directory}`, `indexProgress` with `id: null`, `indexFinished` with the index result, or `indexError` with `{directory, error}`. Watch setup and runtime failures are reported; the app does not silently disable watching.
Index progress includes `failed` and a `failure: {path, error}` on each skipped-file event. An index result includes `failed`, the first 100 `failures`, and `failuresOmitted`; `ok: true` means the scan completed or was cancelled, and callers must inspect these counts. Every skipped file is also logged to stderr. Reported error messages are bounded to 2,048 UTF-8 bytes plus an explicit truncation notice; full errors remain in stderr. The UI retains the bounded report across browsing/search until the next scan.

Requests are capped at 64 KiB, queued requests at 16, queries at 4,096 UTF-8 bytes, pages at 100 and scans at 250,000 files. Search pages also obey a 1.5 MB serialized budget below the transport ceiling; a shortened page reports `responseLimited: true`, retains total, and advances by its actual returned item count. Unknown fields and invalid values fail visibly. EOF drains accepted requests; explicit shutdown and SIGINT/SIGTERM finish in-flight inference, preserve completed files, discard incomplete multi-segment replacements, reject queued work and checkpoint the catalog. Do not kill the process to stop a scan.

## Development and validation

From the repository root:

```bash
cargo fmt --all --check
cargo check --workspace --locked
cargo test --workspace --locked
```

These commands cover both workspace members: the native search core and plugin backend.

With pinned assets and newly generated indexed-512 reference fixtures available, validate the resident runtime:

```bash
cargo run -p fractal-search-core --example validate_embeddinggemma2 -- \
  /absolute/path/to/models /absolute/path/to/indexed-fixtures/fixtures.json
```

The example uses real image decoding/tokenization and requires cosine similarity of at least 0.9999 against the indexed 512D reference. Add `--native-output` before the model directory to validate existing native 768D fixtures explicitly. Reference fixtures can be generated with the optional Python development tool described in the [model guide](../docs/MODELS.md). Python is not required to run the plugin.

The [embedding validation](../docs/MODELS.md#providers-and-validation) describes checks against the pinned export. Historical prior-model fixture results do not validate this replacement. Report numerical parity and retrieval measurements only after running the current checks; they are not a labeled-library relevance benchmark.

Run the opt-in backend integration/lifecycle smoke check with a built binary and pinned model cache:

```bash
python3 quickshell-app/tests/smoke_backend.py \
  target/debug/fractal-search /absolute/path/to/models \
  --fixtures /absolute/path/to/native-fixtures
```

Omit `--fixtures` to use generated PNG inputs and the small synthetic corrupt-JPEG fixture only. Its hexadecimal text is decoded into the temporary test directory; no image fixture is tracked in Git. The script tests indexing, search pagination, preview dimensions, incremental rescans, missing-thumbnail repair, corrupt-file continuation and repair (including a JPEG extraneous-byte warning), saved preferences, folder-filtered ranking, source removal, native file-change watching, strict requests, cancellation, catalog locking, shutdown/reopen and EOF drain. All runtime catalogs and images are created under `/tmp` and removed afterward.

#### Historical reconstructed-source validation (2026-10-07)

This section predates the 512D indexed-profile change. Its native 768D parity and retrieval results do not establish current indexed 512D parity or relevance; rerun the current commands above.

This implementation was reconstructed after a temporary workspace was removed. The checks below were rerun against the reconstructed source; earlier results were not reused as evidence.

- Default and CUDA-feature all-target checks passed. Each workspace test suite passed **107 tests**, with one environment-blocked Unix-socket failure and three tests ignored by default. All 84 non-ignored core tests passed, including ten hybrid regressions. The complete suite still exits nonzero because the supported permission retry also returned `Operation not permitted` for the private Unix socket
- The opt-in real FFmpeg catalog segmentation and pinned-tokenizer tests passed separately
- Native preprocessing passed all 39 fresh fixtures, including image batch eight. Native CPU inference passed 38 cases / 39 embeddings across all modalities: minimum reference cosine `0.999999999992`, maximum component error `5.96e-7`. Inference image batches were at most two
- Real multimodal/hybrid, backend lifecycle and read-only MCP checks passed, including failed-file preservation, cold/warm loading, cancellation, watching, shutdown/reopen, EOF/SIGTERM and cached access after originals were removed
- End-to-end retrieval comparison passed 153 scores from 17 synthetic images and nine queries: maximum score error `2.45e-7`, nine of nine top-1 matches, 100% top-five overlap and no inversions beyond tolerance
- Formatting, whitespace, Python/Bash syntax, eight downloader tests and the actual Controller JavaScript regression harness passed

The cloud workspace used Rust 1.99 and CPU ONNX Runtime 1.22.0. Validation-only CLI settings disabled dependency optimization, debug information and incremental artifacts to reduce rebuild cost; production manifests, dependency versions, features and assertions were unchanged. Unoptimized SHA-256 caused an initial cold-start harness timeout, so normal level-two optimization was restored only for `sha2:0.10.9`; identical digests and eight model/hash tests were verified. Final backend/MCP/retrieval checks used those rebuilt binaries. The full suites, native parity and multimodal checks preceded this build-profile adjustment and cover unchanged source.

A separate initial five-image indexing request exceeded the generic 60-second metadata deadline while progressing normally. Only that long-running request now uses the retrieval comparator's bounded 300-second async wait. Every assertion and all other metadata/cancellation deadlines remain unchanged; the full lifecycle rerun passed.

Native Quickshell/Omarchy rendering/input, actual CUDA hardware/provider placement, GPU-ready packaging, full batch-eight inference, release performance and large-corpus relevance/storage/latency remain unverified. CUDA-feature checks used CPU ORT; Node/static checks are not render tests. Synthetic parity compares HF preprocessing plus the pinned ONNX export, not the original Google training checkpoint or a labeled relevance benchmark.

## Historical validation before the model replacement

The following dated results describe earlier source revisions and the previous model. They are retained as history only and do not validate the reconstructed EmbeddingGemma 2 multimodal/hybrid source. Current build, inference and native UI checks must be rerun.

Validation on 2026-09-30 passed 18 workspace unit tests, the backend smoke check including corrupt-file continuation/repair and SIGTERM during indexing, and 35 native model fixtures (43 embeddings; minimum cosine 0.99999999156). UI checks covered empty, loading, missing-model/error, populated, search, preview and partial-scan recovery states; compact layouts at 560×620 and 768×419; active theme directory replacement; and window close during inference and before service readiness. Corrupt JPEG/WebP inputs, the skipped-files report, search with a retained report, repair/retry and clean exit passed in isolated offscreen and native Wayland windows. The normal launcher also ran as a real tiled Hyprland/Wayland window with native folder-picker open/cancel and clean exit. These are functional checks with synthetic images, not a full-library performance benchmark.

The mockup UI revision was additionally checked with a synthetic NDJSON fixture: summon/results/settings, empty and error states, inline selection previews, preferences, narrow layouts, dismiss/toggle and clean exit passed offscreen. Native Wayland checks confirmed compositor tiling, native folder-picker open/cancel, Down selection, Ctrl+, settings, Escape back/dismiss, and Enter/Ctrl+Enter command dispatch. The fixture deliberately rejects open/reveal operations to verify visible errors; it does not validate desktop viewer/file-manager execution or native inference. QML lint and formatting checks passed.

Validation on 2026-10-01 passed 21 workspace tests, formatting/checks, and the extended backend smoke check with real ONNX inference and synthetic PNG/corrupt JPEG inputs. Native Wayland UI checks passed model-folder setup, corrupt-file report retention across search, top-match preferences, source filtering, automatic folder watching, persisted settings/catalog reopen, confirmed source removal, resident dismiss/summon/toggle/query, Nautilus reveal, and actual window close during indexing. Preview and thumbnail proportions, including automatic repair of the earlier square caches, passed regression and native screenshot checks. The `summon` IPC name avoids Quickshell's reserved CLI `show` subcommand. Restored models live in the persistent user cache, not `/tmp`. Native parity fixtures were not regenerated and no full-library performance benchmark or desktop viewer launch was performed in this revision.

The compact launcher revision also passed native Wayland checks for the actual 680 × 56 surface, interpolated expansion to 980 × 640, typing and result navigation, Escape collapse, interrupted animations, settings/narrow layouts, empty/error/missing-model states, resident dismiss/summon and Ctrl+Q. Native folder-picker open/close released and restored panel focus. A successful synthetic reveal acknowledgement verified the panel's focus/layer handoff without opening another app. Real ONNX indexing/search/preview with a temporary two-image catalog and one corrupt file verified visible progress/errors, the skipped-files dialog, preferences and clean close during search. All 21 workspace tests, formatting, checks and QML lint passed. No user library or desktop configuration was changed for these tests.

The hotkeys-footer revision passed native Wayland checks for the 680 × 95 compact surface, persistent footer in summon/results/settings and empty/error states, expansion/collapse, result navigation, narrow-screen wrapping, picker focus and shutdown through IPC. Nine isolated launcher cases covered cold launch, existing-instance reuse, unrelated Omarchy instances, duplicate instances, readiness/IPC/list/JSON failures and rejected runtime overrides. The launch/summon command also passed through Omarchy's Lua exec API; simulated global keypresses did not activate the binding, but the user confirmed Super+I opens the app from the physical keyboard. All 21 workspace tests, formatting/checks, QML lint and Bash syntax checks passed. The explicitly requested personal Super+I binding was backed up, registered and validated without replacing an existing shortcut. Native inference was not rerun for these UI/launcher changes.

Animation tuning uses the destination dimensions for result/settings layouts and footer wrapping, then clips their reveal through the animated surface. Native Wayland frame sampling with a synthetic service confirmed stable layout dimensions and breakpoints throughout 160 ms transitions, replacing the previous mid-animation row/preview/footer changes. Typing triggered search after approximately 180 ms; interrupted transitions, keyboard navigation, previews, narrow layouts, empty/error states, native picker focus and clean IPC shutdown passed. The real backend also passed browsing, preview and ONNX semantic search with an existing temporary two-image catalog, followed by a clean checkpoint/shutdown. All 21 workspace tests, formatting/checks and QML lint passed. These checks do not measure animation performance during full-library indexing.

The resize-artifact revision was checked with compositor screenshots against an isolated backdrop. The previous pale regions were reproduced before the change; 44 captures across seven transitions verified the visual mask, compact sizing and header alignment after the change. Scrollbar gutters passed at 520 × 740 and 420 × 360, including screenshots with the thumb visible. Native checks passed Enter/Down/Ctrl+,/Escape, interrupted transitions, hide during resize and resummon, empty/error/missing-model states, picker focus and actual window close. Real ONNX search/preview on the temporary two-image catalog and clean checkpoint/exit passed. All 21 workspace tests, formatting/checks and QML lint passed. No desktop configuration or user library was changed.

The report/resize timing revision passed native mouse checks for Skipped files, Details, dialog Close, Dismiss, and the retained report in Settings. Watched-scan fixtures confirmed that acknowledgement does not erase diagnostics or suppress new errors. Keyboard search/navigation, empty/setup states, narrow settings, native picker focus, interrupted resizing, hide/resummon and actual close passed. Outside clicks dismissed the panel in summon/results/settings and with a report open; a subsequent click reached the isolated backdrop, confirming desktop input release. Reports reopened after multiple native hide/resummon cycles, and hiding an open report or native picker closed it before the next summon. Frame sampling checks that no visible painted panel exceeds the native surface and that animation starts only after its buffer is ready. 64 compositor captures across eight transitions/remappings reproduced the second native movement in the resizing-surface design and confirmed centered expansion/collapse with the fixed rendering surface. Real ONNX search/preview with the plain production QML and a temporary two-image catalog passed repeated hide/resummon and clean shutdown. Workspace formatting/checks, all 21 tests and QML lint passed; no user library or desktop configuration was changed. These isolated checks do not measure animation performance during full-library indexing.

For isolated native UI checks, copy `ui/` to a temporary directory, add a tooling-only `Commons` link to `$OMARCHY_PATH/shell/Commons`, run `fractal-search daemon --socket /tmp/fractal-search-ui-test/run/backend.sock --data-dir /tmp/fractal-search-ui-test/data --model-dir /absolute/path/to/models`, then launch the developer-only `shell.qml` harness with `FRACTAL_SEARCH_SOCKET` pointing at that socket, `FRACTAL_SEARCH_PICKER` pointing at the built `target/debug/fractal-folder-picker`, and `FRACTAL_SEARCH_UI_TEST=1`. The harness is excluded from installed bundles. Its `search` and `test` IPC handlers provide visibility, geometry, indexing, settings, reports and screenshot checks; only this isolated harness may call `Qt.quit()`. Always hide/close the test panel promptly to release desktop input. Never kill or restart Omarchy to close a plugin test.

Run `python3 quickshell-app/tests/smoke_plugin.py target/debug/fractal-search /absolute/path/to/models` for real-inference daemon tests. It verifies private/exclusive sockets, cache reuse, indexing across UI detach/reconnect, session IDs, report recovery, search/preview, strict frames, shutdown/reopen, SIGTERM drain, and offline download progress/cancellation. Tests create temporary catalogs and synthetic images only. Linux IPC tests require host access when a sandbox blocks Unix sockets.

## Historical plugin conversion validation (2026-10-01)

Workspace formatting/checks and 25 tests passed. Real ONNX daemon smoke checks passed cached-model verification without redownload, indexing across UI disconnect/reconnect with retained skipped-file reports, unique request IDs, search/preview, graceful shutdown/reopen, SIGTERM during indexing, and offline model-download progress/cancellation. Installer checks passed clean install/update, catalog preservation, damaged checksums, unmanaged paths, missing native dependencies and rollback after a failed enable.

An isolated native Wayland harness passed empty/populated/error/report/settings states, real indexing/search/preview, UI hot reload during indexing, compact footer and narrow settings, and graceful exit. The live `fractal.search` plugin was enabled in the existing Omarchy process; summon/hide/backend stop worked, the launcher shortcut command worked, and the top-bar configuration and shell PID remained unchanged. No user photo library was indexed by these tests. Actual network downloads were not repeated: the real pinned cache was reused, and download/cancel behavior used an offline curl fixture. Public packaging/signing and performance benchmarking are not part of these checks.

Native picker protocol checks use a private session bus and synthetic folders:

```bash
FRACTAL_PICKER_PRIVATE_BUS=1 dbus-run-session -- python3 quickshell-app/tests/smoke_picker.py target/debug/fractal-folder-picker
```

The test requires Python GObject bindings only as development tooling; the packaged picker is a Rust executable. It checks early responses, selection of local paths with spaces/Unicode, cancellation, SIGTERM request closure, malformed responses and missing/old portal interfaces.

The portal-picker revision passed all 27 workspace tests, the private-bus picker protocol checks, and native Wayland selection of both model and image folders using a temporary catalog. Cancel, hide/reload cancellation, resummon and graceful shutdown passed. Installer checks passed update/catalog preservation, checksum and dependency failures, and rollback. Both live Browse buttons opened and cancelled successfully inside the existing Omarchy shell; its PID and status-bar configuration remained unchanged. The settings layout selector retained its longest-label width across both selections; a compositor capture confirmed the open menu shows both labels completely. Settings navigation uses the header and footer Back controls, without a third button in the scrollable content.

On the installed GTK portal 1.15.3, its Cancel button returns normal cancellation, while Escape/window close returns portal response 2 (an error). Fractal reports that response visibly; it does not reinterpret portal errors as cancellation. This is separate from the original in-process GTK/GVfs abort: the portal implementation now runs outside the shared Omarchy shell.

The close/stop revision passed all 27 workspace tests, QML lint, offline close-routing checks, and installer update/rollback/uninstall checks. Isolated native UI checks covered confirmation, Cancel as the default action, repeated Ctrl+Q, Escape, narrow layouts, focused picker cancellation and graceful stop. Real ONNX indexing/search on a temporary synthetic catalog continued after Cancel/hide and drained after confirmed stop. The installed live keyboard test did not complete: simulated input opened Omarchy's System menu, and the test's cleanup helper accidentally clicked Logout after hiding Fractal. The journal recorded `uwsm stop` and normal session shutdown. That test was retired; its result is not a passing live shortcut validation.

Automated keyboard/mouse tests must run in an isolated compositor. Do not inject global keys or coordinate clicks into the user's shared Omarchy session, or send cleanup input after hiding a test surface. Use scoped plugin IPC and read-only inspection for shared-session checks; leave physical shortcut verification to the user.

The hide-after-open path hides the overlay directly; it lowers the panel only when the preference keeps it visible. Summoning restores the overlay layer before mapping its native window, avoiding a hide/remap race with the external-viewer layer change.

Run the viewer/resummon regression in a private nested Hyprland session (requires `imv`, `grim`, Quickshell and host compositor/GPU access):

```bash
python3 quickshell-app/tests/smoke_resummon.py target/debug/fractal-search /absolute/path/to/models
```

It uses a private session bus, synthetic images and a temporary catalog, tests both checkbox states with 16 actual viewer open/close cycles, and checks empty/loading/search/preview states, missing-original errors and clean shutdown. Artifacts and a compositor capture remain in the printed `/tmp/fsr-*` directory. On 2026-10-02 the corrected panel regained visibility and focus within 0.151 seconds in these checks; the reported 10–15 second delay was not reproduced in the isolated baseline, so physical Super+I verification remains necessary.

## CUDA, diagnostics and MCP validation (2026-10-02)

Both CPU and CUDA workspace checks and all 45 tests passed, along with formatting, QML lint and Bash syntax checks. Unix-socket tests and systemd installer verification required host permissions after the development sandbox returned EPERM. Real GPU tests verified indexing/search, CPU/CUDA embedding parity, CUDA matrix/attention execution and explicit failure for an invalid GPU. The read-only MCP smoke passed against synthetic photos with the originals removed, including lazy text inference, privacy limits, strict requests and EOF draining. See [CUDA.md](CUDA.md) for measured GPU timings and [MCP.md](MCP.md) for the client command and protocol limits.

Diagnostics passed five Rust checks and six offscreen UI cases; run `python3 quickshell-app/tests/smoke_diagnostics_ui.py` to repeat its success, failure, malformed/oversized output and timeout cases. Native Settings rendering and an actual dependency report passed in a private compositor. The updated viewer regression repeated all 16 cycles with a maximum resummon time of 0.158 seconds. The CUDA bundle passed installer update/provider persistence, matching diagnostic/service loader paths, stable MCP launcher, rollback, checksum and missing-dependency checks. The installed service subsequently loaded its text model on CUDA device 0. These checks used temporary synthetic catalogs; no user photo library was indexed for validation.

## QMD-inspired indexing and memory controls

Preferences can group explicit Text results by file with expandable additional passages. Performance offers independent text/media Keep warm or idle-release timers and live session counters/load timings; these do not measure RSS/VRAM. Folders offers a Cancel-default **Rebuild text passages** confirmation; it requests explicit structural reindex plus source-content verification, preserving complete old rows until each file succeeds. Ordinary Rescan remains incremental. See [QMD improvements](../docs/QMD-SEARCH-IMPROVEMENTS.md) and [labeled evaluation](../docs/QMD-EVALUATION.md). Run `python3 quickshell-app/tests/smoke_qmd_ui.py` for deterministic UI/controller checks; native rendering and interactions still require the isolated desktop workflow.
