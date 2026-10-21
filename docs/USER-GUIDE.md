# Fractal Search user guide

[Project overview](../README.md) · [All documentation](README.md) · [Agent-assisted setup](AGENT-SETUP.md)

Installation, everyday use, supported files and troubleshooting.

## Install and update

**Manual setup is required.** End users should use a compatible prebuilt bundle's `install.sh`; no Rust compiler or Python environment is required. Omarchy's generic `omarchy plugin add` command only clones the repository; it does not build the native binaries, create the user service or generate runtime paths. Do not enable a raw clone as a working installation.

Use a compatible [CPU or CUDA release bundle](https://github.com/nigh8w0lf/fractal-search/releases/tag/v0.1.0) and its checksums. For source builds, see the [development guide](DEVELOPMENT.md). Check the release's architecture/system-library requirements and archive checksum, extract it, then run from the extracted directory:

```bash
sha256sum --check SHA256SUMS
./install.sh
omarchy-shell shell toggle fractal.search
```

Runtime dependencies include Omarchy Shell/Quickshell, curl, jq, libheif, libjpeg-turbo, gcc-libs, GLib’s `gio`, FFmpeg/ffprobe, and a desktop portal providing FileChooser. Inline playback additionally requires Qt Multimedia and its FFmpeg backend. Show in folder needs `gdbus` and a file manager implementing FileManager1.ShowItems. System dependencies are installed separately; the installer reports missing requirements.

For explicitly selected NVIDIA acceleration, use the CUDA archive and `./install.sh --provider cuda --device-id 0`; see its `CUDA.md` for the separately installed driver/runtime libraries. Updates preserve the current provider when flags are omitted. The agent guide covers setup, verification and troubleshooting. Release preparation does not itself publish download links.

Open an existing installation without building or touching the checkout:

```bash
omarchy-shell shell summon fractal.search '{}'
# Equivalent repository convenience command:
./start-quickshell-search.sh --no-build --summon
```

Use the installed summon command for a desktop shortcut. Installation preserves existing shortcuts; optional `--bind-close` explicitly installs managed Super+W close routing. Do not restart the desktop shell to update this plugin.

After a successful install, obsolete runtime builds are removed automatically. The current and immediately previous builds are retained, including across reinstalls; builds still running or containing the configured CUDA library directory are retained until a later update. Failed installs do not prune builds. Cleanup is limited to recognized runtime bundle directories and preserves catalogs, models, shared CUDA libraries and unrelated files. Cleanup errors are reported without rolling back the successful installation.


First-run **Set up model** downloads and verifies about 1.50 GB of pinned assets. **Choose folder** under Settings → Performance can reuse an existing verified model cache. Then add your folders under Settings → Folders.

In **Settings → Preferences → Connect an AI agent**, copy the installed server path or MCP JSON configuration, or open the setup guide. Paste the configuration into your MCP client; the client launches its own read-only server. This does not register clients automatically.

## Search and settings

- **Search:** a compact description field with a Settings gear icon and keyboard hints. Typing searches after a short pause; Enter submits immediately. There is no Browse all photos button. The existing empty-query Enter action still lists indexed segments.
- **Results:** a thumbnail list and selected-result preview, friendly folder names, All folders and All file types / Images / Text / Audio / Video filters, and explicit **Open file** / **Show in folder** buttons. Preferences offers list-and-preview or list-only layout; narrow windows use the list with actions beneath it. A photo grid is not implemented.
- **Settings → Folders:** add folders, see file and segment counts, remove a folder from the index, rescan folders, keep folders up to date and review indexing issues. Removing a folder preserves originals and asks for confirmation.
- **Settings → Preferences:** prepare search at login, hide after opening a file, maximum results (default 6), result layout and MCP client setup.
- **Settings → Performance:** model setup, a compact Accelerator section, image batch size, shortcut setup and Stop service. Accelerator checks GPU requirements automatically when shown, displays actual CPU/GPU activity, and greys out an unavailable GPU. Details expands the report; Copy diagnostics prepares it to share with your agent. Availability does not mean active GPU inference and never changes the configured provider.

Up/Down selects results, Enter opens the selection, Ctrl+Enter shows it in the file manager, and Ctrl+, opens Settings. Escape goes back or hides the panel. Ctrl+Q asks to stop the service; Cancel is focused by default, using the same confirmation component as folder removal. Left/Right switches a focused settings tab. All panels follow Omarchy’s theme and font tokens, including popup border colors, gradients and widths. Shortcut hints are clickable, with consistent spacing and visible hover/keyboard focus states.

### Background service

Hiding or reloading the UI keeps the backend and indexing alive. Stopping the service drains active work and saves the catalog; summoning starts it again. **Prepare search at login** controls the actual systemd user-unit enabled state and preloads only text inference. Vision and audio encoders load lazily for the media being indexed. Idle ONNX sessions are released safely after configurable timeouts (default 15 minutes for text, 5 minutes for media); Keep warm is available.

### Image batch size

**Image batch size** accepts 1–8 (default 1) for CPU and GPU indexing. Larger batches use more memory. Saving a change prompts **Restart now** or **Later**; the running service keeps its current size until restarted. Restart now safely drains indexing and reconnects the panel. Consecutive images share a batch; skipped/unchanged files and the final partial batch can produce smaller batches. Inference errors stop the scan and drain preparation workers; they never trigger retries at a smaller size or on another provider. Non-finite model outputs are fatal and must be diagnosed rather than indexed or silently skipped.

For indexing internals and resource limits, see the [worker contract](INDEXING-WORKERS.md).

## Supported media and result segments

All media and queries share the pinned normalized 512-dimensional EmbeddingGemma 2 retrieval space. The pinned FP16 model emits 768 dimensions; the app keeps the leading 512 values and L2-normalizes them before indexing or search. Results represent searchable segments; one file can produce several results. Optional Text file grouping keeps the strongest document result and expandable additional passages. Counts distinguish `fileCount` (distinct source files) from `assetCount` (searchable segments).

- Images: JPEG, PNG, WebP, GIF, BMP, TIFF, HEIC and HEIF retain one embedding per still image and the 280-soft-token image budget
- Text: UTF-8 `.txt`, `.md` and `.markdown` files, at most 2 MiB, split into overlapping passages of at most 4,096 UTF-8 bytes with at most 256-byte overlap at Unicode boundaries, structural Markdown/paragraph boundaries and real formatted-input token checks, up to 1,024 passages/file. Explicit text reindexing upgrades unchanged legacy passages. This is plain-text passage indexing; Text mode adds BM25 + semantic hybrid retrieval; PDF, Office and OCR are not implemented
- Audio: decoded with FFmpeg to mono 16 kHz samples and embedded in segments up to 30 seconds. These are native audio embeddings; no transcription is generated
- Video: sampled at 1 frame per second into visual segments up to 30 seconds / 30 frames, using 140 soft tokens per frame. An available soundtrack is additionally indexed as separate audio segments with its own vectors and timestamps. Visual and audio matches can therefore refer to the same video file

Audio/video files are limited to 2 GiB and two hours. Video sources may contain up to 16,777,216 pixels with sides up to 8,192 pixels. Sampled frames are downscaled, preserving aspect ratio to pixel rounding and never upscaling, to at most 2,073,600 pixels and a 4,096-pixel side before model preprocessing. Originals remain unchanged. Ambiguous `.ts` files are checked for transport-stream headers and text content: TypeScript source is excluded, while binary inputs still undergo normal decoder validation. Audio is resampled to mono 16 kHz and trimmed to the segment’s exact sample budget (at most 480,000 samples), with normalized output timestamps. The system `ffmpeg` and `ffprobe` commands are explicit audio/video dependencies; missing tools, unsupported codecs and decode failures are reported. FFmpeg is not bundled, and dependency diagnostics do not prove codec support or successful inference. Decoder commands are confined to local files and explicit supported demuxers, with bounded output and timeouts.

### Indexing issues

A video soundtrack decode failure is a visible partial success: all visual segments and every valid audio segment commit together, including valid audio after a failed interval. Only failed audio intervals are omitted; stale rows for those intervals are removed. **Indexing issues** reports the skipped count and intervals (with bounded display text and full decoder errors in the backend log). A generation-bound retry marker ensures the next rescan retries incomplete audio, including after service restart; unchanged visual and valid audio vectors are reused. Segments are decoded again to verify them, so this is embedding reuse rather than decode caching. Cancellation, source changes, visual decode failures, native input preparation, model/inference, database and cache-write failures do not use this exception. Standalone audio decode failures still skip the file and preserve its previous complete rows. A partial video counts as both updated and a file with an issue; progress advances once. No catalog reset is needed.

### Previews and opening files

Rows show media kind, passage number or time range, and cached text excerpts (at most 2,048 UTF-8 bytes). Images have bounded previews; video shows a cached poster until you press Play. Audio and video play inline in the preview pane, starting at the selected match’s timestamp and continuing through the original file. Controls provide play/pause, seeking, elapsed/total time, mute and volume. Playback stops and releases the file when selection changes, the pane is hidden (including narrow/list-only layouts), Settings opens, or the panel closes. Loading/codec errors stay visible; playback never starts automatically. Qt Multimedia and its FFmpeg backend are required (`qt6-multimedia` and `qt6-multimedia-ffmpeg` on Omarchy). Text/audio do not request image previews. **Open file** opens the whole original with its default desktop app; it does not request seeking to the matched segment. **Show in folder** reveals the original. Files stay in place.

Opening uses GLib's `gio open`, which resolves the local file's MIME association
and honors terminal-based desktop entries. Fractal Search does not override
desktop defaults. Missing opener/application errors and launch timeouts stay
visible; the panel hides or releases focus only after GIO accepts the launch.
An application can still fail after accepting the request.

## Removal and permissions

For complete application removal, run the installed runtime's uninstaller **before** using Omarchy's generic plugin removal:

```bash
SEARCH_PLUGIN_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins/fractal.search"
SEARCH_INSTALLED_RUNTIME=$(cat "$SEARCH_PLUGIN_DIR/.fractal-search-install")
"$SEARCH_INSTALLED_RUNTIME/uninstall.sh"
```

This stops and disables the user service, removes the panel and managed service unit, and removes the optional managed close binding. Catalogs, models and runtime bundles remain on disk. `omarchy plugin disable fractal.search` hides the panel only; indexing and the service continue. `omarchy plugin remove` also removes the panel's installation marker, so it is not the application's uninstaller. Updates use the new bundle's installer rather than `omarchy plugin update`.

The app runs as your user and requires no root service. It reads selected folders, writes its catalog/caches under the XDG data/cache directories, and launches original files only through desktop actions. Installation writes the user plugin, a systemd user unit and versioned runtime files; optional shortcut changes require an explicit installer option or UI action. System dependencies must be installed separately. Model setup downloads pinned assets from Hugging Face over HTTPS and verifies their sizes and SHA-256 hashes; indexing and search run locally. The same-user desktop socket can read originals; see the [read-only MCP boundary](../quickshell-app/MCP.md).

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
