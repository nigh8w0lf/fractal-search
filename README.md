# Fractal Search

Local semantic search for images, text, audio and video on Omarchy Shell.

Add your folders, search in natural language, and open a match in its default app. Your files stay where they are; indexing and search run on your machine.

![Find a driving license by meaning — frame from the Fractal Search demo](preview.png)

![Find an insurance scan — frame from the Fractal Search demo](docs/previews/insurance.png)

## Features

- **Search by meaning:** find images and media without knowing their filenames. Text search combines semantic matches with keyword search.
- **Preview your results:** view images, read text excerpts, and play audio/video from the matched timestamp.
- **Fit your desktop:** Omarchy themes, keyboard navigation, folder/type filters and a choice of list or split preview layouts.
- **Keep your library current:** optional folder watching, incremental rescans and visible indexing issues.
- **Choose your hardware:** CPU support, with optional NVIDIA CUDA acceleration.
- **Connect an agent:** copy client configuration from Settings → Preferences. A separate read-only MCP server provides catalog search, excerpts and cached thumbnails without exposing original paths or file bytes.

## Supported files

| Type | Support |
| --- | --- |
| Images | JPEG, PNG, WebP, GIF, BMP, TIFF, HEIC and HEIF |
| Text | UTF-8 `.txt`, `.md` and `.markdown`, up to 2 MiB per file |
| Audio | FFmpeg-supported audio, searched in segments |
| Video | Visual content and available soundtracks, searched in segments |

Audio/video files are limited to 2 GiB and two hours. PDF, Office documents, OCR and transcription are not supported. See the [user guide](docs/USER-GUIDE.md#supported-media-and-result-segments) for decoding requirements and detailed limits.

## Get started

On **up-to-date x86_64 Omarchy with Omarchy Shell**, add and enable the plugin:

```bash
omarchy plugin add https://github.com/nigh8w0lf/fractal-search.git --enable
omarchy-shell shell toggle fractal.search
```

The first-run screen offers **Install CPU** or **Install NVIDIA CUDA**. It downloads
and verifies the pinned release, installs the backend and user service, and reports
missing dependencies. Installation continues if you close the panel. CUDA requires
separately installed CUDA 12/cuDNN 9; enter their library folder if installed
privately. CPU needs no NVIDIA libraries.

After installation, open Fractal Search again and select **Set up model** (about
1.50 GB), then add folders under Settings → Folders. No compiler or Python is needed.
The initial git checkout is retained as a recovery copy under
`${XDG_STATE_HOME:-~/.local/state}/fractal-search/setup/`; the installed app uses the
release bundle. Update it with a newer release bundle's installer, not `omarchy plugin update`.

### Install a release directly

Download [v0.1.2](https://github.com/nigh8w0lf/fractal-search/releases/tag/v0.1.2):
[CPU](https://github.com/nigh8w0lf/fractal-search/releases/download/v0.1.2/fractal-search-0.1.2-linux-x86_64-cpu.tar.gz),
[CUDA](https://github.com/nigh8w0lf/fractal-search/releases/download/v0.1.2/fractal-search-0.1.2-linux-x86_64-cuda.tar.gz),
[checksums](https://github.com/nigh8w0lf/fractal-search/releases/download/v0.1.2/SHA256SUMS).

Or run this complete download, verification and installation sequence:

```bash
SEARCH_VARIANT=cpu # Use cuda for NVIDIA acceleration with CUDA 12/cuDNN 9 installed.
SEARCH_DOWNLOAD_DIR=$(mktemp -d)
cd "$SEARCH_DOWNLOAD_DIR"
SEARCH_RELEASE=https://github.com/nigh8w0lf/fractal-search/releases/download/v0.1.2
curl --fail --location --remote-name "$SEARCH_RELEASE/fractal-search-0.1.2-linux-x86_64-$SEARCH_VARIANT.tar.gz"
curl --fail --location --remote-name "$SEARCH_RELEASE/SHA256SUMS"
sha256sum --check --ignore-missing SHA256SUMS &&
tar -xzf "fractal-search-0.1.2-linux-x86_64-$SEARCH_VARIANT.tar.gz" &&
cd "fractal-search-0.1.2-linux-x86_64-$SEARCH_VARIANT" &&
./install.sh --provider "$SEARCH_VARIANT"
omarchy-shell shell toggle fractal.search
```

The v0.1.2 CPU/CUDA archives were built on GitHub-hosted runners and carry signed
build attestations. See [verification instructions](docs/AGENT-SETUP.md#obtain-and-check-the-bundle)
for checking their repository, workflow and source commit. Checksums alone do not
establish build provenance.

Prebuilt installations need no Rust compiler or Python. Check the release’s architecture and system-library requirements. For NVIDIA acceleration, use the CUDA bundle with `./install.sh --provider cuda --device-id 0` and follow the [CUDA setup guide](quickshell-app/CUDA.md).

Required system packages include FFmpeg, Qt Multimedia with its FFmpeg backend, libheif, libjpeg-turbo, GLib, curl, jq and a FileChooser portal. The installer reports missing dependencies. CUDA additionally needs separately installed **CUDA 12 and cuDNN 9** libraries; an NVIDIA driver alone is insufficient.

On first launch, select **Set up model** to download and verify about **1.50 GB** of model assets, or choose an existing verified cache. Add folders under **Settings → Folders**, let indexing finish, then type a description to search.

To build from source, follow the [development guide](docs/DEVELOPMENT.md#build-and-install-from-source). For agent-assisted installation or troubleshooting, use [AGENT-SETUP.md](docs/AGENT-SETUP.md); it is also included in release bundles.

## Updates and removal

Update by running the new bundle’s `install.sh`. To uninstall, run the installed runtime’s uninstaller before removing the plugin through Omarchy:

```bash
SEARCH_PLUGIN_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins/fractal.search"
SEARCH_INSTALLED_RUNTIME=$(cat "$SEARCH_PLUGIN_DIR/.fractal-search-install")
"$SEARCH_INSTALLED_RUNTIME/uninstall.sh"
```

If backend installation has not started, remove the initial clone with
`omarchy plugin remove fractal.search`. If setup is running, let it finish before
uninstalling so it cannot recreate the app during removal.

Uninstall preserves your catalog, models and runtime bundles. See the [user guide](docs/USER-GUIDE.md#removal-and-permissions) for details.

## Documentation

| I want to… | Guide |
| --- | --- |
| Use the app, update, uninstall or troubleshoot | [User guide](docs/USER-GUIDE.md) |
| Have an agent set it up | [Agent setup](docs/AGENT-SETUP.md) |
| Configure NVIDIA acceleration | [CUDA](quickshell-app/CUDA.md) |
| Connect an MCP client | [MCP setup and privacy](quickshell-app/MCP.md) |
| Build or modify the app | [Development](docs/DEVELOPMENT.md) · [Contributor instructions](AGENTS.md) |
| Prepare release bundles or a marketplace listing | [Release preparation](docs/RELEASING.md) |

[All documentation](docs/README.md) includes architecture, model contracts, indexing details and validation reports.

## Privacy and license

Model setup downloads pinned assets from Hugging Face over HTTPS and verifies their sizes and hashes. Indexing and search run locally. The desktop app reads the folders you select and opens originals through your desktop’s file associations. Agent clients connected through MCP can receive bounded cached excerpts and thumbnails; see the [MCP privacy boundary](quickshell-app/MCP.md) before connecting a client.

[MIT license](LICENSE) · Copyright 2026 nigh8w0lf (Ashwin Cheekati). Dependencies and models retain their own licenses; release bundles include third-party notices and covered source.
