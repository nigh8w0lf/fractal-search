# Fractal Search

Local image search for Omarchy Shell, with a Quickshell panel, native Rust FG-CLIP2 inference, SQLite/sqlite-vec storage and a read-only MCP server. Index folders and find photos using descriptions. CPU and explicit NVIDIA CUDA inference are supported.

## Build and install

Run from the repository root:

```bash
./start-quickshell-search.sh --dev
```

To build and select CUDA:

```bash
./start-quickshell-search.sh --dev --cuda --provider cuda --device-id 0
# Add --cuda-lib-dir /absolute/persistent/path/to/lib for private runtime libraries.
```

The launcher builds, installs and opens the plugin. Omit `--dev` for a release build. An existing installation opens directly with:

```bash
omarchy-shell shell summon fractal.search '{}'
```

Use that installed command for desktop shortcuts; opening the plugin does not require a source checkout. The installer does not change your summon shortcut. The optional `--bind-close` flag installs the managed Super+W close routing.

Choose **Set up model** on first open to download and verify the pinned assets, or select an existing model cache. Settings includes CUDA requirement diagnostics. Setup failures are visible; CUDA errors never retry model inference on CPU.

Development requires Rust, a C/C++ toolchain, pkg-config and native image libraries. Runtime requirements include Omarchy Shell/Quickshell, curl, jq, libheif, libjpeg-turbo, gcc-libs and a desktop portal with FileChooser support. CUDA additionally requires a compatible NVIDIA driver and runtime libraries. See the guides below for setup and validation.

## Project layout

| Path | Contents |
| --- | --- |
| `quickshell-app/ui/` | QML search panel and settings. |
| `quickshell-app/backend/` | Native service, folder chooser and stdio MCP executable. |
| `crates/fractal-search-core/` | Decoding, preprocessing, tokenization, ONNX inference and catalog. |
| `quickshell-app/tests/` | Backend, MCP, installer and isolated UI smoke checks. |
| `quickshell-app/packaging/` | Installer, uninstaller and runtime notes. |
| `scripts/` | Bundle packaging and optional model validation tools. |

The two Cargo workspace members are fully contained in this repository. Runtime catalogs, model caches and downloaded CUDA libraries live outside the checkout. The installed catalog is normally `~/.local/share/fractal-search`; models are under `~/.cache/fractal-search/models/fgclip2-33f3b9d4`. Source updates and reinstallations preserve these locations.

## Guides

- [Plugin usage, architecture and validation](quickshell-app/README.md)
- [CUDA setup, diagnostics and measured validation](quickshell-app/CUDA.md)
- [MCP client configuration and privacy boundary](quickshell-app/MCP.md)
- [Model assets, preprocessing contract and reference fixtures](docs/MODELS.md)
- [Contributor instructions](AGENTS.md)

This is a local development project. Model weights and generated bundles are not checked in; see the [bundle notices](quickshell-app/packaging/NOTICE) for attribution and redistribution notes.
