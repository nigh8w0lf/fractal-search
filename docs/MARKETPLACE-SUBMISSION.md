# Marketplace submission draft

Not submitted. Keep the repository private until the owner authorizes publication.
The intended title is **[Plugin]: Fractal Search**. After publishing the reviewed
source and release assets, copy the six sections below into the official issue
form. The unchecked statements must be verified and confirmed by the owner before
submission; the current public-repository statement is deliberately unchecked.

### Repository URL

https://github.com/nigh8w0lf/fractal-search

### Category

Productivity

### Tags

ai, launcher, media

### Suggest a missing tag

_No response_

### Maintainer notes

Fractal Search provides local semantic search for images, text, audio and video
on Omarchy Shell. Text mode combines semantic and keyword retrieval. Audio/video
previews play from the matched timestamp. A separate read-only MCP server exposes
bounded cached excerpts/thumbnails without original paths or file bytes.

Please apply `manual-setup` before approval. The generic git-plugin installer
cannot install our required native binaries or user service. Users install the
CPU or CUDA prebuilt bundle through its documented `install.sh`; no compiler or
Python is needed. Release downloads and system requirements must be linked from
the root README before this draft is submitted.

Installation writes a user-owned plugin, versioned runtime and systemd user unit.
It preserves catalogs, model caches, provider settings and existing bindings.
Shortcut changes require an explicit option; login startup is opt-in. The bundled
uninstaller removes the service/plugin and managed binding while retaining data.
The existing Omarchy Shell is never killed or restarted.

The runtime uses standard system dependencies, with CUDA 12/cuDNN 9 installed
separately for explicit NVIDIA acceleration. First-run setup downloads about
1.50 GB of pinned model assets over HTTPS and verifies sizes/SHA-256. Indexing and
search run locally. CPU/CUDA bundles include full third-party notices and covered
source. Models and NVIDIA/system libraries are not redistributed in the archives.

### Submission checklist

- [ ] The repository is public and contains installation and removal instructions.
- [ ] I have documented the plugin license and any external dependencies.
- [ ] I confirm that I own or have permission to submit this plugin and its preview assets.
- [ ] The plugin does not overwrite user configuration without explicit consent.
- [ ] I understand that approval is for listing and is not a security review.
