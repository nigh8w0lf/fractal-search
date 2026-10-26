# Marketplace submission

Submitted with the owner’s approval on 2026-10-22:
[Fractal Search — issue #12503](https://github.com/omacom/omarchy-plugin-marketplace/issues/12503).
Initial repository/manifest/Quattro validation passed. The automated security
baseline reported no findings; installer, service-management and package-manager
capabilities require maintainer review. The v0.1.1 snapshot was approved and listed with `manual-setup` on 2026-10-25.

The root `preview.png` shows the driving-license search from the “Ready to go”
demo video (7.7 seconds). `docs/previews/insurance.png` shows the insurance search
(13.7 seconds) in the README. Both are unaltered 1920×1080 frames from the rendered
video, combining concise story text with actual native app footage. The footage
uses owner-approved fictional sample documents, a temporary catalog and a private
compositor. Sample originals and video binaries are not included in Git. These
two screenshots replace the earlier mountain/berry images and remain the only
intentional media exceptions; the marketplace uses the single root preview.

## First-run update request — 2026-10-27

Submit the new source through the marketplace verification form, selecting
**Verify and publish a newer upstream commit**, with the exact pushed HEAD.
The root entry now opens setup, downloads a checksum-pinned v0.1.1 CPU/CUDA bundle
on user action, then hands off to the existing installer. Model setup remains in
the installed app. No release assets are replaced. See the current validation
scope in [release readiness](RELEASE-READINESS.md#first-run-update--2026-10-27).

Request fresh review of installer/service-management capabilities and guidance on
removing `manual-setup`. Current standard-installation policy accepts only a passing
baseline or an installer-only review, so our additional service-management
capability is a policy constraint even though the UI now handles installation.
Keep the override until maintainers explicitly approve a supported path.

The listing details below record the original manual-install submission.

## Listing details

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
Python is needed. Release downloads and system requirements are linked from the root README.

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

- [x] The repository is public and contains installation and removal instructions.
- [x] I have documented the plugin license and any external dependencies.
- [x] I confirm that I own or have permission to submit this plugin and its preview assets.
- [x] The plugin does not overwrite user configuration without explicit consent.
- [x] I understand that approval is for listing and is not a security review.
