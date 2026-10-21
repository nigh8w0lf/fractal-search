# Release readiness — 2026-10-22

Target: Fractal Search 0.1.1 for current **x86_64 Omarchy with Omarchy Shell**.
The owner authorized public repository visibility and the CPU/CUDA release on
2026-10-22. [Release v0.1.0](https://github.com/nigh8w0lf/fractal-search/releases/tag/v0.1.0)
provides both bundles, matching source, checksums and validation/platform reports.
Generated release output is excluded from Git. The owner subsequently authorized
[marketplace submission #12503](https://github.com/omacom/omarchy-plugin-marketplace/issues/12503).
Initial validation passed; maintainer review and listing approval remain pending.

## v0.1.1 update

The owner authorized publication of CPU and CUDA v0.1.1 bundles on 2026-10-22.
This update adds **Connect an AI agent** under Preferences, with copyable MCP
configuration, the actual installed launcher path and a setup-guide action.
It also includes the approved demo-frame previews and updated documentation.
There is no catalog, model, native inference or dependency change, and no reindex
is required. Cargo package versions remain unchanged for this UI/installer patch;
the plugin manifest and bundle metadata identify release 0.1.1.

The native clipboard/narrow-layout and full settings chrome checks passed for
the UI change. Fresh bundle checks and exact source provenance are recorded in
v0.1.1's attached VALIDATION.json. Historical evidence below describes v0.1.0
unless explicitly identified otherwise. Keep the published v0.1.0 assets intact.

## Compatibility

The validation host runs the packaged Omarchy 4.0.6, glibc 2.44 and Quickshell
0.3.2.r7. CPU runtime dependencies resolve through system packages; prebuilt users
need neither Rust nor Python. Omarchy's package list includes Quickshell, imv,
mpv and portal backends, which bring many dependencies transitively. Qt Multimedia
and its FFmpeg backend must also be present; do not assume every user has every
optional desktop package. The installer reports missing dependencies.

NVIDIA acceleration requires separately installed CUDA 12 (including cuRAND) and
cuDNN 9 in addition to the driver. This host uses a private CUDA 12.8.90/cuDNN
9.10.2.21 runtime directory. That directory is not bundled or a default Omarchy
component. CPU operation does not depend on it.

CPU multimodal tests passed in a Bubblewrap namespace with an empty home, no
network, no GPU devices, no developer toolchain/cache and no private libraries.
Only system `/usr`/`/etc`, read-only application files and the existing verified
model cache were exposed; catalogs/media were temporary. This checks accidental
developer-environment dependencies. It is **not** a fresh Omarchy installation or
a guarantee for older package sets, other architectures or every hardware model.

## Validation evidence

- Optimized CPU build passed with the locked offline dependency graph.
- CPU real image/text/audio/video indexing, search, rescans, failed replacement
  preservation and read-only MCP passed in the isolated environment above.
- CPU temporary installer suite passed install/update/uninstall, checksums,
  provider/catalog/login preservation, rollback and runtime pruning.
- CUDA real image/text/audio/video indexing, search, rescans and read-only MCP
  passed on the existing 8 GiB NVIDIA GPU with explicit CUDA selection.
- Native Qt Multimedia tests passed manual playback, timestamp seeking,
  pause/volume, stale replies, selection/hide, narrow controls and corrupt input
  in a private compositor.
- Native search/settings regression passed safe batch restart and 16 external
  viewer close/resummon cycles across both hide-after-opening settings.
- Clean source export passed Omarchy manifest validation. Validate an export,
  not a checkout containing build-tree symlinks under `target/`.
- Final bundle-specific results, checksums, source commit and platform details
  accompany the refreshed archives. Earlier release candidates must not be
  mistaken for the final snapshot.

No production catalog, personal media or desktop bindings were changed for these
checks. Sandbox-denied sockets were retried through the approved host-access
mechanism; production code was not weakened.

## Marketplace preflight

The registry at upstream commit `50c1c6bdc05cad6f6249516845f2233c1efc98a4`
contains no `fractal.search`, including its retired-ID list. Availability must be
checked again at submission time; it is not a reservation.

The same revision's static analysis, run through a local text-file adapter,
reported **no findings** and the expected `installer`, `service-management` and
`package-manager` capabilities requiring maintainer review. The adapter uses
upstream path classification and size limits but does not impersonate a public
GitHub snapshot. This is local preflight, not marketplace approval or a complete
security audit. The actual submission must request `manual-setup` before approval.

## Attribution and repository review

The pinned fax/fax_derive archives declare MIT but omit standalone license files.
We now include the actual upstream pdf-rs MIT license/copyright from revision
`8f26f4f63a54fb93529ffdedd9b5a584c3ed01af`, with URL/hash provenance, while
retaining the older exact source and original author/license metadata. The
attribution explicitly identifies the later origin of the license file.
Other dependency notices and covered source remain included in each bundle.

After local history cleanup, a known-token/private-key pattern scan of all 590 reachable Git blobs found no
matches and no personal library paths under Downloads/Documents/Pictures/Videos.
This limited scan is not proof that no secret exists. Current source excludes
media originals, weights, runtime data and generated binaries. The owner approved
the root `preview.png` marketplace screenshot and `docs/previews/insurance.png`
README screenshot using sample content; these are the only intentional binary-media
exceptions.

The two historical generated UI mockups and synthetic JPEG fixture were removed
from all local commit histories with git-filter-repo. The current source tree
was unchanged, and the post-rewrite reachable-object scan found no binary media.
A private recovery bundle and commit map are stored outside this repository.

GitHub main was cleaned with an explicit expected-tip force-with-lease check.
All five other remote branches were subsequently deleted at the owner's request;
only main remained when release publication was prepared. A fresh single-branch
GitHub clone confirmed that none of the three media paths occurs in main's
history. Existing clones must be refreshed to avoid reintroducing removed blobs.

## Release handoff and marketplace follow-up

The README links the versioned CPU/CUDA downloads and checksums. The release
contains the exact clean source snapshot identified by bundle.json, with the
same tested executables and complete third-party notices. Archive contents,
permissions, provenance and checksums are verified before upload. These archives
are unsigned; checksums provide integrity rather than independent publisher identity.

Remaining marketplace work:

1. Refresh the existing submission against the preview/documentation commit;
   preserve the published v0.1.0 release artifacts and source snapshot.
2. Obtain maintainer review of the expected installer/service/package-manager
   capabilities (the initial automated baseline reported no findings).
3. Have the maintainer apply `manual-setup` before exact-commit approval and listing.

The [release workflow](RELEASING.md) describes packaging and distribution;
[official submission requirements](https://github.com/omacom/omarchy-plugin-marketplace/blob/main/SUBMISSION.md)
and [security policy](https://github.com/omacom/omarchy-plugin-marketplace/blob/main/SECURITY.md)
remain authoritative for marketplace review.
