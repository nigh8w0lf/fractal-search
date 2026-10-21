# Private release readiness — 2026-10-22

Target: Fractal Search 0.1.0 for current **x86_64 Omarchy with Omarchy Shell**.
The repository remains private. No public release, tag or marketplace submission
is authorized by this preparation. Local archives are under `dist/releases/`;
release output is excluded from Git.

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

A known-token/private-key pattern scan of all 582 reachable Git blobs found no
matches and no personal library paths under Downloads/Documents/Pictures/Videos.
This limited scan is not proof that no secret exists. Current source excludes
media, weights, runtime data and generated binaries.

Three binary media paths remain in historical commits: two generated UI mockups
and a synthetic JPEG regression fixture (now stored as hexadecimal text). They
are not in the current tree or release source snapshot. Removing them from Git
history would require a separately approved history rewrite and coordinated
force-push; no history was rewritten during preparation.

## Deliberately remaining before publication

1. Decide whether to remove those historical media blobs before exposing history.
2. Authorize public visibility and release publication, then create the release
   tag and upload the reviewed CPU/CUDA/source archives, checksums and notes.
3. Replace the README's unavailable-download note with the actual release link.
4. Confirm the ownership/submission checklist and submit the prepared
   [listing draft](MARKETPLACE-SUBMISSION.md).
5. Resolve the marketplace's exact-commit review and obtain maintainer approval.

The [release workflow](RELEASING.md) describes packaging and distribution;
[official submission requirements](https://github.com/omacom/omarchy-plugin-marketplace/blob/main/SUBMISSION.md)
and [security policy](https://github.com/omacom/omarchy-plugin-marketplace/blob/main/SECURITY.md)
remain authoritative for marketplace review.
