# Preparing prebuilt releases

Fractal Search is a native Linux/Omarchy application. Prepare separate CPU and
CUDA archives; end users run the included installer and need no Rust/Python.
Publishing a marketplace listing does not build these binaries. Request the
marketplace's manual-setup designation and link the actual release downloads.

## Build and package

Run from a reviewed source checkout, with the pinned Cargo.lock and required
native build dependencies. `cargo build --release` uses the existing optimized
release profile; development incremental caching is unchanged. Builds share
Cargo's target directory, so always package CPU before building CUDA.

```bash
python3 scripts/fetch-release-sources.py
cargo build --release --locked -j 2 -p fractal-search -p fractal-folder-picker -p fractal-search-mcp --bins
scripts/package-fractal-search.sh --no-build --output /absolute/new/cpu-bundle
cargo build --release --locked -j 2 -p fractal-search -p fractal-folder-picker -p fractal-search-mcp --bins --features fractal-search/cuda,fractal-search-mcp/cuda
scripts/package-fractal-search.sh --no-build --cuda --output /absolute/new/cuda-bundle
```

Packaging requires Python 3 for attribution generation, cargo/rustc, jq, and
standard system tools. Install the active Rust toolchain's `rust-docs` component
(`rustup component add rust-docs`) to supply its runtime copyright/license
inventory; missing attribution fails packaging. Those are maintainer dependencies, not end-user Python
requirements. Provider checks reject the wrong binaries. `bundle.json` records
architecture, release/debug profile, source commit and dirty state, compiler,
and build glibc. A dirty tree is a candidate, not a claim that HEAD reproduces
all included documentation. Supply the matching source snapshot with archives;
commit/review the final source before public version tagging.

## Attribution and redistribution inputs

`scripts/generate-third-party-notices.py` uses locked/offline Cargo metadata for
the host Linux target and selected CPU/CUDA features. It conservatively includes
build/dev dependencies, preserving full declared license expressions and all
license/notice/copyright files (including nested native-code notices). It fails
on missing declarations/texts or altered provenance checksums. This inventory
is a superset, not a linker-level software bill of materials.

Version-specific texts omitted from published crates are stored in
`quickshell-app/packaging/licenses/`, with upstream URLs, revisions and hashes in
`provenance.json`; `overrides.json` maps exact package versions. Dependency
updates require reviewing these inputs again. ONNX Runtime's pinned 1.24.2
license and ThirdPartyNotices accompany its static runtime and CUDA providers.
Rust's installed runtime copyright inventory and license texts are included.
The runtime archive's verified download hashes remain pinned by ort-sys rc.12.

The MPL-2.0 option-ext crate is unmodified; its exact complete source is shipped
under `third-party/sources/`, with its license. ONNX Runtime's pinned Eigen
source archive is included there too, with its upstream build metadata and
both Eigen patches under `third-party/onnxruntime-1.24.2/`. The maintainer fetch
step downloads and verifies the pinned archive into `target/release-sources`;
packaging remains offline and fails if that source is missing or changed.
The MIT-declared fax/fax_derive
crates have no standalone license/copyright file at their published source
revision. We preserve their declared authors/metadata, include canonical SPDX
MIT terms marked as a template, and supply their exact source. We do not invent
an upstream copyright notice. LanceDB's manifest-only patch and attribution are
included. License texts are preserved, not relicensed under our MIT license.

System libraries (including libheif, libjpeg-turbo, libc/libstdc++, Qt and
FFmpeg), NVIDIA driver/CUDA/cuDNN and model assets are not bundled. Their own
terms apply to their separate installation/download. Do not add them to release
archives without reviewing that change. The model manifest is included, not the
model weights. This process records attribution and source availability; it is
not a legal certification or an upstream endorsement.

## Validation and distribution

- Run `sha256sum --check SHA256SUMS` in each bundle and verify both provider reports.
- Run `smoke_installer.py` with each bundle in its temporary environment.
- Run real CPU/CUDA `smoke_multimodal.py` against bundle binaries with an existing
  verified model cache; pass `--provider cpu` or `--provider cuda` explicitly and
  expose the bundle's provider libraries and separately installed NVIDIA runtime
  through `LD_LIBRARY_PATH` for CUDA. Use temporary synthetic catalogs, never
  the user's data. Run `smoke_cuda.py` for CUDA daemon forwarding and explicit
  invalid-device failure checks.
- Confirm the CPU bundle has no CUDA provider libraries or NVIDIA ELF dependency.
- Confirm AGENTS.md, AGENT-SETUP.md, full third-party inventory and covered source
  survive installation into the versioned runtime.
- Test on the declared Omarchy/system-library baseline. Host-built binaries are
  not portable promises for older glibc, other distributions or architectures.
- Archive the bundle directory, supply an outer SHA-256 checksum file, and include
  source/provenance and release notes. Exclude model assets, private data, logs,
  target caches and secrets from source snapshots.

The included checksum manifests detect corruption, not publisher identity.
No signing key is configured by this workflow. Publish only after reviewing the
artifacts and obtaining authorization; do not silently create a public release,
upload private source, tag, or submit a marketplace issue as part of packaging.
Updates use the new bundle installer, preserving data/provider/settings and
pruning only obsolete managed runtimes. Uninstall uses the installed uninstaller.
