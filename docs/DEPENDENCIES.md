# Dependency and build footprint

Shared dependencies are declared once in the root `Cargo.toml` under
`[workspace.dependencies]`; members inherit them and select the features they
need. Keep inference, tokenizer and catalog pins aligned with the model and
schema contracts. Updating a transitive major version solely to eliminate a
Cargo-tree duplicate can change behavior and is not a safe cleanup.

## Package separation and ICU-only patch (2026-10-12)

The workspace now has four packages: core, daemon, native picker and read-only
MCP server. The picker owns zbus; the MCP server owns rmcp and its bounded stdio
transport. The daemon does not compile either SDK for targeted builds.

```bash
cargo build -p fractal-folder-picker --locked
cargo build -p fractal-search --locked
cargo build -p fractal-search-mcp --locked
# CUDA is independently forwarded by the daemon and MCP packages:
cargo build -p fractal-search -p fractal-folder-picker -p fractal-search-mcp --bins --locked --features fractal-search/cuda,fractal-search-mcp/cuda
```

Packaging selects all three packages explicitly, preserves their existing binary
names and output paths, and rejects mismatched CPU/CUDA capabilities. Incremental
caching remains enabled. The picker-only Linux tree has **111 unique packages**,
including build/development dependencies, and no inference, catalog, model
tokenizer or MCP SDK dependencies. Full-workspace feature unification can make
its shared-library feature set larger than a targeted picker build.

The exact published LanceDB 0.37.1 snapshot is stored under `vendor/lancedb` and
selected by `[patch.crates-io]`. Its only implementation/manifest change disables
Lance Index's explicit Jieba/Lindera feature requests; all Rust source is unchanged.
Provenance, original manifests, source checksums and the upstream license are
preserved; see [patch notes](../vendor/lancedb/PATCHES.md). The bundle and installed
runtime carry `LANCEDB-LICENSE`. ICU, schema/index metadata, vector ranking and
hybrid RRF remain unchanged; no catalog migration or reindexing is introduced.

The patch removes **29 third-party packages**, with no upgrades or replacements.
The resolved Linux graph is now **547 packages**, including four workspace
members: separation added two project packages to the previous 574 total,
then the patch removed 29. Package counts do not measure compile-time or binary
size savings. The picker target's isolation is the main development-build gain.

```bash
python3 quickshell-app/tests/smoke_dependency_layout.py
```

This guard checks SDK ownership, picker isolation, the resolved local patch,
absence of dictionary-tokenizer features and unchanged upstream source bytes.
Multilingual ICU/hybrid search and explicit errors for disabled dictionary
tokenizers are also covered by core behavioral tests.

### Separation and patch validation

Locked/offline CPU and CUDA workspace checks passed. Each workspace test run
passed **109 tests**, with zero failures and three ignored. Formatting, shell
syntax, dependency-layout/source-integrity and diff whitespace checks passed.
The picker-only check and the all-three-binary CUDA development build passed;
both inference executables reported compiled CPU/CUDA support.

The rebuilt picker passed the private-bus portal fixture. Real CPU inference in
the MCP smoke check passed concurrent read-only access, hidden originals,
bounded thumbnails, lazy model loading and transport/lifecycle checks. The fresh
CUDA bundle passed plugin validation and the temporary-directory installer smoke
check, including provider persistence, catalog preservation, checksum rejection,
rollback, obsolete-runtime pruning and uninstall. These checks used temporary
data and the existing verified model cache; the installed app was not updated.
CUDA-feature tests and capability reporting do not verify GPU inference.

The real multimodal CPU smoke check also passed image/text/audio/video segments,
video soundtracks, stable rescans, atomic failed replacements, all-media cosine
ordering, text-only hybrid search, explicit source/media scope and cached
read-only MCP access after originals were removed.

## Initial dependency review (2026-10-12)

The 2026-10-12 review used Cargo's locked, Linux-filtered metadata, inspected
source usage, and compared resolved package identities before and after the
manifest changes. The default CPU graph decreased from **582 to 574 packages**,
with no package upgrades or new package identities.

| Change | Effect |
| --- | --- |
| Use zbus's Tokio backend with its blocking API | Removes async-executor, async-io, async-process, async-signal, async-task, blocking, piper and polling from the Linux graph. The native portal helper still uses the blocking API. |
| Remove the core's unused direct `half` dependency | Removes an obsolete direct declaration; `half` remains required transitively by Arrow and TIFF. Inference remains FP32. |
| Disable rmcp's `transport-io` feature | The server already supplies its own bounded stdio transport; `server` retains the SDK's required transport traits. |
| Select TOML parsing/Serde features explicitly | Avoids requesting TOML serialization/display for the app's theme reader. Build dependencies can still request their own TOML features. |
| Centralize eight shared dependency declarations | Keeps common version and default-feature choices consistent across both workspace members. |

The catalog stack is the largest dependency group: LanceDB's dependency closure
contained **456** of the **574** Linux packages after the initial review (including
LanceDB itself; closures overlap). It owns Arrow storage, query execution,
vector retrieval and ICU-based hybrid text search. Cloud/remote default features
are already disabled. Its pinned release still requires substantial
DataFusion/Arrow and tokenizer infrastructure. Removing that stack would require
a catalog/search redesign and explicit data-format decisions, not just a
manifest edit.

Image codecs remain explicit and match the supported formats. The native HEIF
and TurboJPEG integrations remain in place. ONNX Runtime's `std` feature also
activates ndarray support. The pinned rc.12
binding uses ndarray 0.17 while Lance uses 0.16; this deliberate duplicate comes
from their supported APIs and must not be removed by forcing a transitive upgrade.
The binding enables explicit native runtime/download/TLS features, with CUDA
remaining opt-in. No decoder, model, provider or storage fallback is introduced.

## Inference-only development checks

The core already separates inference from its optional catalog. When validating
only preprocessing or models, disable default features instead of compiling the
catalog too:

```bash
cargo check -p fractal-search-core --locked --no-default-features --features runtime --examples
# CUDA inference-only validation enables runtime automatically:
cargo check -p fractal-search-core --locked --no-default-features --features cuda --examples
```

Use the same feature arguments with `cargo run --example validate_embeddinggemma2`
or `cargo run --example validate_cuda`, respectively, and their documented model/
output arguments. The reviewed CPU runtime-only Linux tree contains **177 unique
packages**, including development/build dependencies, and no Lance/DataFusion
packages. This narrower mode does not validate the app, catalog or MCP server;
full workspace checks must retain their catalog features.

## Development artifacts

The previous debug directory reached approximately 165 GiB of allocated space,
including repeated test/example binaries and incremental caches. Dependency
count alone does not explain that growth: full debug symbols and repeated
CPU/CUDA, test and example builds also contribute.

The development profile keeps level-1 debug information for workspace code,
disables debug information for third-party dependencies, and retains incremental
compilation for faster rebuilds. Existing optimization levels are preserved (app 1, dependencies 2,
native core 3). Tests inherit the development profile; release settings are
unchanged. Smaller debug information reduces artifact growth; incremental caches
still use disk space and can accumulate across builds. External-library debugging
requires rebuilding with a suitable debug-info profile override.

Cargo profiles do not impose a disk quota or prune obsolete artifacts. When no
build or developer binary is running, clear development outputs with:

```bash
cargo clean --profile dev
```

This does not remove the installed runtime, model cache or catalog. The next
build needs to recompile; `--no-build` packaging requires fresh binaries after
cleaning. Successful installation independently prunes old installed runtimes.

To repeat the dependency review on this Linux target:

```bash
cargo metadata --locked --offline --format-version 1 --filter-platform x86_64-unknown-linux-gnu
cargo tree --workspace --locked --offline --target x86_64-unknown-linux-gnu --duplicates
```

Resolved-package counts are specific to the target, selected features and
lockfile; they are not measurements of binary size, compile time or runtime
memory. Historical test results elsewhere do not replace validation of a new
manifest change.

## Initial review validation (2026-10-12)

Workspace formatting and locked/offline CPU and CUDA-feature checks passed.
Both workspace test runs passed 108 tests with zero failures and three ignored.
The runtime-only CPU/CUDA example checks above also passed. All three native
binaries built with CUDA support; the desktop and MCP capability commands both
reported compiled CPU/CUDA support without loading models.

The private-bus portal fixture passed early responses, local Unicode folder
selection, cancellation, SIGTERM request closure, invalid responses, portal
owner disconnection, and old/missing portal errors with the Tokio-backed picker.
These are protocol checks against a temporary fixture, not a live GTK dialog
or GPU inference validation. No installed runtime or user catalog was changed.
