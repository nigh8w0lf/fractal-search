# CUDA processing in Fractal Search

Fractal Search supports native CUDA inference in its Omarchy Shell plugin backend
and read-only MCP server.

The current indexed/query profile is normalized 512D MRL: native 768D FP32 outputs are truncated to the leading 512 and L2-normalized on both providers. This does not reduce graph inference work or VRAM. Historical timings below are not measurements of this dimension change; see the [current model contract](../docs/MODELS.md).

## Selection and behavior

CPU remains the default. A CUDA-enabled build supports both providers; select CUDA
explicitly with `--provider cuda` or `FRACTAL_SEARCH_PROVIDER=cuda`. Select a GPU
with `--device-id N` or `FRACTAL_SEARCH_DEVICE_ID=N`; CUDA defaults to device 0.
Device IDs are rejected for CPU selection. Unknown providers, unavailable CUDA
build support, missing libraries and invalid GPUs fail visibly; the runtime never
retries a failed CUDA session on CPU.

CUDA accelerates the native EmbeddingGemma 2 vision, audio and shared/text ONNX sessions. Decoding,
patch preparation, tokenization and LanceDB retrieval remain on CPU. ONNX Runtime
may place small shape/control operations on CPU; actual placement must be verified. The validation below verifies that matrix and attention kernels
execute on CUDA in all three graphs. TF32 is disabled to preserve the FP32 embedding
profile. CPU and CUDA embeddings within the current indexed profile are compatible;
changing providers alone does not require a reindex. Moving from the previous
768D indexed profile to 512D still requires a fresh catalog and explicit reindexing.

`fractal-search providers` returns compiled capabilities without opening a catalog,
loading a model or initializing a GPU. The normal status response reports
`provider` (`CPU`/`CUDA`) and `deviceId` (null for CPU). Changing the model directory
retains the selected provider/device. The daemon explicitly forwards both to its
resident worker.

Inference fetches only `sentence_embedding`, avoiding a copy of the unused
`last_hidden_state` token matrix to host memory. CUDA sessions disable CPU
worker-pool spinning so those threads sleep while waiting for work. CPU sessions
retain their existing threading behavior. The FP32/profile and image budgets
are unchanged; these optimizations do not require reindexing.

## Image batching

Settings → Performance → **Image batch size** accepts 1–8 (default 1). Save it and choose **Restart now** to apply it, or **Later** to keep the running size until the next service start. Both manual and watched scans use that startup value. Compatible consecutive images share inference; tails and changes in patch budget produce smaller batches. GPU out-of-memory and other inference errors remain visible without retrying at a smaller size or on CPU.

To verify every batch size against single-image embeddings on CPU and CUDA with tiny/elongated and landscape/portrait synthetic images under the pinned 280-soft-token image budget (2,520 padded patch slots), use an existing verified cache and the provider library path:

```bash
env FRACTAL_SEARCH_BATCH_TEST_MODELS=/absolute/path/to/models \
  LD_LIBRARY_PATH=/absolute/path/to/cuda-libraries:$PWD/target/debug \
  cargo test --workspace --locked --features cuda \
  image_batches_preserve_single_image_embeddings_on_cpu_and_cuda -- --ignored --nocapture
```

This opt-in check uses CUDA device 0 and requires host GPU access. It never opens a catalog or uses original photos.

## Build and install

```bash
./start-quickshell-search.sh --dev --provider cuda --device-id 0
```

This builds the CUDA feature and installs the selected provider. For a prebuilt
bundle:

```bash
./scripts/package-fractal-search.sh --cuda --output /tmp/fractal-search-cuda-bundle
/tmp/fractal-search-cuda-bundle/install.sh --provider cuda --device-id 0
```

The CUDA bundle includes ONNX Runtime's CUDA/shared provider libraries. It does
**not** redistribute NVIDIA's driver, CUDA or cuDNN libraries. This pinned ORT
build uses CUDA 12.8 and cuDNN 9; see the [official compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
The required runtime loader names include `libcuda.so.1`, `libcudart.so.12`,
`libcublas.so.12`, `libcublasLt.so.12`, `libcufft.so.11` and `libcudnn.so.9`.
The CUDA 12.8 cuFFT package additionally needs `libnvJitLink.so.12`.

If these libraries are in a custom directory, pass
`--cuda-lib-dir /absolute/persistent/path/to/lib` to the installer or launcher.
The installer preserves that path in the managed service, MCP wrapper and UI
diagnostic environment. Use a persistent directory for an installed application;
the `/tmp` location in the development recipe below disappears on reboot.

To inspect setup without model/catalog startup:

```bash
target/debug/fractal-search diagnostics --provider cuda --device-id 0
```

Settings → CUDA accelerator diagnostics → Check/Refresh runs the same offline
check, including when the model worker cannot start. It reports the compiled
provider, driver/GPU visibility, CUDA/cuDNN and ORT libraries, transitive loader
dependencies, the effective library path and setup advice. Installed libraries
outside the loader path are distinguished from libraries that were not found.
The `nvcc` compiler is optional. Passing this preflight does not claim successful
model inference; use the real smoke check below to establish that.

## Historical temporary development dependencies

The 2026-10-02 validation used the following pinned NVIDIA wheels from PyPI. These
provide native runtime libraries only; Python is not an inference dependency.
Download size was approximately 1,463 MiB and extracted size 2.2 GiB.

Pinned package records: [CUDA runtime 12.8.90](https://pypi.org/project/nvidia-cuda-runtime-cu12/12.8.90/),
[cuBLAS 12.8.4.1](https://pypi.org/project/nvidia-cublas-cu12/12.8.4.1/),
[cuFFT 11.3.3.83](https://pypi.org/project/nvidia-cufft-cu12/11.3.3.83/),
[cuDNN 9.10.2.21](https://pypi.org/project/nvidia-cudnn-cu12/9.10.2.21/),
[nvJitLink 12.8.93](https://pypi.org/project/nvidia-nvjitlink-cu12/12.8.93/).

```bash
uv pip install --target /tmp/fractal-search-cuda-deps --no-cache --no-deps \
  nvidia-cuda-runtime-cu12==12.8.90 \
  nvidia-cublas-cu12==12.8.4.1 \
  nvidia-cufft-cu12==11.3.3.83 \
  nvidia-cudnn-cu12==9.10.2.21 \
  nvidia-nvjitlink-cu12==12.8.93
```

Create a single library directory containing relative symlinks within that
temporary dependency tree:

```python
from pathlib import Path
root = Path('/tmp/fractal-search-cuda-deps')
flat = root / 'lib'
flat.mkdir()
for directory in sorted((root / 'nvidia').glob('*/lib')):
    for library in directory.glob('*.so*'):
        (flat / library.name).symlink_to('../' + str(library.relative_to(root)))
```

The NVIDIA driver must already be installed on the host. GPU access can be hidden
by a development sandbox even when `nvidia-smi` succeeds; a `cudaGetDeviceCount`
failure must be resolved through the appropriate host permissions.

## Validation

Run from the repository root with pinned model assets already available:

```bash
cargo build --workspace --bins --example validate_cuda --features cuda --locked
cargo test --workspace --features cuda --locked

env LD_LIBRARY_PATH=/tmp/fractal-search-cuda-deps/lib:"$PWD/target/debug/examples" \
  target/debug/examples/validate_cuda \
  "$HOME/.cache/fractal-search/models/embeddinggemma2-daa72c51" \
  /tmp/fractal-search-cuda-validation

env LD_LIBRARY_PATH=/tmp/fractal-search-cuda-deps/lib:"$PWD/target/debug" \
  python3 quickshell-app/tests/smoke_cuda.py target/debug/fractal-search \
  "$HOME/.cache/fractal-search/models/embeddinggemma2-daa72c51"
```

Choose a new benchmark output directory each time. The explicit `target/debug`
or `examples` loader path is required for development binaries: ORT's provider
libraries sit beside them but are loaded dynamically. The packaged launchers
configure this path automatically.
Build the binaries and example together to retain the workspace's dependency
feature set and reuse existing artifacts; building only the core example can
trigger a separate LanceDB/DataFusion rebuild.

`validate_cuda` creates three synthetic PNG images, compares CPU/CUDA normalized
vectors for single images, a mixed-shape batch, queries, documents, audio and video, and requires
cosine similarity of at least 0.9999. It records five warmed image/query calls,
writes `report.json`, and examines actual ORT profiles to reject CPU matrix or
attention execution in the CUDA sessions. Profiling is opt-in for this benchmark;
normal desktop inference does not write profiling files.

`smoke_cuda.py` uses temporary catalogs and sockets. It verifies capabilities
without catalog writes, invalid configuration rejection, explicit CLI selection
overriding the environment and reaching the worker, model-directory changes,
real indexing/search, invalid-device failure without CPU retry and clean shutdown.

### Development optimization validation, 2026-10-08

The current optimized development runtime passed the CPU and CUDA-feature
workspace suites: 108 tests in each, zero failures and three ignored by default.
The CUDA binaries and `validate_cuda` example built together; the build log is
`/tmp/fractal-search-quickwins-build-cuda-agent.log`. Both desktop and MCP
capabilities reported compiled CPU/CUDA support. Dependency diagnostics passed
all nine required libraries after the host system update, on driver 610.57.04
and device 0 (`/tmp/fractal-search-quickwins-providers.json`).

Real `validate_cuda` inference passed twelve CPU/CUDA comparisons across single
images, mixed-shape image batches, queries, documents, audio and video. Minimum
cosine was `0.9999999999971763`; maximum component error was
`2.942979335784912e-7`. The three CUDA profiles recorded:

| Graph | CUDA matrix/attention events | CPU matrix/attention events |
| --- | ---: | ---: |
| Vision encoder | 1,584 | 0 |
| Audio encoder | 159 | 0 |
| Shared backbone | 3,880 | 0 |

CPU shape/control events remained present. The report and all six CPU/CUDA
profiles are under `/tmp/fractal-search-quickwins-parity-20261008/`; the run log
is `/tmp/fractal-search-quickwins-parity.log`. The CUDA daemon smoke also passed
real indexing/search, provider forwarding/retention, invalid-device rejection
without CPU retry and clean stop (`/tmp/fractal-search-quickwins-smoke-cuda.log`).
The hardware runner succeeded and resumed the previously installed service,
recorded in `/tmp/fractal-search-quickwins-hardware-state.json`.

The [synthetic indexing A/B comparison](README.md#development-optimization-validation-2026-10-08)
observed 7.119 → 5.130 seconds for eight images at batch size 2, 9.56 → 5.18 CPU
seconds and 24 identical search scores. Provider/device, six runtime threads,
image fixture, affinity to CPUs 10 and 11 and nice level 10 were unchanged.
This single constrained development-build sample compared an installed Rust
1.98.1 binary with a Rust 1.99.0 build, so it cannot isolate the source
optimizations or establish unrestricted/release performance. Full CUDA batch
sizes 1–8 were not rerun; this evidence does not supersede the batch-four
allocation failure documented below.

### EmbeddingGemma 2 hardware validation on Predator, 2026-10-07

The merged EmbeddingGemma 2 source at `317e39a` passed real inference on NVIDIA
GeForce RTX 2080 Max-Q (8 GiB), driver 610.57.04, using the existing private
CUDA 12.8.90/cuDNN 9.10.2.21 libraries and the eleven verified pinned assets at
`~/.cache/fractal-search/models/embeddinggemma2-daa72c51`. The installed FG-CLIP2
plugin and its catalog were preserved; all validation catalogs were synthetic
and temporary.

`validate_cuda` passed twelve CPU/CUDA comparisons across single images,
mixed-shape image batches, queries, documents, audio and video. Minimum cosine
was `0.9999999999971763`; maximum component error was `2.942979335784912e-7`.
Actual ORT profiles recorded:

| Graph | CUDA matrix/attention events | CPU matrix/attention events |
| --- | ---: | ---: |
| Vision encoder | 1,584 | 0 |
| Audio encoder | 159 | 0 |
| Shared backbone | 3,880 | 0 |

CPU shape/control operations remained in those CUDA sessions. The report and
six profiles are under `/tmp/fractal-predator-cuda-parity-20261007/`.
`smoke_cuda.py` also passed real daemon indexing/search, provider retention and
invalid-device rejection without CPU retry.

The opt-in `image_batches_preserve_single_image_embeddings_on_cpu_and_cuda`
test passed CPU sizes 1–8 for both geometry groups. CUDA passed sizes 1–3 for
the tiny/elongated group, then size 4 failed in `MultiHeadAttention` while
allocating a 2,537,256,960-byte buffer. The test exited nonzero without retrying
at a smaller size or on CPU. CUDA sizes 5–8 and the landscape/portrait group
were not reached. The batch log is `/tmp/fractal-predator-batch-1-8.log`.
This establishes a memory limit under the current desktop conditions, not a
passing CUDA batch-eight result. The resident previous-model service was kept
running; after the test it used 638 MiB of the GPU's 8,192 MiB.

These checks ran sequentially with nice level 10 and affinity to logical CPUs
10 and 11, with no concurrent compilation. The benchmark's six ORT threads
shared that affinity. Its recorded timings are resource-constrained diagnostic
measurements, not unrestricted desktop or release-performance results.

### Historical FG-CLIP2 evidence from NVIDIA RTX 2080 Max-Q (8 GiB), 2026-10-02

Both host runs passed all nine embedding comparisons: minimum image cosine
was 0.999999999888 and minimum text cosine exceeded 0.9999999999996. Profiles
recorded 4,464 CUDA vision kernel events, including 945 matrix/attention events,
and 3,720 CUDA text events, including 776 matrix/attention events. No matrix or
attention event ran on CPU. The vision graph also recorded 999 CPU shape/control
events. Concurrent compilation affected the first run's timings. A second run
with compilation and other inference paused measured:

| Operation | CPU median | CUDA median | Ratio |
| --- | ---: | ---: | ---: |
| One 512×384 synthetic image | 655.29 ms | 50.41 ms | 13.0× |
| One short text query | 52.67 ms | 6.62 ms | 7.96× |

These are five warmed calls with six CPU threads, including preprocessing and
diagnostic profiling overhead. They exclude model verification/startup and do not
measure full-library indexing, image I/O or interactive cold-start latency.
The report and actual provider profiles were written under
`/tmp/fractal-search-cuda-benchmark-20261002-quiet/`.

The CUDA core test run passed all 25 tests. The separate daemon smoke passed
real GPU indexing/search, provider retention through a model-directory change,
explicit invalid-device errors with no CPU retry, and orderly shutdown. All
inputs and catalogs were synthetic and temporary.

Runtime diagnostics passed five Rust tests and six offscreen UI cases covering
valid results/environment, subprocess failure, malformed or oversized output,
a missing executable and the 20-second timeout. Full QML lint passed. The real
host CLI preflight reported all nine required libraries visible on driver
610.57.04; sandboxed NVML access correctly produced an inconclusive result.

Historical GPU measurements above used the previous model. They do not validate EmbeddingGemma 2 placement, parity, memory or latency. All three graphs must pass a new actual-GPU run. Warmed timing fields still cover image/query calls only.
