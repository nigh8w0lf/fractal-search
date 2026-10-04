# CUDA processing in Fractal Search

Fractal Search supports native CUDA inference in its Omarchy Shell plugin backend
and read-only MCP server.

## Selection and behavior

CPU remains the default. A CUDA-enabled build supports both providers; select CUDA
explicitly with `--provider cuda` or `FRACTAL_SEARCH_PROVIDER=cuda`. Select a GPU
with `--device-id N` or `FRACTAL_SEARCH_DEVICE_ID=N`; CUDA defaults to device 0.
Device IDs are rejected for CPU selection. Unknown providers, unavailable CUDA
build support, missing libraries and invalid GPUs fail visibly; the runtime never
retries a failed CUDA session on CPU.

CUDA accelerates the native FG-CLIP2 vision and text ONNX sessions. Decoding,
patch preparation, tokenization and LanceDB retrieval remain on CPU. ONNX Runtime
also deliberately places small vision shape/control operations on CPU; startup
logs state this. The validation below verifies that matrix and attention kernels
execute on CUDA in both models. TF32 is disabled to preserve the FP32 embedding
profile. Existing CPU embeddings use the same profile; changing providers does
not require a reindex.

`fractal-search providers` returns compiled capabilities without opening a catalog,
loading a model or initializing a GPU. The normal status response reports
`provider` (`CPU`/`CUDA`) and `deviceId` (null for CPU). Changing the model directory
retains the selected provider/device. The daemon explicitly forwards both to its
resident worker.

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

## Temporary development dependencies

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
cargo build --locked -p fractal-search --features cuda
cargo build --locked -p fractal-search-core --features cuda --example validate_cuda
cargo test --locked -p fractal-search-core --features cuda

env LD_LIBRARY_PATH=/tmp/fractal-search-cuda-deps/lib:"$PWD/target/debug/examples" \
  target/debug/examples/validate_cuda \
  "$HOME/.cache/fractal-search/models/fgclip2-33f3b9d4" \
  /tmp/fractal-search-cuda-validation

env LD_LIBRARY_PATH=/tmp/fractal-search-cuda-deps/lib:"$PWD/target/debug" \
  python3 quickshell-app/tests/smoke_cuda.py target/debug/fractal-search \
  "$HOME/.cache/fractal-search/models/fgclip2-33f3b9d4"
```

Choose a new benchmark output directory each time. The explicit `target/debug`
or `examples` loader path is required for development binaries: ORT's provider
libraries sit beside them but are loaded dynamically. The packaged launchers
configure this path automatically.

`validate_cuda` creates three synthetic PNG images, compares CPU/CUDA normalized
vectors for single images, a mixed-shape batch and three queries, and requires
cosine similarity of at least 0.9999. It records five warmed image/query calls,
writes `report.json`, and examines actual ORT profiles to reject CPU matrix or
attention execution in the CUDA sessions. Profiling is opt-in for this benchmark;
normal desktop inference does not write profiling files.

`smoke_cuda.py` uses temporary catalogs and sockets. It verifies capabilities
without catalog writes, invalid configuration rejection, explicit CLI selection
overriding the environment and reaching the worker, model-directory changes,
real indexing/search, invalid-device failure without CPU retry and clean shutdown.

### Evidence from NVIDIA RTX 2080 Max-Q (8 GiB), 2026-10-02

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
