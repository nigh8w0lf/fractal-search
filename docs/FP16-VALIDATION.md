# FP16 graph change — 2026-10-13

The source now selects the upstream `model_fp16`, `vision_encoder_fp16` and
`audio_encoder_fp16` graphs from the same pinned EmbeddingGemma 2 revision
`daa72c51243991dfcaf9f9137d2c573d8f7790c0`. Their graph/data sizes and SHA-256 hashes
are pinned in the model manifest. All floating graph inputs/outputs remain
float32; most weights are float16 and some operations retain float32. No runtime
conversion, model substitution, provider retry or new production dependency was
introduced.

The eleven assets total **1,497,222,022 bytes**, down from 2,960,874,003 (49.4%
less). This is a download/storage comparison, not a measured VRAM or latency
improvement. The native output stays 768D; leading-512 truncation and L2
normalization still produce float32 stored/query vectors.

The native profile is `embeddinggemma2-daa72c51-fp16-multimodal-v2`; the indexed
profile is `embeddinggemma2-daa72c51-fp16-multimodal-mrl512-v3`. Schema remains 4.
Old FP32 catalogs, including 512D catalogs, are rejected without modification.
Private vector-cache keys include the new precision/profile/manifest. Reference
validation rejects old FP32 or unversioned fixtures in either output mode.

## Deployment boundary

Source validation initially preserved the running installation and catalog.
After explicit user authorization to replace the installation and discard its
indexed data, the CUDA bundle `0.1.0-70d00856ca27-close-cuda-0-5376c7db` was
installed on 2026-10-13. The service was drained before deleting the old catalog,
thumbnails and previews; no originals were modified. The backend created a
fresh schema-4 FP16/512D catalog, verified empty through the installed read-only
MCP endpoint (zero files, segments and sources). No user folders were reindexed.

Models are selected from
`~/.cache/fractal-search/models/embeddinggemma2-daa72c51-fp16`. CUDA device 0,
private library path, UI preferences, existing bindings and disabled login
registration were preserved. Installed binary hashes match the validated build;
the plugin responds with the new version, the service is active, and logs confirm
FP16 CUDA text warmup completed in 7.52 seconds. The desktop shell was not
restarted. Automatic cleanup pruned one obsolete runtime bundle.

## Validation

- `cargo fmt --all --check` and CPU/CUDA `cargo check --workspace --locked` passed.
- CPU and CUDA workspace suites each passed 155 tests; five opt-in core tests
  remained ignored in each default suite. The initial sandbox run blocked a
  temporary Unix socket; the complete suites passed with socket access.
- Eleven projection/reference-contract Python tests and 34 QMD evaluation tests
  passed. SearchPanel QML lint passed; its only UI change is the model-download
  size string. No new native UI interaction run is claimed.
- All eleven downloaded assets passed size/SHA-256 verification. ONNX graph
  inspection confirmed unchanged names, shapes and float32 interfaces.
- Python ONNX Runtime 1.22.0 CPU comparison of three synthetic queries (including
  Unicode) and one document against the old FP32 graph gave 512D cosine agreement
  0.9999997164–0.9999998105, maximum coordinate difference 0.000111994. This is
  numerical drift on four examples, not labeled retrieval-quality validation.

- The Rust native validator matched the independent Python ONNX Runtime text
  references in both indexed512 and native768 modes: four embeddings each,
  minimum cosine greater than 0.999999999999998. Token IDs and masks also matched.
  Explicit rejection checks passed for old FP32 fixture profiles in both modes.

- The opt-in native projection test passed for queries, documents, a two-image
  batch, audio and video: each indexed output matched leading-512/L2 projection
  of its corresponding native output and had unit norm.

- `validate_cuda` passed 12 CPU/CUDA embedding comparisons across queries,
  documents, single images, a three-image batch, audio and video. Minimum cosine
  was 0.9999830751, above the existing 0.9999 threshold. All vectors were finite
  and normalized. On the RTX 2080 Max-Q, profiling confirmed matrix/attention
  execution on CUDA in all three graphs, with zero such kernels on CPU.
  Shape/control nodes still execute on CPU as expected. The report is
  `/tmp/fractal-fp16-cuda-validation/report.json`. Timings included profiling and
  concurrent compilation; they are not a controlled precision-speed benchmark.

- The backend, MCP and validation examples built successfully with CUDA. Both
  binary capability commands report CPU and CUDA support.
- `smoke_multimodal.py` passed with CUDA desktop indexing and CPU read-only MCP
  queries against a synthetic temporary catalog. It covered image/text/audio/video
  segments, video soundtracks, stable rescans, failed-replacement preservation,
  scope filters, all-media cosine order, Text-only hybrid retrieval and cached
  MCP reads after deleting the test originals. Catalog hashes were unchanged
  by MCP access. No live user data was used.

No full reference-processor fixture regeneration, large-corpus retrieval
benchmark, FP32-versus-FP16 memory benchmark or batch-eight claim is made by
these checks.

## Intermittent non-finite CUDA outputs — 2026-10-19

The installed ONNX Runtime 1.22.0 reproduced a non-finite shared-backbone output
on the approved video copy's first audio segment, even outside catalog indexing
and the preparation pool. The complete video passed on CPU. Native and independent
NumPy audio preprocessing matched exactly, and encoder features were finite.
The two approved still images passed alone and together; the original failing
batch partners were not available for this targeted reproduction.

A synthetic image sequence reproduced the error in both Rust and an independent
Python 1.22.0 caller using the same CUDA provider libraries. An identical later
call could return a finite result. GPU instrumentation also exposed an impossible
intermediate reshape dimension. Arithmetic flags, strict normalization, disabled
graph optimization and memory-pattern/arena changes did not reliably fix the
reproduction. No such workaround, retry, clamping or provider substitution is
retained. The instrumentation did not establish a specific bad GPU memory access;
the bounded memcheck run completed one inference before being stopped.

The binding is updated to ort rc.12 / native ONNX Runtime 1.24.2, with matching
CUDA 12 provider libraries. Independent 1.24.4 diagnostic runs passed the same
synthetic sequence and repeated video/audio inputs. Upstream corrected missing
[CUDA opset-21 shape-kernel registrations](https://github.com/microsoft/onnxruntime/pull/26075)
after 1.22 and documented a related
[corrupted-shape failure](https://github.com/microsoft/onnxruntime/issues/27102).
These support the runtime-compatibility diagnosis; this investigation does not
claim to have isolated the exact upstream kernel responsible for every NaN.
The pinned native 1.24.2 validator passed 12 CPU/CUDA comparisons across images
(including a three-image batch), queries, documents, audio and video. Minimum
cosine was 0.9999830226; all outputs were finite and normalized. Added checks
compare repeated image batches, audio/video after modality changes, and warmed
image/query outputs against their first calls. Profiling confirmed CUDA
matrix/attention execution in all three graphs, with none on CPU. The report is
`/tmp/fractal-ort124-native-parity/report.json`; profiling and concurrent builds
make its timings unsuitable for an update-performance claim.

Separately, preparation-pool stop/abandon notifications now hold the predicate
mutex. Previously a notification could arrive between a worker's stop check and
its condition-variable wait, hanging cleanup on join. A regression test covers
that synchronization. Non-finite inference remains fatal; error messages identify
the encoder or shared-model row/provider and no longer suggest reducing image
batch size for every model error. Model assets, preprocessing, indexed profile,
schema and user catalogs are unchanged by these corrections.


CPU and CUDA workspace checks and unit suites passed after the runtime update:
169 tests per configuration, with seven opt-in core tests excluded from each
default suite. The initial CPU run's private-socket check was blocked by the
sandbox and passed with the required access. Eleven model-contract and 34 QMD
fixture/evaluation-contract tests passed, as did dependency-layout/vendor-integrity
checks. CUDA-13 packaging rejection was checked explicitly. The dependency update
keeps incremental compilation enabled; no model, provider, batch-size or numerical
fallback is introduced.


The rebuilt CUDA backend indexed the two approved images separately, together,
and as a four-image batch of copies, with zero failed or skipped files. Logs
confirmed real batch-two/batch-four inference rather than embedding reuse. The
approved 32.704-second video completed atomically with four rows: visual and audio
segments starting at 0 and 30 seconds. The service drained and stopped cleanly.
The original failing image-batch combinations were not reproduced exactly.

The synthetic multimodal smoke test also passed: all four media types, video
soundtrack, stable rescans, failed-replacement preservation, hybrid filtering and
read-only CPU MCP queries against CUDA-indexed data after synthetic originals
were removed. Temporary installer tests passed install/update, provider
persistence, catalog preservation, checksums, rollback, pruning and uninstall.
The approved-file report is `/tmp/fractal-reported-probe-k6sqr5ab/report.json`;
no live user catalog was used for these tests.

The final bundle was installed as `0.1.0-bbcf92fcf8ed-close-cuda-0-5376c7db`.
Installed binary/provider hashes matched the tested bundle, plugin ping returned
that version, and the service remained active with CUDA text warmup complete on
native 1.24.2. Settings, provider configuration and personal bindings were
hash-identical; login registration remained disabled. The required cuRAND library
was added to the existing private CUDA directory with its distribution metadata.
No catalog reset or reindex was performed.

Deployment exposed an installer omission: an active service was stopped but not
restarted. The installer now preserves active/inactive state on success and
restores prior activity after rollback, reporting rollback errors and retaining
recovery files. Temporary installer tests cover both states and injected start
failure recovery. These tests required host access for systemd unit validation;
all installed fixtures and service commands were isolated. Final deployment
verified the active-service path and pruned one obsolete runtime bundle.
Diagnostic copies and temporary catalogs were removed after validation; reports
and logs remain under `/tmp`. Temporary inference tools, scratch executables,
superseded bundles and the diagnostic Rust toolchain were removed; project
incremental/build caches were retained.

## CUDA arena retention — 2026-10-20

A watched scan failed on video segment 0 at an attention Transpose allocation of
1,376,256 bytes. Read-only host inspection found the installed backend holding
7,796 MiB on the 8,192 MiB RTX 2080 Max-Q, with 1 MiB reported free device memory.
The preceding log included image batches up to four and audio/video inference.
This was an allocation failure; it did not report non-finite embeddings.

The three resident CUDA graphs previously used separate, non-shrinking arenas
with geometric growth. The runtime now selects `kSameAsRequested` growth and
requests unused GPU arena reclamation after every graph call through
`memory.enable_memory_arena_shrinkage=gpu:N`. The shared model still selects only
`sentence_embedding`. Resident weights/live outputs remain allocated, while
unused scratch regions are released. CPU allocator behavior, graph precision,
preprocessing, batching, provider and segment budgets are unchanged. No retry,
quality reduction, provider substitution, model change or catalog migration is
introduced. See [ORT arena documentation](https://onnxruntime.ai/docs/get-started/with-c.html#features)
and the [pinned run-option contract](https://github.com/microsoft/onnxruntime/blob/v1.24.2/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h).

The isolated synthetic workload used image batches of 4/3/2/1/4 with differing
aspect ratios, a 30-second tone and a 30-second 768×1280 video. Retained process
GPU memory after the mixed sequence was 6,326 MiB before versus 1,710 MiB after.
Changing only arena growth had left 5,668 MiB retained, so per-run reclamation was
also necessary. A fresh-video-only run held 2,848 MiB before versus 1,130 MiB
after. These are `nvidia-smi` snapshots after completed calls, not transient peak
measurements or universal GPU requirements. Reallocation has a potential latency
cost; the concurrent builds and short runs do not establish a throughput change.

With explicit authorization, a temporary copy of the reported 720×1280,
11.966667-second video was indexed both fresh and after the mixed sequence. Both
its video and audio were searchable. Retained memory after the mixed case was
5,428 MiB before versus 1,712 MiB after. Across 57 search scores for three queries
in the fresh/mixed catalogs, maximum absolute before/after difference was
5.960464477539063e-8. The original live scan and its exact accumulated allocation
state were not replayed; even the isolated baseline completed this copied file.
The evidence establishes reduced retention and successful inference, not a
promise that every workload fits every GPU. Original media and the live catalog
were not used as test outputs, reset or reindexed.

Reports: `/tmp/fractal-video-memory-ehgl1qxh/report.json` (synthetic baseline),
`/tmp/fractal-video-memory-ujxx8293/report.json` (synthetic update),
`/tmp/fractal-video-memory-s_fkaivz/report.json` (approved-copy baseline),
`/tmp/fractal-video-memory-jzlu_x4z/report.json` (approved-copy update), and
`/tmp/fractal-video-memory-score-parity.json` (57-score comparison).

Current-source CPU/CUDA workspace checks and suites passed: 169 tests per
configuration, seven opt-in tests ignored in each default suite. Formatting and
diff checks passed. Native validation passed 12 CPU/CUDA embedding comparisons
(minimum cosine 0.9999830226), repeated cross-modality calls, and all three
CUDA graph profiles with zero CPU matrix/attention events. The report is
`/tmp/fractal-memory-parity/report.json`. Isolated installer checks passed
install/update, provider/configuration preservation, rollback, service activity
restoration, pruning and uninstall. Diagnostic media and temporary catalogs were
removed after the approved-copy/synthetic measurements; reports/logs were retained.

Installed bundle `0.1.0-750324cfc993-close-cuda-0-5376c7db` matched the validated
binary/provider hashes and returned that version through plugin ping. The
service remained active and completed CUDA text warmup. Settings, provider
configuration and personal bindings were hash-identical; login registration
remained disabled. The installer pruned one obsolete runtime and preserved the
catalog and model cache. No live-library rescan was initiated. The temporary
Rust toolchain and staging bundle were removed; incremental/build caches remain.
