# Default 512-dimensional embedding validation

Validated 2026-10-08 against source tree `ae6736c0589a188d9c9e654f09c7c1797d631cb0`, published as implementation commit `6548728de6c53ecb2a405da7b1eb8b7c8aba797e` (local verification commit `ce26a9ae6eeddbce972f812228d6bdc998079c90`), based on QMD checkpoint `cf09d084ffd07bdd81fb8b4f79b91b5eab7427e9`. This is the default indexed/query contract; native full-width APIs exist only for reference validation. No user dimension selector, migration, installation or catalog deletion was performed.

## Contract

- Native FP32 graphs, pinned asset manifest, tokenizer, task prefix, pooling and image/video budgets are unchanged
- Query, document, image single/batch, audio and video indexed output takes the leading 512 of exactly 768 native values and L2-normalizes again
- Native input validation rejects malformed dimensions, nonfinite values anywhere (including discarded coordinates), zero vectors and zero prefixes
- Catalog identity is `embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2`, dimension 512 and projection `native768-leading512-l2-v1`; schema remains 4
- Derived text-vector cache identity changes with the projection/profile. File-state reuse for all modalities is profile-bound
- Full-width catalogs are rejected before maintenance/writes. Writer, reader and read-only agent regressions verify original bytes remain unchanged. The agent returns path-free reindex guidance
- Source rows retain null vectors. Raw indexed FP32 payload is 2,048 bytes, versus 3,072 bytes at native width. This does not reduce native model computation or model VRAM

## Passed cloud checks

- `cargo fmt --all --check`
- `cargo check --workspace --all-targets --locked`
- `cargo check --workspace --all-targets --locked --features cuda`
- `cargo test --workspace --locked -- --skip socket_is_private_exclusive_and_never_overwrites_an_ordinary_file`: **153 passed, 5 ignored, 1 filtered**
- The same available workspace test suite with `--features cuda`: **153 passed, 5 ignored, 1 filtered**. This is compile/unit coverage, not CUDA inference
- `cargo test --workspace --locked native_audio_video_and_soundtrack_segments_share_one_catalog -- --ignored --nocapture`: **1 passed**, using native FFmpeg extraction and a synthetic embedder, not ONNX model inference
- `cargo build -p fractal-search --bins --locked`; built backend and MCP `providers` diagnostics both report CPU-only capability as expected
- `python3 scripts/test_embeddinggemma2_contract.py`: **10 passed**
- `PYTHONPATH=scripts python3 -m unittest discover -s scripts/qmd_eval -p 'test_*.py' -v`: **34 passed**
- `python3 scripts/evaluate-qmd.py validate`: **202 corpus documents, 24 queries**. Corpus validation is not a new retrieval-quality measurement
- `python3 quickshell-app/tests/smoke_hybrid_text_ui.py` and `smoke_qmd_ui.py`: model-free source/contract checks passed
- Changed/new Python files compile; fixture/comparator CLI help and direct comparator rejection of native vectors or bare non-unit slices passed
- `git diff --check`

Rust tests include leading-coordinate selection, unit norms, normalized/raw projection equivalence, batch/single projection equality, projected dot versus truncated cosine, full-width write/query/cache refusal, all-modality file-state identities, nullable source placeholders and preserved old catalogs. Independent read-only code review found no remaining concrete issue after the agent error-path fix.

## Explicit limits and native follow-up

The cloud environment has no complete verified model cache, no GPU/Omarchy session and no Quickshell executable. No new model downloads were attempted. Native model parity, real image-batch parity, original-checkpoint parity, end-to-end service retrieval and user-library quality/latency were **not** measured. The multimodal UI harness could not launch because `quickshell` is absent.

The one filtered Unix-socket privacy test requires a bind denied by this cloud sandbox; the prior QMD validation documents the failed supported escalation. No production code or sandbox policy was weakened. Five ignored tests remain explicitly gated, including real-model multimodal projection and image-batch parity.

With an existing verified model cache, run `every_modality_and_query_projects_the_same_native_space` and `image_batches_preserve_single_image_embeddings_on_cpu_and_cuda` explicitly with `FRACTAL_SEARCH_BATCH_TEST_MODELS` set. Use the fixture generator/validator's explicit `--native-output` mode for unchanged full-768 graph parity; their default mode validates indexed512. The retrieval comparator can explicitly project historical native fixtures with `--native-fixtures`. See [model contract and commands](MODELS.md).

Previously published model-card scores and historical QMD/SQLite timings are not measurements of this default change. SQLite's separately published 768D experimental checkpoint needs its own 2,048-byte BLOB guard/profile/fixture follow-up before use with this contract; simply cherry-picking the common code is insufficient.
