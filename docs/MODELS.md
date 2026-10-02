# Native FG-CLIP2 models

Fractal Search runs FG-CLIP2 through native ONNX Runtime. The [model manifest](../crates/fractal-search-core/models/fgclip2.json) pins every asset by size and SHA-256 to `zihuv/fg-clip2-base-onnx` revision `33f3b9d426dfa8aeda700fd3b8fc5f7b6e3a9c40`.

## Assets and setup

Use **Set up model** in the panel, or the optional developer downloader:

```bash
python3 scripts/fetch-fgclip2-onnx-probe.py /absolute/model/cache
```

Downloads total approximately 1.14 GB. Existing assets must pass verification; corrupt or incomplete files fail visibly. The default persistent cache is `$XDG_CACHE_HOME/fractal-search/models/fgclip2-33f3b9d4`, normally under `~/.cache`. Never put model weights in the source repository or use a temporary directory for the installed application.

| Asset | Purpose |
| --- | --- |
| `visual.onnx` | Dynamic-patch image encoder. |
| `text.onnx` | One short text query per inference call. |
| `vision_pos_embedding.bin` | Float32 16×16×768 table for native positional interpolation. |
| `text_token_embedding.bin` | Float16 256,000×768 table, converted to float32 for graph input. |
| `tokenizer.json` | Pinned tokenizer configuration. |

Upstream: [FG-CLIP](https://github.com/360CVGroup/FG-CLIP), [original model](https://huggingface.co/qihoo360/fg-clip2-base), [ONNX export](https://huggingface.co/zihuv/fg-clip2-base-onnx). The export declares Apache-2.0; preserve model attribution and license when distributing assets. The development bundle downloads assets separately. See [NOTICE](../quickshell-app/packaging/NOTICE).

## Embedding contract

The catalog profile is `fgclip2-base-33f3b9d4-dynamic-torch291-short64-v1`. Images and text produce finite, nonzero 768-dimensional vectors which the runtime normalizes to unit length. The catalog stores one vector per image. Model/profile changes require coordinated reindexing; incompatible vectors must never be mixed.

Image indexing chooses each image's own dynamic patch budget and reproduces the pinned Torch 2.9.1 preprocessing, including positional interpolation for narrow images. Text uses lowercase input, a fixed length of 64 tokens and the short-query graph. Longer input is token-truncated according to that contract; the separate 196-token long mode is unsupported. Batch-size changes must preserve preprocessing and profile semantics.

CPU and CUDA use the same FP32 profile. CUDA disables TF32 and fails visibly when its selected device or runtime cannot initialize. See [CUDA validation](../quickshell-app/CUDA.md) for device requirements, numerical comparisons and graph profiling.

## Validation

The 2026-10-01 synthetic CPU audit covered 35 fixtures and 43 embeddings. Minimum image cosine against the pinned PyTorch reference was 0.9999999915629018; minimum text cosine was 0.9999999999988171. A temporary service catalog compared 192 scores across 24 images and eight queries; top-result and top-five membership matched for all queries. This establishes numerical parity on those fixtures, not semantic accuracy or recall on a labeled photo library.

Fixtures cover dynamic patch budgets, mixed image shapes, JPEG/PNG/WebP/HEIC decoding, narrow/tiny images, Unicode and truncated queries. The generator uses cached upstream revision `430fbc8a912c86fd4de601381b6245a0edab22f0` and performs no model download or catalog write.

Optional reference generation requires an isolated Python environment with Torch 2.9.1, Transformers 4.57.6, Tokenizers 0.22.2, Pillow 12.1.0, NumPy and pillow-heif. Prepare the upstream model/tokenizer/processor cache separately. The generator explicitly loads the pinned upstream model's custom code with `trust_remote_code=True`; inspect that pinned code before preparing a new reference environment. These tools are not needed for normal plugin operation.

```bash
python3 scripts/prepare-fgclip2-onnx-fixtures.py /absolute/model/cache /absolute/new/fixtures
cargo run --locked -p fractal-search-core --example validate_fgclip2 -- \
  /absolute/model/cache /absolute/new/fixtures/fixtures.json
python3 quickshell-app/tests/compare_fgclip2_retrieval.py \
  target/debug/fractal-search /absolute/model/cache /absolute/new/fixtures/fixtures.json
```

Use synthetic inputs and temporary catalogs. GPU comparisons and real-inference service smoke commands are documented in [CUDA.md](../quickshell-app/CUDA.md) and the [plugin guide](../quickshell-app/README.md).
