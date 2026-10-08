# Native EmbeddingGemma 2 models

Fractal Search uses the exact [onnx-community/embeddinggemma-2-ONNX export](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/tree/daa72c51243991dfcaf9f9137d2c573d8f7790c0), pinned to revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0`. The [manifest](../crates/fractal-search-core/models/embeddinggemma2.json) records every asset's byte count and SHA-256. This is the multimodal EmbeddingGemma 2 model, not the older text-only EmbeddingGemma 300M.

## Setup and assets

Use **Set up model** in the panel, or the optional development downloader:

```bash
python3 scripts/fetch-embeddinggemma2.py /absolute/model/cache
```

The eleven pinned FP32 assets total 2,960,874,003 bytes, approximately 2.96 GB (2.76 GiB). Existing files must pass size/hash verification; corrupt or incomplete files fail visibly. Nested graph paths and distinct per-file `.part` files are preserved. Only verified downloads become final assets; cancellation retains previously verified assets.

The persistent application cache is `$XDG_CACHE_HOME/fractal-search/models/embeddinggemma2-daa72c51`, normally under `~/.cache`. Select the cache root containing `tokenizer.json` and `onnx/`, not the `onnx/` subdirectory. Do not store weights in the repository or use a temporary directory for an installed application.

| Assets | Purpose |
| --- | --- |
| `onnx/model.onnx` and `.onnx_data` | Shared text backbone and normalized sentence output |
| `onnx/vision_encoder.onnx` and `.onnx_data` | Image/frame features |
| `onnx/audio_encoder.onnx` and `.onnx_data` | Audio features |
| `tokenizer.json`, `tokenizer_config.json` | Exact text vocabulary and tokenization |
| `config.json`, `processor_config.json`, `preprocessor_config.json` | Pinned model/processor contract and reference tooling |

The export declares Apache-2.0. Preserve applicable licenses and attribution when distributing assets. Development bundles download assets separately; see [NOTICE](../quickshell-app/packaging/NOTICE) and the [Google model card](https://huggingface.co/google/embeddinggemma-2).

## Embedding and catalog contract

The pinned FP32 graph and model asset manifest are unchanged: `sentence_embedding` natively contains 768 dimensions under the native graph contract `embeddinggemma2-daa72c51-fp32-multimodal-v1`. Production uses a separate indexed contract, `embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2`: validate the native output, keep the **leading 512 values**, then **L2-normalize that prefix**. Apply this identical projection to every image, text passage, audio segment, temporal visual-video segment and query. Indexed vectors must be finite, nonzero and unit normalized. Do not merely slice a normalized native vector and assume it stays normalized.

Schema stays **4**, with its separate lexical/extraction contracts. Catalog metadata validates the indexed model profile, dimension 512 and `fractal.embedding_projection = native768-leading512-l2-v1` together; a missing or conflicting projection guard is incompatible. The private vector-cache format is also versioned and keys its entries by the indexed dimension/projection contract. Existing schema-4 catalogs carrying the previous 768-dimensional profile are incompatible and fail clearly. Use a separate fresh `--data-dir /absolute/new/catalog` and explicitly reindex the desired folders; the current installation preserves old catalogs and never deletes, converts or mixes them automatically. The verified pinned model cache can be reused. No 768/256 dimension selector is exposed to users; explicit full-768 output is for native reference/parity tooling only.

This follows [Google's Matryoshka guidance](https://huggingface.co/google/embeddinggemma-2#3-matryoshka-dimension-truncation). Published model-card scores at 512 versus native 768 dimensions are MTEB multilingual v2 61.17 versus 61.36, MIEB lite 64.32 versus 64.64, and MMEB v2 overall 58.38 versus 59.01. These are upstream benchmark results, not measurements of this app or its labeled corpus.

Raw FP32 vector payloads fall from 3,072 to 2,048 bytes per segment, a one-third reduction before database/index overhead. Similarity scoring processes fewer coordinates, but application latency improvements require measurement. The native graph still computes all 768 outputs: this change does **not** reduce model weights, inference work, model RAM or VRAM.

The shared graph takes int64 `input_ids`/`attention_mask` and float32 image/video/audio feature matrices with 512 columns. Absent modalities still receive real empty `[0, 512]` tensors. Use `sentence_embedding`, which already includes the graph's native projection, masked mean pooling and 768D normalization, then perform the separate leading-512/L2 indexed projection described above. `last_hidden_state` is not the current retrieval representation. Video features go to `video_features`, not `image_features`.

### Queries and documents

Queries automatically receive `task: search result | query: `. Documents receive `title: {filename} | text: {passage}`, with `none` for an empty title. Users enter ordinary text without manually supplying prefixes. Tokenization preserves case/Unicode and inserts BOS/EOS from the pinned template. Literal strings such as `<|image|>` in text are treated as text content, not missing media features.

The graph context limit is 8,192 tokens including prefixes and boundary tokens. Application queries are limited to 4,096 UTF-8 bytes. UTF-8 `.txt`, `.md` and `.markdown` files up to 2 MiB are divided into passages of at most 4,096 bytes with 256-byte overlap, at most 1,024 passages. Whitespace-only chunks are skipped. Titles are UTF-8-safe filename prefixes of at most 1,024 bytes. Excessive inputs fail explicitly; there is no silent tokenizer truncation.

Schema 4 keeps filename plus the full passage privately for BM25 and a separate public passage-only excerpt of at most 2,048 bytes. Text-only search fuses BM25 and semantic ranks; other scopes remain semantic. See [hybrid text search](HYBRID-TEXT-SEARCH.md). PDF/Office extraction, OCR and transcription are not implemented.

### Images

Native preprocessing converts RGB, performs aspect-preserving antialiased bicubic uint8 resizing and rescales by `1/255`, without CLIP mean/std normalization. It uses 16×16 patches, pooling kernel 3, a 280-soft-token budget and 2,520 padded patch slots. Positions are `[column, row]`, padded with `[-1, -1]`. Each sample retains its independent aspect ratio and token count.

Consecutive images can be batched (saved setting 1–8, default 1). Skips, unchanged files and batch tails may yield smaller batches. Failure never triggers a smaller-batch retry or provider fallback. A larger batch needs more memory; no fixed RAM/VRAM envelope is claimed.

### Audio and video

System FFmpeg decodes mono 16 kHz float PCM in segments up to 30 seconds. Audio extraction uses a periodic 320-sample Hann window, hop 160 and FFT 512; prepends 160 zeros; right-pads to a multiple of 128; and calculates 128 unnormalized HTK mel bands from 0–8 kHz over magnitude, not power. It computes `log(mel + 0.001)` without normalization, dither or preemphasis. Boolean masks downsample every four frames, producing at most 750 soft tokens per 30 seconds. Fewer than 161 samples are rejected.

Video visual segments are at most 30 seconds, sampled at 1 fps with at most 30 frames and 140 soft tokens per frame. Their ordered frame features form one backbone sequence; the vector is not an average of independent frame embeddings. The app uses deterministic FFmpeg `fps=1:round=up:start_time=0` sampling, with no automatic rotation or scaling. Exact frame-index parity with every upstream variable-frame-rate loader is not claimed. Model-level fixtures use explicit sampled frames.

A video's soundtrack is indexed separately as timestamped audio segments. The first audio and first non-cover-art video track are selected; alternate tracks/subtitles/attachments are not indexed. Stream offsets are normalized to the container origin. Tiny final intervals adjust the preceding boundary without losing or overlapping source content.

Audio/video inputs are capped at 2 GiB/two hours. Video frames above 2,073,600 pixels or a 4,096-pixel side are rejected. Decoders have local-only protocol/demuxer allowlists, time/output/memory bounds and process-group cleanup. Corrupt packets and error-level stderr fail visibly, even if FFmpeg exits successfully. A failed segment prevents partial replacement of the file's previous complete segment set.

## Providers and validation

CPU and CUDA use the same native FP32 graphs and normalized 512D indexed profile. CUDA is explicit, disables TF32, and fails visibly without retrying on CPU. The shared backbone loads for text or any media inference; vision/audio encoders remain lazy. Browsing and metadata load no models. See [CUDA.md](../quickshell-app/CUDA.md) for actual provider-placement validation.

Reference generation uses Hugging Face's Gemma4 image/audio preprocessing and the pinned tokenizer, followed by the pinned export in Python ONNX Runtime. It does not compare with Google's original PyTorch checkpoint. Optional development dependencies are NumPy, Pillow, Torch, Torchvision, Transformers exposing these processors, and ONNX Runtime; production inference remains Rust-only. The generator records dependency versions and performs no remote model-code execution or automatic asset download.

```bash
python3 scripts/prepare-embeddinggemma2-fixtures.py /absolute/model/cache /absolute/new/fixtures
cargo run --locked -p fractal-search-core --example validate_embeddinggemma2 -- \
  /absolute/model/cache /absolute/new/fixtures/fixtures.json
python3 quickshell-app/tests/compare_embeddinggemma2_retrieval.py \
  target/debug/fractal-search /absolute/model/cache /absolute/new/fixtures/fixtures.json
```

The generator and Rust validator default to the **indexed 512D** contract. `environment.json` records `output_mode`, `embedding_dimension`, `native_embedding_dimension` and `model_profile`, alongside pinned export identity and dependency versions. All fixture modalities go through the same leading-512/L2 reference projection. The retrieval comparator checks the indexed profile and rejects mismatched fixture modes instead of silently comparing 768D scores with a 512D catalog.

For explicit **native 768D graph parity**, use separate fixture output:

```bash
python3 scripts/prepare-embeddinggemma2-fixtures.py --native-output \
  /absolute/model/cache /absolute/new/native-fixtures
cargo run --locked -p fractal-search-core --example validate_embeddinggemma2 -- \
  --native-output /absolute/model/cache /absolute/new/native-fixtures/fixtures.json
```

Existing full-768 reference fixtures remain useful in this explicit native validator mode. To compare them with the production 512D catalog, the retrieval comparator's `--native-fixtures` option explicitly projects their references first; it does not switch the catalog or queries to 768D. Freshly generated native fixtures include mode metadata; pre-mode native fixtures require this explicit option and every vector is still checked as native 768D.

Model-free projection, fixture-mode and evaluation-protocol checks:

```bash
python3 scripts/test_embeddinggemma2_contract.py
PYTHONPATH=scripts python3 -m unittest discover -s scripts/qmd_eval -p 'test_*.py' -v
```

For metadata-only preprocessing validation, pass `--preprocess-only` to both generator and validator and provide the five pinned metadata/tokenizer JSON files. HF reference libraries are still required; no weights or inference are used. Use `--max-image-batch` to bound full reference inference memory.

The validator checks tokens, masks, patches and feature tensors, then requires embedding cosine at least 0.9999 in the explicitly selected indexed/native space. The retrieval comparator requires score error at most `1e-4`. These are acceptance thresholds, not claims of unrun checks. Synthetic parity does not establish labeled-library relevance, large-corpus latency, GPU memory use or batch-eight inference. Previous results do not validate reconstructed source; rerun checks against the current tree and record actual limitations.
