# ColBERT-style retrieval with EmbeddingGemma 2

## Conclusion

The pinned ONNX checkpoint exposes token vectors, so a ColBERT-style experiment is technically possible without re-exporting the graph. That does not establish that token MaxSim improves this model's retrieval quality. This branch implements pooled embeddings and Text-only BM25/RRF, not late interaction or token-vector storage.

The released model is documented around pooled multimodal embeddings and Matryoshka representations. The inspected model card and implementation do not establish MaxSim training or token-level retrieval evaluation; the complete original loss recipe was not established in this investigation. Pooled cross-modal alignment does not prove query-token-to-media-token alignment. [Google model card](https://huggingface.co/google/embeddinggemma-2), [Hugging Face implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py)

## Available outputs and scoring

The exact FP32 export at `daa72c51243991dfcaf9f9137d2c573d8f7790c0` exposes `last_hidden_state [batch, tokens, 768]` and `sentence_embedding [batch, 768]`. The graph projects contextual token representations from 512 to 768 dimensions, performs masked mean pooling and normalizes the pooled output. Token vectors are not independently normalized; a cosine MaxSim experiment must normalize each retained token. [Pinned export](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/tree/daa72c51243991dfcaf9f9137d2c573d8f7790c0)

ColBERT-style scoring takes each query token's best document-token cosine and sums those maxima. Original ColBERT trains on late-interaction relevance scores and uses its own query/document conventions. Copying its markers, query augmentation or masks into EmbeddingGemma without validation is not justified. Keep EmbeddingGemma's encoding prefixes; whether their output tokens participate in scoring is an evaluation choice. Do not remove every special-token position: repeated image/audio/video positions contain actual media features. [ColBERT paper](https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf), [official scoring implementation](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/colbert.py)

## Database support and cost

Pinned LanceDB 0.37.1/Lance 10 support multivector columns and summed best-match scoring. A distinct token column would use `List<FixedSizeList<Float32, 768>>`. Query vectors may be appended with `add_query_vector`, but column type determines the meaning: against the existing single-vector column this dispatches independent searches, not token MaxSim. Cosine is the appropriate supported metric for normalized token vectors. [Pinned Rust query planner](https://raw.githubusercontent.com/lancedb/lancedb/v0.37.1/rust/lancedb/src/table/query.rs), [multivector documentation](https://docs.lancedb.com/search/multivector-search)

Raw FP32 content-vector storage, excluding boundary tokens and index overhead:

| Representation | Per segment |
| --- | ---: |
| Current pooled vector | 3 KiB |
| 256 text tokens | 768 KiB |
| 280 image tokens | 840 KiB |
| 30-second audio, approximately 750 tokens | 2.2 MiB |
| 30 video frames × 140 tokens | 12.3 MiB |

For 10,000 full-budget images, raw vectors rise from about 31 MB to 8.6 GB. The backbone pass is shared, but normalization, copying, storage, I/O and scoring add work. Exact scoring grows with query tokens × document tokens × dimensions × candidates; whole-catalog brute force would be inappropriate. Published Matryoshka results concern pooled vectors and do not establish the quality of 128-dimensional token vectors. MaxSim also needs separate score metadata; a summed token score is not a single cosine or probability.

## Recommended next experiment, if requested

Retain pooled retrieval as the baseline and evaluate token-MaxSim reranking on a bounded shortlist before changing catalog storage. Compare dense-only, BM25/RRF and reranking on labeled exact-term and paraphrase queries; measure recall, nDCG/MRR, p95 latency and storage. Text is a practical first scope because complete passages are already cached privately. Media reranking without stored tokens needs a separately designed inference/cache path; the read-only MCP server must not start reading originals.

An implementation requires explicit token masks, per-token normalization, variable-length storage if retained, a scorer and a distinct retrieval profile. If the checkpoint lacks useful late-interaction signal, MaxSim-aware fine-tuning/distillation would require positive/hard-negative examples, multimodal pairs where relevant, a new checkpoint/export and reindexing. Neither that training nor a labeled retrieval benchmark was performed here.
