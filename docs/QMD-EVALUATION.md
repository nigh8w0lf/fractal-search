# QMD labeled text-retrieval baseline

This is an executable, offline **retrieval-quality baseline**, separate from
[`compare_embeddinggemma2_retrieval.py`](../quickshell-app/tests/compare_embeddinggemma2_retrieval.py).
That older script checks numerical agreement against reference-model vectors; it
cannot establish whether results answer a user's query. This harness scores
explicit relevance judgments using real native indexing and retrieval.

No measured retrieval-quality improvement is claimed by adding this harness.
The committed corpus and metric tests are usable without model weights. A real
quality/latency run requires the pinned model cache and a built native adapter.
Missing assets, inference failures, incomplete output or scope leaks fail
explicitly; they never become substitute-vector results or passing gates.

## Components and scope

- [`scripts/evaluate-qmd.py`](../scripts/evaluate-qmd.py): executable CLI
- [`scripts/qmd_eval/corpus.json`](../scripts/qmd_eval/corpus.json): synthetic documents, query labels and provisional category gates
- [`scripts/qmd_eval/metrics.py`](../scripts/qmd_eval/metrics.py): conventional metrics and passage-judgment binding
- [`scripts/qmd_eval/test_metrics.py`](../scripts/qmd_eval/test_metrics.py): deterministic metric, label, confinement and failure tests
- [`examples/evaluate_qmd.rs`](../crates/fractal-search-core/examples/evaluate_qmd.rs): native adapter using `NativeRuntime`, `Catalog::index` and `Catalog::search_text_evaluation`

The Python tooling uses only the standard library. The adapter links the normal
core dependencies; it is a development example, not a new desktop or MCP search
mode. An installed desktop binary alone does not contain this adapter: build it
from the same source and use the existing verified model cache. The harness
never installs dependencies, downloads weights, starts a desktop service,
opens a user catalog or writes to an original user document. It creates its own
synthetic files and catalog in an automatically cleaned temporary directory.

The three retrieval arms share a catalog, actual EmbeddingGemma 2 passage vectors,
literal ICU BM25 tokenizer and identical Text/source prefilters:

1. **BM25-only:** native lexical ranking; no query-model load or query embedding
2. **Dense-only:** exact cosine ranking of the actual model query embedding
3. **Hybrid:** current equal-weight, one-based RRF with k=60

No LLM expansion, reranking, alternate tokenization or synthetic-vector scoring is
used. The BM25 arm still uses the same real indexed corpus; omitting query
inference must not be confused with indexing without model assets.

## Corpus and judgments

The versioned fixture materializes **202 documents**: 22 hand-authored positive or
related documents plus 180 deterministic negative notices. There are **24
queries**, four in each category:

| Category | What it exercises |
| --- | --- |
| `exact_filename` | Literal Markdown/text filenames, punctuation and numbered titles |
| `identifier` | Error code, camelCase setting, part number and incident ID |
| `paraphrase` | Needs expressed differently from the relevant text |
| `multilingual` | Spanish, Japanese, German and Arabic; same-language retrieval |
| `long_document` | Evidence deep in two >20 KB files; correct file versus correct passage |
| `folder_filter` | Conflicting same-named files in separate sources, plus graded related documents |

The negative notices are explicitly irrelevant by fixture construction. They
provide more than 100 documents so Recall@100 is not automatically perfect for
dense ranking. These are modest distractors, not a representative hard-negative
collection. Some individual folder scopes contain fewer than 100 documents;
there Recall@100 is intentionally easier and MRR/nDCG matter more.

Each fixture document has an explicit stable canonical ID. Search hits map to it
through the registered synthetic path. Filenames alone are never identities: the
two `release-checklist.md` files remain distinct. Native opaque asset IDs are
retained as diagnostics only. Passage IDs are `document_id#segment_index` for the
actual indexed chunk contract.

Document judgments have relevance grades 1–3. A judgment with evidence strings
marks the native passages containing at least one of those strings relevant at
the same grade. Every evidence string must occur intact in at least one native
passage; absent or boundary-split evidence is an error, not silently dropped.
An explicitly empty evidence list means all passages in that document are
relevant, used for filename lookup. Overlapping passages may both be relevant.

The Rust adapter extracts the manifest with the **same real-token-count chunker
used by indexing**, and records byte/line locations and complete synthetic
passage text. Python does not approximate chunk boundaries or inspect only the
2,048-byte public excerpt. The indexed and extracted passage counts must agree.
A changed chunker can legitimately change passage qrels. The native index,
search and cold outputs separately emit `extraction_profile` from
`media::TEXT_EXTRACTION_PROFILE`, which identifies the structural/token-budget
chunker, and `text_search_profile`, which identifies the lexical/search contract.
The harness rejects profile mismatches and saves both in the report. Preserve
both profiles, the model profile and the fixture hash when comparing reports.

## Metrics and result budgets

Metrics are computed per query, then macro-averaged overall and separately for
each category. Both canonical-document and passage-level metrics are reported:

- **Recall@100:** relevant identities in the first 100 ranks divided by all
  relevant identities, not success/hit rate
- **MRR@100:** reciprocal rank of the first positive judgment within 100, or zero
- **nDCG@6:** DCG over the first six ranks divided by ideal DCG@6, using
  `(2^grade - 1) / log2(rank + 1)` and one-based ranks

Unjudged identities have zero relevance. Missing rankings or missing positive
judgments are errors. Document ranks collapse repeated passages at their first
occurrence **before** the document cutoff; passages retain their separate ranks.
Thus ten chunks from one file cannot inflate document recall or DCG. A file with
a relevant paragraph may have high document quality but low passage quality if
the wrong chunk appears first.

Quality runs request **1,000 native passages** per query. The existing hybrid
candidate formula reaches its 1,000-per-leg cap at this result limit. The supplied
fixture must index at most 1,000 passages; larger fixtures are rejected instead
of silently evaluating an unknown truncated document ranking. This yields
conventional top-100 canonical-document metrics while keeping the existing
bounded native search API. Dense and hybrid outputs must contain exactly every
in-scope manifest passage, including negatives; incomplete sets are errors even
if all positive judgments were returned. BM25 may return fewer passages because
lexical matching is sparse. It is a controlled high-recall baseline, not a claim
that a default six-result search uses the same candidates.

Production-limit-six searches are measured separately, with the current
**100 candidates per leg**. The report also includes their actual passage nDCG@6
and mean distinct documents in six result slots. These fields help detect a
candidate-depth or repeated-chunk issue hidden by the high-recall baseline.

Native ties use opaque asset IDs, so rebuilding the temporary catalog can change
tied ranks. Save raw results and compare score/profile/fixture provenance; do not
claim bit-for-bit ranking reproducibility just because the scoring code is
deterministic.

## Latency protocol

Default settings use one untimed warmup pass, three timed repetitions of all 24
queries, and three cold samples for each retrieval arm. The warm query order is
rotated deterministically between repetitions.

- **Warm query latency:** query embedding, where applicable, plus native catalog
  search and result conversion, at limit six. Catalog opening, warmup, quality
  passes, desktop IPC and rendering are excluded
- **Cold first-result latency:** a fresh adapter process opens the same temporary
  catalog and performs one query without warming a model. Includes adapter
  manifest validation, catalog open and any actual model verification/load
- **Cold process wall latency:** the harness measures from process launch through
  exit, also including startup, serialization and I/O
- **Index timing:** adapter preparation, real model indexing and checkpointing;
  not a production-scale throughput benchmark

Cold processes are separate for BM25, dense and hybrid. BM25 deliberately avoids
model load. The default cold samples use the same filename, paraphrase and
long-document queries for all three modes. The OS page cache is **not** flushed;
these are process/model-session-cold observations, not disk-cold results.

Reports include sample counts, median p50, nearest-rank p95, minimum and maximum;
warm timings also split by category. A p95 with fewer than 20 observations carries
a low-sample warning. Default cold p95 is simply the largest of three samples and
must not be presented as a reliable tail-latency estimate. Increase repetitions
on a suitable machine; no arbitrary latency gate is claimed without calibration.
All timed results are checked against the six-result budget and folder scope;
dense/hybrid must return `min(6, in-scope passages)`. Cold query IDs and repetition
numbers must match their actual requests. Provider, device, thread count,
executable hash, source commit/dirty state, model revision, model profile,
text-search profile, extraction profile and fixture hash are recorded.

## Run without models

From the repository root:

```bash
python3 scripts/evaluate-qmd.py validate
PYTHONPATH=scripts python3 -m unittest discover -s scripts/qmd_eval -p 'test_*.py' -v
```

These checks validate labels, metric formulas, cutoffs, duplicate handling,
passage binding, category gating, folder leaks and explicit missing-model failure.
They do **not** measure retrieval quality or validate native inference.

## Run the real baseline

Use a suitable native build environment with dependencies already available:

```bash
cargo build --offline --locked -p fractal-search-core --example evaluate_qmd

python3 scripts/evaluate-qmd.py run \
  --binary target/debug/examples/evaluate_qmd \
  --models /absolute/path/to/existing/verified/models \
  --output /absolute/new/qmd-evaluation-report
```

For CUDA, build with `--features cuda` and pass `--provider cuda --device-id 0`.
Use the existing CUDA library configuration. A CUDA error does not retry on CPU.
Use a release build for representative timings, and record the exact build and
hardware; debug and release results are not directly comparable.

Optional controls: `--threads 6`, `--warmups 1`, `--repeats 3`,
`--cold-repeats 3`, `--timeout 3600` (seconds per native process), and
`--corpus /path/to/reviewed-fixture.json`. The report directory must not already
exist. The adapter accepts a strict request JSON only through this harness; do
not point it at a user library. No operation is resumed from a partly initialized
catalog and no old catalog is migrated.

Outputs:

- `report.json`: complete metrics, per-query and category summaries, latency,
  provenance, limitations and every quality-gate check
- `summary.md`: readable category comparisons, timings and failed gates
- `corpus-manifest.json`, `queries.json`, `passage-judgments.json`: audit inputs
- `raw/`: native manifest, ranked hits, timed samples and stderr logs

Exit codes are **0** for a complete run passing provisional gates, **1** for a
complete run failing quality gates, and **2** for a blocked/incomplete run.
A blocked report has `quality: null` and `latency: null`; partial raw diagnostics
may remain, but no complete quality result is claimed. Invalid CLI arguments and
an existing output directory also fail without replacing prior reports.

## Provisional category gates and next evidence

The committed gates are deliberately visible, versioned **development policy**,
not measured findings or calibrated release thresholds. They require hybrid
Recall@100 = 1 for each category and use these macro MRR@100/nDCG@6 floors for both
document and passage metrics:

| Category | Floor |
| --- | ---: |
| Exact filename | 0.80 |
| Identifier | 0.75 |
| Paraphrase | 0.50 |
| Multilingual | 0.50 |
| Long document | 0.50 |
| Folder filter | 0.75 |

In addition, hybrid nDCG@6 and Recall@100 may trail the stronger single leg in each
category by at most 0.05. Every failure is shown; a good overall average cannot
hide an identifier or folder-filter regression. The minimum-value and
single-leg checks run independently at document and passage levels. Filter leaks
are hard errors at all measured limits.

Four queries per category cannot support a statistically robust quality claim.
Before claiming an improvement or tuning RRF/candidate budgets, expand reviewed
realistic labels, add difficult negatives and multi-relevant queries, reserve a
held-out set, and run on a representative corpus size and hardware. Compare
matching profiles and candidate budgets, inspect per-query regressions, and
report uncertainty. Keep numerical parity and labeled relevance as separate
validation requirements.
