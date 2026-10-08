"""Orchestrate the real Rust adapter and score its canonical rankings."""

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import tempfile
import time

from . import fixtures, metrics

MODES = ("bm25", "dense", "hybrid")
ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = pathlib.Path(__file__).with_name("corpus.json")


def write_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_rankings(fixture, native, passage_ids, mode):
    expected = {query["id"]: query for query in fixture["queries"]}
    docs = {doc["id"]: doc for doc in fixture["materialized_documents"]}
    rankings = {}
    for row in native["quality_rankings"]:
        query_id = row["query_id"]
        if query_id not in expected or query_id in rankings:
            raise ValueError("Unknown or duplicate query ranking")
        validate_hits(row["hits"], expected[query_id], docs, passage_ids)
        if mode in ("dense", "hybrid"):
            required = in_scope_passage_ids(expected[query_id], docs, passage_ids)
            actual = {hit["passage_id"] for hit in row["hits"]}
            if len(row["hits"]) != len(required) or actual != required:
                raise ValueError("Native dense/hybrid quality ranking omitted in-scope passages")
        rankings[query_id] = row["hits"]
    if set(rankings) != set(expected):
        raise ValueError("Native output omitted one or more queries")
    return rankings


def validate_hits(hits, query, docs, passage_ids):
    seen = set()
    for hit in hits:
        if hit["document_id"] not in docs:
            raise ValueError("Unknown canonical result document")
        if hit["passage_id"] != f"{hit['document_id']}#{hit['segment_index']}" or hit["passage_id"] not in passage_ids:
            raise ValueError("Unknown or inconsistent result passage")
        if hit["passage_id"] in seen:
            raise ValueError("Duplicate result passage")
        seen.add(hit["passage_id"])
        if query.get("folder") and docs[hit["document_id"]]["folder"] != query["folder"]:
            raise ValueError("Folder prefilter leaked an out-of-scope result")


def in_scope_passage_ids(query, docs, passage_ids):
    return {passage_id for passage_id in passage_ids
            if not query.get("folder") or docs[passage_id.rsplit("#", 1)[0]]["folder"] == query["folder"]}


def validate_timed_sample(sample, query, docs, passage_ids, mode, repeat):
    if sample["query_id"] != query["id"] or sample["repeat"] != repeat:
        raise ValueError("Native timed sample query ID or repetition disagrees with request")
    if len(sample["hits"]) > 6:
        raise ValueError("Native timed sample exceeded the six-result budget")
    validate_hits(sample["hits"], query, docs, passage_ids)
    if mode in ("dense", "hybrid"):
        required_count = min(6, len(in_scope_passage_ids(query, docs, passage_ids)))
        if len(sample["hits"]) != required_count:
            raise ValueError("Native dense/hybrid timed sample omitted expected results")


def evaluate(fixture, indexed, runs):
    passage_ids = {p["passage_id"] for p in indexed["passages"]}
    if len(passage_ids) != indexed["passage_count"]:
        raise ValueError("Native passage manifest count mismatch")
    if {p["document_id"] for p in indexed["passages"]} != {doc["id"] for doc in fixture["materialized_documents"]}:
        raise ValueError("Native passage manifest omitted or added a document")
    passage_qrels = metrics.passage_judgments(fixture["queries"], indexed["passages"])
    results = {mode: validate_rankings(fixture, runs[mode], passage_ids, mode) for mode in MODES}
    by_query = []
    for query in fixture["queries"]:
        row = {"query_id": query["id"], "category": query["category"], "modes": {}}
        for mode in MODES:
            hits = results[mode][query["id"]]
            row["modes"][mode] = {
                "document": metrics.score([hit["document_id"] for hit in hits],
                                          {j["document_id"]: j["grade"] for j in query["judgments"]}),
                "passage": metrics.score([hit["passage_id"] for hit in hits], passage_qrels[query["id"]]),
                "returned_passages": len(hits),
                "returned_documents": len({hit["document_id"] for hit in hits}),
            }
        by_query.append(row)
    categories = {}
    for category in sorted(fixtures.CATEGORIES):
        rows = [row for row in by_query if row["category"] == category]
        categories[category] = {"query_count": len(rows)}
        for mode in MODES:
            categories[category][mode] = {level: metrics.average([row["modes"][mode][level] for row in rows])
                                          for level in ("document", "passage")}
    overall = {mode: {level: metrics.average([row["modes"][mode][level] for row in by_query])
                      for level in ("document", "passage")} for mode in MODES}
    return {"overall": overall, "categories": categories, "queries": by_query,
            "gates": metrics.quality_gates(categories, fixture["quality_gates"]),
            "passage_judgments": passage_qrels}


def invoke(binary, request, work, output, name, timeout):
    path = work / f"{name}.request.json"
    write_json(path, request)
    started = time.perf_counter()
    completed = subprocess.run([str(binary), str(path)], text=True, capture_output=True, timeout=timeout)
    wall_ms = (time.perf_counter() - started) * 1000
    (output / f"{name}.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Native {name} failed with exit {completed.returncode}: {completed.stderr[-4000:]}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Native {name} returned invalid JSON; stdout is reserved for one report") from error
    result["process_wall_ms"] = wall_ms
    write_json(output / f"{name}.json", result)
    return result


def validate_native_metadata(indexed, result, operation, mode):
    if result.get("operation") != operation or result.get("mode") != mode or result.get("rank_limit") != 1000:
        raise ValueError("Native operation, retrieval mode or result budget disagrees with request")
    for key in ("model_profile", "text_search_profile", "extraction_profile", "provider", "threads"):
        if result.get(key) != indexed[key]:
            raise ValueError(f"Native evaluation provenance changed between indexing and queries: {key}")


def check_assets(directory):
    """Cheap readiness check; NativeRuntime verifies every pinned hash itself."""
    manifest_path = ROOT / "crates/fractal-search-core/models/embeddinggemma2.json"
    manifest = json.loads(manifest_path.read_text())
    missing, wrong = [], []
    for asset in manifest["files"]:
        path = directory / asset["path"]
        if not path.is_file():
            missing.append(asset["path"])
        elif path.stat().st_size != asset["bytes"]:
            wrong.append(asset["path"])
    if missing or wrong:
        raise RuntimeError(f"Existing pinned model cache is incomplete (missing={missing}, wrong_size={wrong}). No models were downloaded; supply an already verified cache.")
    return {"repository": manifest["repository"], "revision": manifest["revision"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "verification": "size preflight here; full hash verification in NativeRuntime"}


def source_identity():
    result = {}
    for key, command in (("commit", ["git", "rev-parse", "HEAD"]),
                         ("working_tree", ["git", "status", "--short"])):
        try:
            process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=10)
            result[key] = process.stdout.strip() if process.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            result[key] = "unavailable"
    return result


def summarize(report):
    lines = ["# QMD labeled retrieval evaluation", "", f"Status: **{report['status']}**", "",
             "Synthetic development corpus only. These results do not establish real-user quality or production-scale performance.", ""]
    if report.get("error"):
        lines += ["## Blocker", "", report["error"], "", "No complete quality result or passing quality gate is claimed.", ""]
    if report.get("quality"):
        lines += ["## Per-category macro averages", "", "Document ranks collapse passages by canonical document ID before applying cutoffs.", "",
                  "| Category | Mode | Doc Recall@100 | Doc MRR@100 | Doc nDCG@6 | Passage nDCG@6 |", "|---|---|---:|---:|---:|---:|"]
        for category, values in report["quality"]["categories"].items():
            for mode in MODES:
                doc = values[mode]["document"]
                passage = values[mode]["passage"]
                lines.append(f"| {category} | {mode} | {doc['recall@100']:.4f} | {doc['mrr@100']:.4f} | {doc['ndcg@6']:.4f} | {passage['ndcg@6']:.4f} |")
        lines += ["", "## Latency at production result limit 6", "",
                  "| Mode | Warm p50 ms | Warm p95 ms | Fresh-process p50 ms | Fresh-process p95 ms |", "|---|---:|---:|---:|---:|"]
        for mode, timing in report["latency"].items():
            warm, cold = timing["warm_query"], timing["cold_process_wall"]
            lines.append(f"| {mode} | {warm['p50_ms']:.2f} | {warm['p95_ms']:.2f} | {cold['p50_ms']:.2f} | {cold['p95_ms']:.2f} |")
        failed = [check for check in report["quality"]["gates"]["checks"] if not check["passed"]]
        lines += ["", f"Provisional quality gates: {'PASS' if not failed else 'FAIL'} ({len(failed)} failed checks)", ""]
        for check in failed:
            lines.append(f"- {check['category']} / {check['level']} / {check['kind']} / {check['metric']}: {check['actual']:.4f} < {check['threshold']:.4f}")
    lines += ["", "## Measurement limits", ""] + [f"- {warning}" for warning in report["limitations"]]
    return "\n".join(lines) + "\n"


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_output = output / "raw"
    raw_output.mkdir()
    report = {
        "schema_version": 1, "status": "blocked", "quality": None, "latency": None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "source": source_identity(), "provider": args.provider,
                        "device_id": args.device_id, "threads": args.threads},
        "limitations": [
            "Hand-authored synthetic development labels: 24 queries, four per category, not representative or held out. Multilingual covers four same-language queries, not cross-language retrieval.",
            "Quality uses 1000 native passage results and RRF candidates per leg (the production formula reaches its 1000 cap). Document ranking collapses duplicates before metric cutoffs; passage ranking stays separate.",
            "Production-limit-6 latency uses 100 hybrid candidates per leg and includes query embedding plus native search. It does not include desktop IPC, UI or rendering.",
            "Cold means a fresh process/catalog/model session, including model verification/load. The OS page cache is not cleared; these are not disk-cold measurements. Cold p95 with fewer than 20 samples is descriptive only.",
            "Warm quality passes and warmups precede timed warm samples. No latency threshold is calibrated; report timings rather than claiming performance acceptance.",
            "Opaque asset-ID tie-breaking can change tied ranks after rebuilding a temporary catalog. Canonical qrels and deterministic metric tests do not eliminate native score ties.",
            "No LLM expansion, reranking, alternative tokenizer or substitute embeddings. Missing dependencies/models block this run rather than fabricate scores.",
        ],
    }
    try:
        fixture = fixtures.load(args.corpus)
        binary = args.binary.resolve(strict=True)
        if not binary.is_file():
            raise ValueError("--binary must name the built evaluate_qmd example executable")
        models = args.models.resolve(strict=True)
        report["model"] = check_assets(models)
        report["environment"]["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        report["fixture"] = {"name": fixture["name"], "sha256": fixture["fixture_sha256"],
                             "documents": len(fixture["materialized_documents"]), "queries": len(fixture["queries"])}
        with tempfile.TemporaryDirectory(prefix="fractal-qmd-eval-") as temporary:
            work = pathlib.Path(temporary)
            root = work / "corpus"
            manifest = fixtures.materialize(fixture, root)
            write_json(output / "corpus-manifest.json", manifest)
            write_json(output / "queries.json", fixture["queries"])
            base = {"corpus_root": str(root), "data_dir": str(work / "catalog"), "model_dir": str(models),
                    "documents": [{key: doc[key] for key in ("id", "folder", "filename")} for doc in manifest],
                    "queries": [{key: query[key] for key in ("id", "text", "folder") if key in query} for query in fixture["queries"]],
                    "mode": "hybrid", "provider": args.provider, "device_id": args.device_id,
                    "threads": args.threads, "rank_limit": 1000, "repeats": args.repeats, "warmups": args.warmups}
            indexed = invoke(binary, base | {"operation": "index"}, work, raw_output, "index", args.timeout)
            if indexed["documents"] != len(manifest):
                raise ValueError("Native indexed document count mismatch")
            report["fixture"]["passages"] = indexed["passage_count"]
            report["index_ms"] = indexed["index_ms"]
            report["model"]["profile"] = indexed["model_profile"]
            report["text_search_profile"] = indexed["text_search_profile"]
            report["extraction_profile"] = indexed["extraction_profile"]
            # Bound the corpus so document@100 never silently becomes metrics
            # over an unknown, truncated top-1000 passage ranking.
            if indexed["passage_count"] > 1000:
                raise ValueError("This baseline requires <=1000 passages; split the fixture or define a larger-corpus evaluation protocol explicitly")
            runs, cold_runs = {}, {}
            expected_cold_queries = [base["queries"][(repeat * len(base["queries"]) // args.cold_repeats) % len(base["queries"])]
                                     for repeat in range(args.cold_repeats)]
            for mode in MODES:
                print(f"[QMD] {mode}: native quality and warm timings", file=sys.stderr)
                runs[mode] = invoke(binary, base | {"operation": "search", "mode": mode}, work, raw_output, f"warm-{mode}", args.timeout)
                cold_runs[mode] = []
                for repeat, query in enumerate(expected_cold_queries):
                    cold_runs[mode].append(invoke(binary, base | {"operation": "cold", "mode": mode, "queries": [query]},
                                                 work, raw_output, f"cold-{mode}-{repeat}", args.timeout))
            for mode in MODES:
                validate_native_metadata(indexed, runs[mode], "search", mode)
                for cold in cold_runs[mode]:
                    validate_native_metadata(indexed, cold, "cold", mode)
            quality = evaluate(fixture, indexed, runs)
            by_id = {query["id"]: query for query in fixture["queries"]}
            docs = {doc["id"]: doc for doc in fixture["materialized_documents"]}
            passage_ids = {passage["passage_id"] for passage in indexed["passages"]}
            latency = {}
            for mode in MODES:
                samples = runs[mode]["samples"]
                expected = {(query["id"], repeat) for query in fixture["queries"] for repeat in range(args.repeats)}
                if len(samples) != len(expected) or {(row["query_id"], row["repeat"]) for row in samples} != expected:
                    raise ValueError("Native warm samples missing or duplicated")
                for sample in samples:
                    validate_timed_sample(sample, by_id[sample["query_id"]], docs, passage_ids, mode, sample["repeat"])
                if len(cold_runs[mode]) != len(expected_cold_queries):
                    raise ValueError("Native cold runs missing or duplicated")
                for cold, query in zip(cold_runs[mode], expected_cold_queries):
                    if len(cold["samples"]) != 1:
                        raise ValueError("Native cold process must return exactly one sample")
                    validate_timed_sample(cold["samples"][0], query, docs, passage_ids, mode, 0)
                latency[mode] = {
                    "warm_query": metrics.latency_summary([row["elapsed_ms"] for row in samples]),
                    "warm_by_category": {category: metrics.latency_summary([row["elapsed_ms"] for row in samples
                                            if by_id[row["query_id"]]["category"] == category])
                                         for category in sorted(fixtures.CATEGORIES)},
                    "cold_process_wall": metrics.latency_summary([row["process_wall_ms"] for row in cold_runs[mode]]),
                    "cold_first_result": metrics.latency_summary([row["samples"][0]["first_result_ms"] for row in cold_runs[mode]]),
                    "cold_queries": [row["samples"][0]["query_id"] for row in cold_runs[mode]],
                    "production_passage_ndcg@6": sum(metrics.score([hit["passage_id"] for hit in row["hits"]],
                        quality["passage_judgments"][row["query_id"]])["ndcg@6"] for row in samples) / len(samples),
                    "production_mean_unique_documents@6": sum(len({hit["document_id"] for hit in row["hits"]}) for row in samples) / len(samples),
                }
            report["quality"], report["latency"] = quality, latency
            report["status"] = "passed" if quality["gates"]["passed"] else "quality_gate_failed"
            write_json(output / "passage-judgments.json", quality["passage_judgments"])
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(f"[QMD] BLOCKED: {report['error']}", file=sys.stderr)
    write_json(output / "report.json", report)
    (output / "summary.md").write_text(summarize(report))
    print(f"{report['status']}: {output / 'report.json'}")
    return 0 if report["status"] == "passed" else 1 if report["status"] == "quality_gate_failed" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check corpus labels without models or Rust")
    validate.add_argument("--corpus", type=pathlib.Path, default=DEFAULT_CORPUS)
    execute = commands.add_parser("run", help="Run real native BM25/dense/hybrid evaluation")
    execute.add_argument("--corpus", type=pathlib.Path, default=DEFAULT_CORPUS)
    execute.add_argument("--binary", type=pathlib.Path, required=True, help="Built target/{debug,release}/examples/evaluate_qmd")
    execute.add_argument("--models", type=pathlib.Path, required=True, help="Existing pinned model cache; never downloaded")
    execute.add_argument("--output", type=pathlib.Path, required=True, help="New output directory; existing directories are rejected")
    execute.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    execute.add_argument("--device-id", type=int)
    execute.add_argument("--threads", type=int, default=6)
    execute.add_argument("--repeats", type=int, default=3)
    execute.add_argument("--warmups", type=int, default=1)
    execute.add_argument("--cold-repeats", type=int, default=3)
    execute.add_argument("--timeout", type=float, default=3600, help="Maximum seconds per native process")
    args = parser.parse_args(argv)
    if args.command == "validate":
        fixture = fixtures.load(args.corpus)
        print(json.dumps({"status": "fixture_valid", "documents": len(fixture["materialized_documents"]),
                          "queries": len(fixture["queries"]), "categories": sorted(fixtures.CATEGORIES),
                          "fixture_sha256": fixture["fixture_sha256"], "measured_quality": None}, indent=2))
        return 0
    if not 1 <= args.threads <= 64 or not 1 <= args.repeats <= 100 or not 1 <= args.cold_repeats <= 100 or not 1 <= args.warmups <= 10 or args.timeout <= 0:
        parser.error("threads: 1..64; repeats/cold-repeats: 1..100; warmups: 1..10; timeout must be positive")
    if args.provider == "cpu" and args.device_id is not None:
        parser.error("--device-id requires --provider cuda")
    return run(args)
