"""Deterministic tests; these do not run inference or assert retrieval quality."""

import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest

from qmd_eval import fixtures, harness, metrics
from qmd_eval.harness import DEFAULT_CORPUS, ROOT, validate_hits, validate_rankings, validate_native_metadata, validate_timed_sample


class MetricTests(unittest.TestCase):
    def test_recall_uses_all_relevant_not_hit_rate(self):
        self.assertEqual(metrics.score(["a"], {"a": 3, "b": 1})["recall@100"], 0.5)

    def test_reciprocal_rank(self):
        result = metrics.score(["x", "y", "a"], {"a": 3})
        self.assertEqual(result["mrr@100"], 1 / 3)
        self.assertEqual(result["ndcg@6"], 1 / math.log2(4))

    def test_graded_ndcg(self):
        actual = metrics.score(["b", "a"], {"a": 3, "b": 1})["ndcg@6"]
        expected = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
        self.assertAlmostEqual(actual, expected)

    def test_ideal_ndcg(self):
        self.assertEqual(metrics.score(["a", "b"], {"a": 3, "b": 1})["ndcg@6"], 1)

    def test_empty_ranking_is_zero(self):
        self.assertEqual(metrics.score([], {"a": 1}), {"recall@100": 0, "mrr@100": 0, "ndcg@6": 0})

    def test_duplicates_cannot_inflate_document_gain(self):
        self.assertEqual(metrics.score(["a", "a", "b"], {"a": 3, "b": 1}),
                         metrics.score(["a", "b"], {"a": 3, "b": 1}))

    def test_document_and_passage_judgments_are_separate(self):
        self.assertEqual(metrics.score(["long"], {"long": 3})["ndcg@6"], 1)
        self.assertEqual(metrics.score(["long#0"], {"long#7": 3})["ndcg@6"], 0)

    def test_cutoffs(self):
        ranking = [f"negative-{i}" for i in range(100)] + ["a"]
        self.assertEqual(metrics.score(ranking, {"a": 3})["recall@100"], 0)
        self.assertEqual(metrics.score(ranking, {"a": 3})["mrr@100"], 0)
        self.assertEqual(metrics.score([str(i) for i in range(6)] + ["a"], {"a": 3})["ndcg@6"], 0)

    def test_invalid_judgments_fail(self):
        for judgments in ({}, {"a": 0}, {"a": -1}, {"a": 4}, {"a": float("nan")}, {"a": True}, {"a": "3"}):
            with self.subTest(judgments=judgments), self.assertRaises(ValueError):
                metrics.score(["a"], judgments)

    def test_macro_average_not_micro_recall(self):
        rows = [metrics.score(["a"], {"a": 3}), metrics.score([], {"a": 3, "b": 3, "c": 3})]
        self.assertEqual(metrics.average(rows)["recall@100"], 0.5)
        with self.assertRaises(ValueError):
            metrics.average([])

    def test_latency_percentile(self):
        result = metrics.latency_summary(list(range(1, 21)))
        self.assertEqual(result["p50_ms"], 10.5)
        self.assertEqual(result["p95_ms"], 19)
        self.assertFalse(result["p95_low_sample_warning"])
        self.assertTrue(metrics.latency_summary([1, 2, 3])["p95_low_sample_warning"])
        self.assertEqual(metrics.latency_summary([1, 2, 3])["p95_ms"], 3)

    def test_bad_latency_fails(self):
        for sample in ([], [-1], [float("nan")], [float("inf")]):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                metrics.latency_summary(sample)

    def test_native_passage_binding(self):
        passages = [{"document_id": "d", "segment_index": i, "passage_id": f"d#{i}", "text": text}
                    for i, text in enumerate(("irrelevant header", "relevant evidence", "overlap relevant evidence"))]
        queries = [{"id": "q", "judgments": [{"document_id": "d", "grade": 3, "evidence": ["relevant evidence"]}]}]
        self.assertEqual(metrics.passage_judgments(queries, passages), {"q": {"d#1": 3, "d#2": 3}})
        queries[0]["judgments"][0]["evidence"] = []
        self.assertEqual(len(metrics.passage_judgments(queries, passages)["q"]), 3)

    def test_absent_or_boundary_split_evidence_fails(self):
        query = [{"id": "q", "judgments": [{"document_id": "d", "grade": 3, "evidence": ["two words"]}]}]
        passages = [{"document_id": "d", "segment_index": i, "passage_id": f"d#{i}", "text": text}
                    for i, text in enumerate(("two", "words"))]
        with self.assertRaisesRegex(ValueError, "absent or crosses"):
            metrics.passage_judgments(query, passages)

    def test_duplicate_native_passage_fails(self):
        passage = {"document_id": "d", "segment_index": 0, "passage_id": "d#0", "text": "abc"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            metrics.passage_judgments([], [passage, passage])

    def test_stronger_single_leg_nonregression_gate(self):
        perfect = {"recall@100": 1.0, "mrr@100": 1.0, "ndcg@6": 1.0}
        category = {mode: {level: perfect.copy() for level in ("document", "passage")}
                    for mode in ("bm25", "dense", "hybrid")}
        category["hybrid"]["passage"]["ndcg@6"] = 0.8
        policy = {"categories": {"example": {"ndcg@6": 0.5}}, "hybrid_max_ndcg_regression": 0.05, "hybrid_max_recall_regression": 0.05}
        result = metrics.quality_gates({"example": category}, policy)
        self.assertFalse(result["passed"])
        failed = [check for check in result["checks"] if not check["passed"]]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["kind"], "single_leg_nonregression")
        self.assertEqual(failed[0]["level"], "passage")


class NativeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.docs = {"positive": {"id": "positive", "folder": "work"},
                     "negative": {"id": "negative", "folder": "work"},
                     "outside": {"id": "outside", "folder": "home"}}
        self.hits = [{"document_id": name, "segment_index": 0, "passage_id": f"{name}#0"}
                     for name in self.docs]
        self.passage_ids = {hit["passage_id"] for hit in self.hits}
        self.query = {"id": "expected-query", "folder": "work"}
        self.fixture = {"materialized_documents": list(self.docs.values()), "queries": [self.query]}

    def native(self, hits, query_id="expected-query"):
        return {"quality_rankings": [{"query_id": query_id, "hits": hits}]}

    def test_dropping_negative_rejected_for_dense_and_hybrid(self):
        for mode in ("dense", "hybrid"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "omitted in-scope"):
                validate_rankings(self.fixture, self.native(self.hits[:1]), self.passage_ids, mode)

    def test_dense_and_hybrid_require_exact_folder_scoped_set(self):
        for mode in ("dense", "hybrid"):
            result = validate_rankings(self.fixture, self.native(self.hits[:2]), self.passage_ids, mode)
            self.assertEqual(len(result["expected-query"]), 2)
            with self.assertRaisesRegex(ValueError, "out-of-scope"):
                validate_rankings(self.fixture, self.native(self.hits), self.passage_ids, mode)

    def test_unfiltered_quality_requires_all_passages(self):
        self.query.pop("folder")
        with self.assertRaisesRegex(ValueError, "omitted in-scope"):
            validate_rankings(self.fixture, self.native(self.hits[:2]), self.passage_ids, "dense")
        validate_rankings(self.fixture, self.native(self.hits), self.passage_ids, "hybrid")

    def test_bm25_sparse_and_empty_rankings_allowed(self):
        for hits in ([], self.hits[:1], self.hits[:2]):
            validate_rankings(self.fixture, self.native(hits), self.passage_ids, "bm25")

    def test_quality_wrong_query_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown or duplicate"):
            validate_rankings(self.fixture, self.native(self.hits[:2], "different-query"), self.passage_ids, "hybrid")

    def test_cold_wrong_query_and_repeat_rejected(self):
        for field, value in (("query_id", "different-query"), ("repeat", 1)):
            sample = {"query_id": "expected-query", "repeat": 0, "hits": self.hits[:2], field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "disagrees with request"):
                validate_timed_sample(sample, self.query, self.docs, self.passage_ids, "hybrid", 0)

    def test_oversized_timed_and_cold_hits_rejected_for_every_mode(self):
        docs = {f"doc-{i}": {"folder": "work"} for i in range(7)}
        hits = [{"document_id": name, "segment_index": 0, "passage_id": f"{name}#0"} for name in docs]
        passage_ids = {hit["passage_id"] for hit in hits}
        for mode in ("bm25", "dense", "hybrid"):
            for repeat in (0, 2):
                sample = {"query_id": "expected-query", "repeat": repeat, "hits": hits}
                with self.subTest(mode=mode, repeat=repeat), self.assertRaisesRegex(ValueError, "six-result budget"):
                    validate_timed_sample(sample, self.query, docs, passage_ids, mode, repeat)

    def test_truncated_dense_timed_results_rejected_bm25_sparse_allowed(self):
        sample = {"query_id": "expected-query", "repeat": 0, "hits": self.hits[:1]}
        for mode in ("dense", "hybrid"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "omitted expected results"):
                validate_timed_sample(sample, self.query, self.docs, self.passage_ids, mode, 0)
        validate_timed_sample(sample, self.query, self.docs, self.passage_ids, "bm25", 0)
        sample["hits"] = self.hits[:2]
        validate_timed_sample(sample, self.query, self.docs, self.passage_ids, "hybrid", 0)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.load(DEFAULT_CORPUS)

    def test_all_categories_and_distractors(self):
        self.assertEqual(len(self.fixture["materialized_documents"]), 202)
        self.assertEqual(len(self.fixture["queries"]), 24)
        for category in fixtures.CATEGORIES:
            self.assertEqual(sum(query["category"] == category for query in self.fixture["queries"]), 4)
        docs = {doc["id"]: doc for doc in self.fixture["materialized_documents"]}
        self.assertGreater(len(docs["long-ops"]["text"].encode()), 20_000)
        self.assertEqual(docs["work-release"]["filename"], docs["home-release"]["filename"])

    def test_materialization_is_deterministic_and_confined(self):
        with tempfile.TemporaryDirectory() as temp:
            first = fixtures.materialize(self.fixture, pathlib.Path(temp) / "a")
            second = fixtures.materialize(self.fixture, pathlib.Path(temp) / "b")
            self.assertEqual(first, second)
            self.assertEqual(len(list((pathlib.Path(temp) / "a").glob("*/*"))), 202)
            with self.assertRaises(FileExistsError):
                fixtures.materialize(self.fixture, pathlib.Path(temp) / "a")

    def test_path_traversal_rejected(self):
        for name in ("..", ".", "", "../escape", "/absolute", "nested/file", "a\\b", "a\0b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fixtures.component(name)

    def test_folder_leak_rejected(self):
        docs = {doc["id"]: doc for doc in self.fixture["materialized_documents"]}
        hit = {"document_id": "home-release", "passage_id": "home-release#0", "segment_index": 0}
        with self.assertRaisesRegex(ValueError, "out-of-scope"):
            validate_hits([hit], {"folder": "work"}, docs, {"home-release#0"})

    def test_unknown_passage_rejected(self):
        docs = {doc["id"]: doc for doc in self.fixture["materialized_documents"]}
        hit = {"document_id": "launch", "passage_id": "launch#9", "segment_index": 9}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_hits([hit], {}, docs, {"launch#0"})

    def test_missing_query_fails_not_zero_score(self):
        with self.assertRaisesRegex(ValueError, "omitted"):
            validate_rankings(self.fixture, {"quality_rankings": []}, set(), "hybrid")

    def test_all_202_document_fixture_scopes_require_complete_rankings(self):
        docs = self.fixture["materialized_documents"]
        passage_ids = {f"{doc['id']}#0" for doc in docs}
        rankings = [{"query_id": query["id"], "hits": [
            {"document_id": doc["id"], "segment_index": 0, "passage_id": f"{doc['id']}#0"}
            for doc in docs if not query.get("folder") or doc["folder"] == query["folder"]]}
            for query in self.fixture["queries"]]
        for mode in ("dense", "hybrid"):
            validate_rankings(self.fixture, {"quality_rankings": rankings}, passage_ids, mode)
            for query_index, query in enumerate(self.fixture["queries"]):
                relevant = {judgment["document_id"] for judgment in query["judgments"]}
                dropped = next(hit for hit in rankings[query_index]["hits"] if hit["document_id"] not in relevant)
                incomplete = list(rankings)
                incomplete[query_index] = rankings[query_index] | {
                    "hits": [hit for hit in rankings[query_index]["hits"] if hit is not dropped]}
                with self.subTest(mode=mode, query=query["id"]), self.assertRaisesRegex(ValueError, "omitted in-scope"):
                    validate_rankings(self.fixture, {"quality_rankings": incomplete}, passage_ids, mode)

    def test_native_provenance_mismatch_rejected(self):
        indexed = {"model_profile": harness.MODEL_PROFILE, "embedding_dimension": harness.EMBEDDING_DIM, "text_search_profile": "lexical-contract", "extraction_profile": "chunk-contract", "provider": "cpu", "threads": 6}
        result = indexed | {"operation": "search", "mode": "hybrid", "rank_limit": 1000}
        validate_native_metadata(indexed, result, "search", "hybrid")
        for field in ("model_profile", "embedding_dimension", "text_search_profile", "extraction_profile"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "provenance changed"):
                validate_native_metadata(indexed, result | {field: "other"}, "search", "hybrid")
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            validate_native_metadata(indexed, {key: value for key, value in result.items() if key != "extraction_profile"}, "search", "hybrid")
        with self.assertRaisesRegex(ValueError, "budget disagrees"):
            validate_native_metadata(indexed, result | {"rank_limit": 6}, "search", "hybrid")

    def test_old_or_missing_indexed_contract_is_rejected(self):
        current = {"model_profile": harness.MODEL_PROFILE, "embedding_dimension": 512}
        harness.validate_indexed_contract(current)
        for invalid in ({}, current | {"embedding_dimension": 768},
                        current | {"model_profile": "embeddinggemma2-daa72c51-fp32-multimodal-v1"}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "indexed 512"):
                harness.validate_indexed_contract(invalid)

    def test_missing_models_produces_blocked_report_without_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp) / "report"
            models = pathlib.Path(temp) / "empty-models"
            models.mkdir()
            process = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate-qmd.py"), "run",
                                      "--binary", sys.executable, "--models", str(models), "--output", str(output)],
                                     text=True, capture_output=True)
            self.assertEqual(process.returncode, 2, process.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertIsNone(report["quality"])
            self.assertIsNone(report["latency"])
            self.assertIn("No models were downloaded", report["error"])
            self.assertTrue((output / "summary.md").is_file())
            self.assertEqual(list((output / "raw").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
