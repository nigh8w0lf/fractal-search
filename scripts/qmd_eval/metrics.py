"""Conventional IR metrics over canonical identities, without external packages."""

import math
import statistics


def unique_ranking(ranking):
    """Keep the first occurrence; repeated passages must not inflate document gain."""
    return list(dict.fromkeys(ranking))


def score(ranking, judgments):
    """Binary Recall/MRR@100; graded nDCG@6 with gain 2**grade - 1.

    Unjudged items have zero gain. At least one positive judgment is mandatory.
    Duplicated identities are collapsed before computing ranks.
    """
    if any(type(grade) is not int or not 0 <= grade <= 3 for grade in judgments.values()):
        raise ValueError("Judgment grades must be integers in 0..3")
    if not judgments or not any(grade > 0 for grade in judgments.values()):
        raise ValueError("A query must have at least one positive judgment")
    ranking = unique_ranking(ranking)
    relevant = {item for item, grade in judgments.items() if grade > 0}
    recall = len(set(ranking[:100]) & relevant) / len(relevant)
    reciprocal_rank = next((1 / rank for rank, item in enumerate(ranking[:100], 1)
                            if item in relevant), 0.0)

    def dcg(grades):
        return sum((2 ** grade - 1) / math.log2(rank + 1)
                   for rank, grade in enumerate(grades, 1))

    ideal = dcg(sorted(judgments.values(), reverse=True)[:6])
    return {"recall@100": recall, "mrr@100": reciprocal_rank,
            "ndcg@6": dcg([judgments.get(item, 0) for item in ranking[:6]]) / ideal}


def average(rows):
    if not rows:
        raise ValueError("Cannot average an empty query category")
    return {metric: statistics.fmean(row[metric] for row in rows)
            for metric in ("recall@100", "mrr@100", "ndcg@6")}


def latency_summary(samples):
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError("Latency samples must be finite, nonnegative and nonempty")
    ordered = sorted(samples)
    return {"samples": len(ordered), "p50_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "min_ms": ordered[0], "max_ms": ordered[-1],
            "p95_method": "nearest rank", "p95_low_sample_warning": len(ordered) < 20}


def passage_judgments(queries, passages):
    """Bind evidence to the native chunker, never a Python reimplementation.

    An empty evidence list explicitly judges every passage in that document
    relevant (filename lookup). Otherwise each evidence string must appear in
    at least one complete native passage; overlap can create multiple positives.
    """
    by_document = {}
    seen = set()
    for passage in passages:
        if passage["passage_id"] in seen:
            raise ValueError("Duplicate native passage identity")
        seen.add(passage["passage_id"])
        expected = f"{passage['document_id']}#{passage['segment_index']}"
        if passage["passage_id"] != expected:
            raise ValueError("Native passage identity disagrees with canonical mapping")
        by_document.setdefault(passage["document_id"], []).append(passage)
    output = {}
    for query in queries:
        grades = {}
        for judgment in query["judgments"]:
            candidates = by_document.get(judgment["document_id"], [])
            if not candidates:
                raise ValueError(f"No indexed passages for {judgment['document_id']}")
            evidence = judgment["evidence"]
            for text in evidence:
                if not any(text in passage["text"] for passage in candidates):
                    raise ValueError(f"Evidence is absent or crosses every chunk boundary: {query['id']}: {text}")
            for passage in candidates:
                if not evidence or any(text in passage["text"] for text in evidence):
                    grades[passage["passage_id"]] = judgment["grade"]
        if not grades:
            raise ValueError(f"Query has no passage judgments: {query['id']}")
        output[query["id"]] = grades
    return output


def quality_gates(categories, policy):
    checks = []
    for category, floors in policy["categories"].items():
        for level in ("document", "passage"):
            hybrid = categories[category]["hybrid"][level]
            for metric, floor in floors.items():
                actual = hybrid[metric]
                checks.append({"category": category, "level": level, "kind": "absolute",
                               "metric": metric, "threshold": floor, "actual": actual,
                               "passed": actual + 1e-12 >= floor})
            for metric, allowance in (("ndcg@6", policy["hybrid_max_ndcg_regression"]),
                                      ("recall@100", policy["hybrid_max_recall_regression"])):
                best = max(categories[category][mode][level][metric] for mode in ("bm25", "dense"))
                checks.append({"category": category, "level": level, "kind": "single_leg_nonregression",
                               "metric": metric, "threshold": best - allowance,
                               "actual": hybrid[metric],
                               "passed": hybrid[metric] + allowance + 1e-12 >= best})
    return {"passed": all(check["passed"] for check in checks), "checks": checks,
            "policy_status": "provisional, uncalibrated synthetic development gates"}
