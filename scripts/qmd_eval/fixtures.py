"""Validate and materialize synthetic documents into a caller-owned temp root."""

import hashlib
import json
import math
import pathlib

CATEGORIES = {"exact_filename", "identifier", "paraphrase", "multilingual", "long_document", "folder_filter"}


def component(value):
    if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value or "\0" in value:
        raise ValueError(f"Invalid fixture path component: {value!r}")


def load(path):
    raw = pathlib.Path(path).read_bytes()
    fixture = json.loads(raw)
    if fixture["schema_version"] != 1:
        raise ValueError("Unsupported corpus schema version")
    documents = []
    for source in fixture["documents"]:
        if ("text" in source) == ("parts" in source):
            raise ValueError("Document needs exactly one text or parts field")
        text = source.get("text")
        if text is None:
            parts = source["parts"]
            if any(type(p["repeat"]) is not int or not 1 <= p["repeat"] <= 1000 for p in parts):
                raise ValueError("Invalid long-document repeat count")
            text = "".join(part["text"] * part["repeat"] for part in parts)
        documents.append({key: source[key] for key in ("id", "folder", "filename")} | {"text": text})
    noise = fixture["distractors"]
    if type(noise["count"]) is not int or not 0 <= noise["count"] <= 5000:
        raise ValueError("Invalid distractor count")
    if not noise["folders"] or not noise["topics"]:
        raise ValueError("Distractors need folders and topics")
    for index in range(noise["count"]):
        topic = noise["topics"][index % len(noise["topics"])]
        documents.append({"id": f"negative-{index:04d}",
                          "folder": noise["folders"][index % len(noise["folders"])],
                          "filename": f"notice-{index:04d}.txt",
                          "text": f"Community notice {index:04d}: {topic}. This ordinary bulletin records the location, attendance and supplies for activity {index:04d}. Check the local notice board for further details.\n"})
    ids, paths = set(), set()
    for doc in documents:
        component(doc["folder"])
        component(doc["filename"])
        if not doc["id"] or "#" in doc["id"] or doc["id"] in ids:
            raise ValueError("Duplicate, empty or ambiguous document ID")
        ids.add(doc["id"])
        relative = (doc["folder"], doc["filename"])
        if relative in paths:
            raise ValueError("Duplicate fixture path")
        paths.add(relative)
        if pathlib.Path(doc["filename"]).suffix.lower() not in (".txt", ".md", ".markdown"):
            raise ValueError("Unsupported fixture extension")
        if not doc["text"].strip() or "\0" in doc["text"] or len(doc["text"].encode()) > 2 * 1024 * 1024:
            raise ValueError("Invalid document content")
    by_id = {doc["id"]: doc for doc in documents}
    query_ids = set()
    categories = set()
    for query in fixture["queries"]:
        if not query["id"] or query["id"] in query_ids:
            raise ValueError("Duplicate or empty query ID")
        query_ids.add(query["id"])
        categories.add(query["category"])
        if not query["text"].strip() or "\0" in query["text"] or len(query["text"].encode()) > 4096:
            raise ValueError("Invalid query text")
        if not query["judgments"]:
            raise ValueError("Missing query judgments")
        judged = set()
        for judgment in query["judgments"]:
            doc = by_id[judgment["document_id"]]
            if doc["id"] in judged:
                raise ValueError("Duplicate document judgment")
            judged.add(doc["id"])
            if type(judgment["grade"]) is not int or not 1 <= judgment["grade"] <= 3:
                raise ValueError("Positive grades must be integers in 1..3")
            if query.get("folder") and query["folder"] != doc["folder"]:
                raise ValueError("Relevant document is outside query folder")
            if any(not text or text not in doc["text"] for text in judgment["evidence"]):
                raise ValueError("Passage evidence not found in document")
    if categories != CATEGORIES or set(fixture["quality_gates"]["categories"]) != CATEGORIES:
        raise ValueError("All six evaluation categories and gates are required")
    gates = fixture["quality_gates"]
    thresholds = [gates["hybrid_max_ndcg_regression"], gates["hybrid_max_recall_regression"]]
    for floors in gates["categories"].values():
        if set(floors) != {"recall@100", "mrr@100", "ndcg@6"}:
            raise ValueError("Every category needs Recall, MRR and nDCG floors")
        thresholds.extend(floors.values())
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Quality gate thresholds must be finite numbers in 0..1")
    fixture["materialized_documents"] = documents
    fixture["fixture_sha256"] = hashlib.sha256(raw).hexdigest()
    return fixture


def materialize(fixture, root):
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=False)
    manifest = []
    for doc in fixture["materialized_documents"]:
        directory = root / doc["folder"]
        directory.mkdir(exist_ok=True)
        content = doc["text"].encode("utf-8")
        (directory / doc["filename"]).write_bytes(content)
        manifest.append({key: doc[key] for key in ("id", "folder", "filename")} |
                        {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    return manifest
