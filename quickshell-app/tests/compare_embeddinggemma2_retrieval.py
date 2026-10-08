#!/usr/bin/env python3
"""Compare real service scores with pinned EmbeddingGemma 2 ONNX fixtures.

This measures implementation agreement on synthetic fixtures, not semantic recall.
Generate indexed-512 fixtures with scripts/prepare-embeddinggemma2-fixtures.py.
--native-fixtures explicitly projects full-768 references for indexed comparison;
it never changes the production catalog or query dimension.
"""

import argparse
import json
import math
import pathlib
import shutil
import sys
import tempfile

from smoke_backend import Service

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from embeddinggemma2_contract import (
    EMBEDDING_DIM, MODEL_PROFILE, project_indexed_vector, validate_fixture_environment,
    validate_indexed_vector,
)

MODEL_REPOSITORY = "onnx-community/embeddinggemma-2-ONNX"
MODEL_REVISION = "daa72c51243991dfcaf9f9137d2c573d8f7790c0"


def cosine(a, b):
    validate_indexed_vector(a)
    validate_indexed_vector(b)
    norm = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b)) / norm


def compare(binary, models, fixtures_path, native_fixtures=False):
    environment = json.loads((fixtures_path.parent / "environment.json").read_text())
    if (environment.get("model_repository"), environment.get("model_revision")) != (MODEL_REPOSITORY, MODEL_REVISION):
        raise ValueError("Regenerate fixtures for the pinned EmbeddingGemma 2 export")
    validate_fixture_environment(environment, native=native_fixtures)
    fixtures = json.loads(fixtures_path.read_text())
    if native_fixtures:
        for fixture in fixtures:
            fixture["reference"] = [project_indexed_vector(vector) for vector in fixture["reference"]]
    images = [f for f in fixtures if f["kind"] == "image" and len(f["image_files"]) == 1]
    queries = [f for f in fixtures if f["kind"] == "text" and f["text"].strip()]
    if not 5 <= len(images) <= 100 or not queries:
        raise ValueError("Need 5–100 single-image fixtures and nonempty text fixtures")
    if any(len(query["text"].encode("utf-8")) > 4096 for query in queries):
        raise ValueError("Fixture query exceeds the application byte limit")
    for fixture in images + queries:
        if len(fixture["reference"]) != 1:
            raise ValueError(f"Expected one reference for {fixture['name']}")
        validate_indexed_vector(fixture["reference"][0])
    score_tolerance = 1e-4
    with tempfile.TemporaryDirectory(prefix="fractal-search-parity-", dir="/tmp") as temporary:
        root = pathlib.Path(temporary)
        photos = root / "photos"
        photos.mkdir()
        references = {}
        for fixture in images:
            if len(fixture["reference"]) != 1:
                raise ValueError(f"Expected one reference for {fixture['name']}")
            source = (fixtures_path.parent / fixture["image_files"][0]).resolve(strict=True)
            if not source.is_relative_to(fixtures_path.parent.resolve()):
                raise ValueError("Image fixture escapes its directory")
            if pathlib.Path(fixture["name"]).name != fixture["name"]:
                raise ValueError("Fixture names must be filename components")
            name = fixture["name"] + source.suffix
            if name in references:
                raise ValueError(f"Duplicate fixture name {name}")
            shutil.copyfile(source, photos / name)
            references[name] = fixture["reference"][0]

        log = root / "backend.log"
        service = Service(binary, models, root / "data", log)
        try:
            ready = service.until(lambda item: item.get("event") == "ready")["payload"]
            if ready["catalog"]["assetCount"] != 0:
                raise RuntimeError("Expected an empty temporary catalog")
            if ready["catalog"]["modelProfile"] != MODEL_PROFILE:
                raise RuntimeError("Unexpected catalog embedding profile")
            indexing_id = service.send({"type": "index", "directory": str(photos)})
            service.until(lambda item: item.get("id") == indexing_id and "ok" in item, timeout=300)
            indexed = service.response(indexing_id)
            if indexed["indexed"] != len(images) or indexed["failed"] or indexed["cancelled"]:
                raise RuntimeError(f"Incomplete indexing: {indexed}")

            results = []
            for query in queries:
                if len(query["reference"]) != 1:
                    raise ValueError(f"Expected one reference for {query['name']}")
                response = service.call({
                    "type": "search", "query": query["text"],
                    "limit": len(images), "maxResults": len(images),
                })
                actual = response["items"]
                if len(actual) != len(images) or {x["filename"] for x in actual} != set(references):
                    raise RuntimeError("Search omitted or duplicated fixture images")
                expected = {name: cosine(query["reference"][0], vector)
                            for name, vector in references.items()}
                if not all(math.isfinite(item["similarity"]) for item in actual):
                    raise RuntimeError("Search returned non-finite scores")
                errors = [abs(item["similarity"] - expected[item["filename"]]) for item in actual]
                if max(errors) > score_tolerance:
                    raise RuntimeError(f"{query['name']} score drift {max(errors)} exceeds {score_tolerance}")
                expected_order = sorted(expected, key=lambda name: (-expected[name], name))
                actual_order = [item["filename"] for item in actual]
                # Exact ties can be ordered by opaque asset ID by the real catalog.
                inversions = [expected[b] - expected[a]
                              for i, a in enumerate(actual_order) for b in actual_order[i + 1:]
                              if expected[b] > expected[a]]
                results.append({
                    "name": query["name"], "text": query["text"],
                    "maximumScoreError": max(errors),
                    "meanScoreError": sum(errors) / len(errors),
                    "top1Matches": actual_order[0] == expected_order[0],
                    "top5Overlap": len(set(actual_order[:5]) & set(expected_order[:5])) / 5,
                    "pairwiseInversions": len(inversions),
                    "inversionsBeyondTolerance": sum(gap > 2 * score_tolerance for gap in inversions),
                    "maximumInvertedReferenceGap": max(inversions, default=0),
                    "scores": [{"filename": item["filename"], "actual": item["similarity"],
                                "reference": expected[item["filename"]]} for item in actual],
                })
                print(f"[Retrieval parity] {query['name']} max score error={max(errors):.9g}", file=sys.stderr)
            return {
                "scope": "Synthetic HF preprocessing and pinned ONNX fixtures; numerical retrieval agreement, not original-checkpoint or labeled retrieval quality",
                "modelRepository": MODEL_REPOSITORY, "modelRevision": MODEL_REVISION,
                "embeddingDimension": EMBEDDING_DIM,
                "referenceMode": "native-projected" if native_fixtures else "indexed",
                "images": len(images), "queries": len(queries),
                "scoreComparisons": len(images) * len(queries),
                "scoreTolerance": score_tolerance,
                "maximumScoreError": max(r["maximumScoreError"] for r in results),
                "top1Matches": sum(r["top1Matches"] for r in results),
                "meanTop5Overlap": sum(r["top5Overlap"] for r in results) / len(results),
                "inversionsBeyondTolerance": sum(r["inversionsBeyondTolerance"] for r in results),
                "modelProfile": ready["catalog"]["modelProfile"],
                "provider": ready["provider"], "threads": ready["threads"],
                "results": results,
            }
        except Exception:
            print(log.read_text(), file=sys.stderr)
            raise
        finally:
            service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=pathlib.Path)
    parser.add_argument("models", type=pathlib.Path)
    parser.add_argument("fixtures", type=pathlib.Path)
    parser.add_argument("--native-fixtures", action="store_true",
                        help="Explicitly project native-output fixtures to indexed 512 dimensions")
    args = parser.parse_args()
    report = compare(args.binary.resolve(strict=True), args.models.resolve(strict=True),
                     args.fixtures.resolve(strict=True), args.native_fixtures)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
