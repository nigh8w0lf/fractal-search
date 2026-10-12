"""Explicit native/reference and indexed EmbeddingGemma 2 contracts for dev tools.

This is independent reference math, not a production inference dependency.
"""

import math

NATIVE_EMBEDDING_DIM = 768
EMBEDDING_DIM = 512
NATIVE_MODEL_PROFILE = "embeddinggemma2-daa72c51-fp32-multimodal-v1"
MODEL_PROFILE = "embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2"


def project_indexed_vector(vector):
    """Validate the native output, keep its leading 512 values, then L2-normalize."""
    if len(vector) != NATIVE_EMBEDDING_DIM:
        raise ValueError("Expected a native 768-dimensional embedding before MRL projection")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Native embedding must contain only finite values")
    prefix = vector[:EMBEDDING_DIM]
    norm = math.sqrt(sum(value * value for value in prefix))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Leading 512 embedding dimensions must have a finite nonzero norm")
    result = [value / norm for value in prefix]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Projected embedding must contain only finite values")
    return result


def validate_indexed_vector(vector):
    """Indexed references must already be normalized, not an unnormalized slice."""
    if len(vector) != EMBEDDING_DIM:
        raise ValueError("Expected an indexed 512-dimensional reference vector")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Indexed reference vector must contain only finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or not math.isclose(norm, 1.0, abs_tol=1e-5, rel_tol=0):
        raise ValueError("Indexed reference vector must be unit normalized after truncation")


def validate_fixture_environment(environment, native=False):
    """Reject mismatched fixture modes instead of silently changing the contract."""
    expected = {
        "output_mode": "native" if native else "indexed",
        "embedding_dimension": NATIVE_EMBEDDING_DIM if native else EMBEDDING_DIM,
        "native_embedding_dimension": NATIVE_EMBEDDING_DIM,
        "model_profile": NATIVE_MODEL_PROFILE if native else MODEL_PROFILE,
    }
    # Old fixture generators emitted only the pinned export identity. The caller
    # must explicitly select native mode; every reference then requires 768 values.
    if native and not any(key in environment for key in expected):
        return
    for key, value in expected.items():
        if environment.get(key) != value:
            raise ValueError(
                f"Fixture {key} must be {value!r}; regenerate fixtures in the requested mode. "
                "Native 768 fixtures remain supported by --native-output parity validation; "
                "use --native-fixtures to project them explicitly for indexed retrieval."
            )
