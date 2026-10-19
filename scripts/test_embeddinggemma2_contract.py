"""Model-free checks for native-vs-indexed fixture contracts and MRL reference math."""

import math
import unittest

from embeddinggemma2_contract import (
    EMBEDDING_DIM, MODEL_PROFILE, NATIVE_EMBEDDING_DIM, NATIVE_MODEL_PROFILE,
    project_indexed_vector, validate_fixture_environment, validate_indexed_vector,
)


class EmbeddingContractTests(unittest.TestCase):
    def test_leading_dimensions_are_renormalized(self):
        native = [3.0, 4.0] + [0.0] * 510 + [12.0] + [0.0] * 255
        native = [value / 13.0 for value in native]
        indexed = project_indexed_vector(native)
        self.assertEqual(len(indexed), EMBEDDING_DIM)
        self.assertAlmostEqual(indexed[0], 0.6)
        self.assertAlmostEqual(indexed[1], 0.8)
        self.assertAlmostEqual(sum(value * value for value in indexed), 1.0)
        self.assertEqual(native[512], 12.0 / 13.0)

    def test_tail_cannot_affect_projected_direction(self):
        a = [float(i + 1) for i in range(512)] + [1.0] * 256
        b = a[:512] + [-100.0] * 256
        self.assertEqual(project_indexed_vector(a), project_indexed_vector(b))

    def test_batched_reference_projection_is_per_vector(self):
        vectors = [[1.0] * 768, [2.0] * 512 + [-10.0] * 256]
        batch = [project_indexed_vector(vector) for vector in vectors]
        self.assertEqual(batch[0], batch[1])
        self.assertEqual(len(batch[0]), 512)
        self.assertAlmostEqual(batch[0][-1], 1.0 / math.sqrt(512))
        for vector in batch:
            validate_indexed_vector(vector)

    def test_wrong_native_dimensions_are_rejected(self):
        for count in (0, 256, 512, 767, 769):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "native 768"):
                project_indexed_vector([1.0] * count)

    def test_nonfinite_native_values_including_discarded_tail_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for index in (0, 512, 767):
                native = [1.0] * 768
                native[index] = value
                with self.subTest(value=value, index=index), self.assertRaisesRegex(ValueError, "finite"):
                    project_indexed_vector(native)

    def test_zero_prefix_with_nonzero_native_tail_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonzero norm"):
            project_indexed_vector([0.0] * 512 + [1.0] * 256)

    def test_overflowed_prefix_norm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite nonzero norm"):
            project_indexed_vector([1e308] * 768)

    def test_indexed_references_reject_bare_slices_and_native_vectors(self):
        native = [1.0 / math.sqrt(768)] * 768
        for invalid in (native, native[:512], [0.0] * 512, [float("nan")] * 512):
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                validate_indexed_vector(invalid)
        validate_indexed_vector(project_indexed_vector(native))

    def test_indexed_fixture_metadata_is_explicit(self):
        indexed = {"output_mode": "indexed", "embedding_dimension": 512,
                   "native_embedding_dimension": 768, "model_profile": MODEL_PROFILE}
        validate_fixture_environment(indexed)
        for key in indexed:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "regenerate"):
                validate_fixture_environment({k: v for k, v in indexed.items() if k != key})
        with self.assertRaises(ValueError):
            validate_fixture_environment(indexed, native=True)

    def test_native_fixtures_require_explicit_mode(self):
        native = {"output_mode": "native", "embedding_dimension": 768,
                  "native_embedding_dimension": 768, "model_profile": NATIVE_MODEL_PROFILE}
        validate_fixture_environment(native, native=True)
        with self.assertRaises(ValueError):
            validate_fixture_environment(native)
        # Unversioned references may contain FP32 outputs, even in native mode.
        with self.assertRaises(ValueError):
            validate_fixture_environment({}, native=True)
        with self.assertRaises(ValueError):
            validate_fixture_environment({})
        with self.assertRaises(ValueError):
            validate_fixture_environment({"embedding_dimension": 512}, native=True)

    def test_fp32_references_are_rejected_in_both_modes(self):
        for native, profile in [(False, "embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2"),
                                (True, "embeddinggemma2-daa72c51-fp32-multimodal-v1")]:
            env = {"output_mode": "native" if native else "indexed",
                   "embedding_dimension": 768 if native else 512,
                   "native_embedding_dimension": 768, "model_profile": profile}
            with self.assertRaisesRegex(ValueError, "model_profile"):
                validate_fixture_environment(env, native=native)


if __name__ == "__main__":
    unittest.main()
