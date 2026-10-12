//! Native-output validation and the fixed indexed Matryoshka projection.
use crate::{EMBEDDING_DIM, NATIVE_EMBEDDING_DIM};
use anyhow::{ensure, Result};

fn validate_native(values: &[f32]) -> Result<()> {
    ensure!(
        values.len() == NATIVE_EMBEDDING_DIM,
        "Native embedding must have {NATIVE_EMBEDDING_DIM} dimensions, got {}",
        values.len()
    );
    // Check the entire output, including dimensions discarded by projection.
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "Native embedding contains non-finite values"
    );
    Ok(())
}

fn normalize_validated(values: &[f32]) -> Result<Vec<f32>> {
    let norm = values
        .iter()
        .map(|x| f64::from(*x).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm > 0.0 && norm.is_finite(),
        "Invalid embedding norm: expected a finite, nonzero vector"
    );
    Ok(values
        .iter()
        .map(|x| (f64::from(*x) / norm) as f32)
        .collect())
}

/// Preserve the complete pinned graph output for native reference parity.
pub fn normalize(values: &[f32]) -> Result<Vec<f32>> {
    validate_native(values)?;
    normalize_validated(values)
}

/// Keep the leading 512 of exactly 768 native dimensions, then normalize again.
/// Accepts normalized or unnormalized native outputs. All production queries
/// and document/image/audio/video vectors must use this identical projection.
pub fn project(values: &[f32]) -> Result<Vec<f32>> {
    validate_native(values)?;
    normalize_validated(&values[..EMBEDDING_DIM])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(values: &[f32]) -> f64 {
        values
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum()
    }
    fn sample(seed: usize) -> Vec<f32> {
        (0..NATIVE_EMBEDDING_DIM)
            .map(|i| ((i * 23 + seed * 17) % 41) as f32 - 20.0)
            .collect()
    }

    #[test]
    fn normalization_preserves_native_direction_and_width() {
        let mut input = vec![0.; NATIVE_EMBEDDING_DIM];
        input[0] = 3.;
        input[1] = 4.;
        let output = normalize(&input).unwrap();
        assert_eq!(output.len(), 768);
        assert!((output[0] - 0.6).abs() < 1e-6 && (output[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn projection_is_leading_dimensions_and_renormalizes() {
        let mut input = vec![0.; NATIVE_EMBEDDING_DIM];
        input[0] = 3.;
        input[EMBEDDING_DIM - 1] = 4.;
        input[EMBEDDING_DIM] = 12.;
        let output = project(&input).unwrap();
        assert_eq!(output.len(), 512);
        assert!((output[0] - 0.6).abs() < 1e-6);
        assert!((output[EMBEDDING_DIM - 1] - 0.8).abs() < 1e-6);
        assert!((norm(&output) - 1.).abs() < 1e-7);
        assert!(norm(&normalize(&input).unwrap()[..EMBEDDING_DIM]) < 0.4);
        assert_eq!(EMBEDDING_DIM * std::mem::size_of::<f32>(), 2048);
    }

    #[test]
    fn normalized_and_unnormalized_projection_and_batch_single_parity() {
        let batch = (0..8).map(sample).collect::<Vec<_>>();
        let projected = batch
            .iter()
            .map(|v| project(v).unwrap())
            .collect::<Vec<_>>();
        for (raw, actual) in batch.iter().zip(projected) {
            assert_eq!(project(raw).unwrap(), actual);
            assert!((norm(&actual) - 1.).abs() < 1e-7);
            let normalized = project(&normalize(raw).unwrap()).unwrap();
            assert!(actual
                .iter()
                .zip(normalized)
                .all(|(a, b)| (*a - b).abs() < 1e-7));
        }
    }

    #[test]
    fn projected_dot_equals_truncated_cosine_not_raw_truncated_dot() {
        let a = normalize(&sample(1)).unwrap();
        let b = normalize(&sample(2)).unwrap();
        let (a_prefix, b_prefix) = (&a[..EMBEDDING_DIM], &b[..EMBEDDING_DIM]);
        let cosine = dot(a_prefix, b_prefix) / (norm(a_prefix) * norm(b_prefix));
        let projected_dot = dot(&project(&a).unwrap(), &project(&b).unwrap());
        assert!((projected_dot - cosine).abs() < 1e-7);
        assert!((projected_dot - dot(a_prefix, b_prefix)).abs() > 1e-3);
    }

    #[test]
    fn native_and_projection_reject_malformed_or_nonfinite_outputs() {
        for dimension in [0, 1, 256, 511, 512, 513, 767, 769] {
            assert!(normalize(&vec![1.; dimension]).is_err());
            assert!(project(&vec![1.; dimension]).is_err());
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in [0, EMBEDDING_DIM, NATIVE_EMBEDDING_DIM - 1] {
                let mut input = sample(0);
                input[index] = bad;
                assert!(normalize(&input).is_err());
                assert!(project(&input).is_err());
            }
        }
        assert!(normalize(&vec![0.; NATIVE_EMBEDDING_DIM]).is_err());
        assert!(project(&vec![0.; NATIVE_EMBEDDING_DIM]).is_err());
        let mut tail_only = vec![0.; NATIVE_EMBEDDING_DIM];
        tail_only[EMBEDDING_DIM] = 1.;
        assert!(normalize(&tail_only).is_ok());
        assert!(project(&tail_only).is_err());
        for scale in [f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
            let projected = project(&vec![scale; NATIVE_EMBEDDING_DIM]).unwrap();
            assert!((norm(&projected) - 1.).abs() < 1e-7);
        }
    }
}
