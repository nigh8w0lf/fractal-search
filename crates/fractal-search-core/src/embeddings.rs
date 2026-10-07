//! Validation and L2 normalization of the pinned 768-dimensional output.
use crate::EMBEDDING_DIM as DIM;
use anyhow::{ensure, Result};

pub fn normalize(values: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        values.len() == DIM && values.iter().all(|x| x.is_finite()),
        "Invalid embedding"
    );
    let norm = values
        .iter()
        .map(|x| f64::from(*x).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(norm > 0.0 && norm.is_finite(), "Invalid embedding norm");
    Ok(values
        .iter()
        .map(|x| (f64::from(*x) / norm) as f32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_preserves_direction_and_rejects_invalid_embeddings() {
        let mut input = vec![0.; DIM];
        input[0] = 3.;
        input[1] = 4.;
        let output = normalize(&input).unwrap();
        assert!((output[0] - 0.6).abs() < 1e-6 && (output[1] - 0.8).abs() < 1e-6);
        assert!(normalize(&vec![0.; DIM]).is_err());
        assert!(normalize(&[1.]).is_err());
        input[0] = f32::NAN;
        assert!(normalize(&input).is_err());
    }
}
