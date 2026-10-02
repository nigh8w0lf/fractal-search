//! Pinned FG-CLIP2 table interpolation, token lookup, and output normalization.
use crate::EMBEDDING_DIM as DIM;
use anyhow::{ensure, Result};
use half::f16;

// Separable bilinear interpolation with widened filtering when downsampling,
// half-pixel centers and boundary renormalization. These are the upstream
// F.interpolate(align_corners=False, antialias=True) semantics.
fn weights(output_size: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = 16.0_f32 / output_size as f32;
    let support = scale.max(1.0);
    (0..output_size)
        .map(|i| {
            let center = scale * (i as f32 + 0.5);
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(16);
            let mut row: Vec<_> = (start..end)
                .map(|j| {
                    (
                        j,
                        (1.0 - ((j as f32 - center + 0.5) / support).abs()).max(0.0),
                    )
                })
                .collect();
            let total: f32 = row.iter().map(|(_, weight)| weight).sum();
            for (_, weight) in &mut row {
                *weight /= total;
            }
            row
        })
        .collect()
}

pub fn position_embeddings(
    table: &[f32],
    spatial: &[[usize; 2]],
    patches: usize,
) -> Result<Vec<f32>> {
    ensure!(
        table.len() == 16 * 16 * DIM,
        "Invalid vision position table"
    );
    ensure!(
        (1..=8).contains(&spatial.len()) && (1..=1024).contains(&patches),
        "Position tensors require 1..=8 images and 1..=1024 patches"
    );
    let mut result = vec![0.0; spatial.len() * patches * DIM];
    for (batch, &[height, width]) in spatial.iter().enumerate() {
        ensure!(
            height > 0
                && width > 0
                && height <= patches
                && width <= patches
                && height.checked_mul(width).is_some_and(|n| n <= patches),
            "Invalid spatial shape {height}x{width} for {patches} patches"
        );
        let horizontal_weights = weights(width);
        let vertical_weights = weights(height);
        let mut horizontal = vec![0.0_f32; 16 * width * DIM];
        for y in 0..16 {
            for (x, row) in horizontal_weights.iter().enumerate() {
                for &(source_x, weight) in row {
                    for channel in 0..DIM {
                        horizontal[(y * width + x) * DIM + channel] +=
                            table[(y * 16 + source_x) * DIM + channel] * weight;
                    }
                }
            }
        }
        let offset = batch * patches * DIM;
        for (y, row) in vertical_weights.iter().enumerate() {
            // Pinned Torch 2.9.1 CPU's float antialias kernel repeats the first
            // output row when width is one and the vertical axis is resized.
            // Preserve those existing embeddings; changing it needs a profile
            // migration and must not silently use different positions.
            let row = if width == 1 && height != 16 {
                &vertical_weights[0]
            } else {
                row
            };
            for x in 0..width {
                for &(source_y, weight) in row {
                    for channel in 0..DIM {
                        result[offset + (y * width + x) * DIM + channel] +=
                            horizontal[(source_y * width + x) * DIM + channel] * weight;
                    }
                }
            }
        }
        // The upstream model pads positions with the first resized position.
        for patch in height * width..patches {
            for channel in 0..DIM {
                result[offset + patch * DIM + channel] = result[offset + channel];
            }
        }
    }
    Ok(result)
}

pub fn token_embeddings(table: &[u8], ids: &[i32]) -> Result<Vec<f32>> {
    ensure!(
        table.len() == 256000 * DIM * 2,
        "Invalid fp16 token table size"
    );
    let mut result = Vec::with_capacity(ids.len() * DIM);
    for &id in ids {
        ensure!((0..256000).contains(&id), "Invalid token ID {id}");
        let offset = id as usize * DIM * 2;
        for bytes in table[offset..offset + DIM * 2].chunks_exact(2) {
            let value = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
            ensure!(value.is_finite(), "Non-finite embedding for token {id}");
            result.push(value);
        }
    }
    Ok(result)
}

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
    fn pinned_torch_single_column_positions_and_input_bounds() {
        let table: Vec<_> = (0..256)
            .flat_map(|i| std::iter::repeat_n(i as f32, DIM))
            .collect();
        let resized = position_embeddings(&table, &[[128, 1]], 128).unwrap();
        assert!((resized[0] - 7.5).abs() < 0.00001);
        assert!(resized.chunks_exact(DIM).all(|row| row == &resized[..DIM]));
        let unscaled = position_embeddings(&table, &[[16, 1]], 128).unwrap();
        assert!((unscaled[15 * DIM] - 247.5).abs() < 0.0001);
        assert_eq!(&unscaled[16 * DIM..17 * DIM], &unscaled[..DIM]);
        assert!(position_embeddings(&table, &[[1, 1]], usize::MAX).is_err());
        assert!(position_embeddings(&table, &[[0, 1]], 128).is_err());
        assert!(position_embeddings(&table, &[[33, 33]], 1024).is_err());
    }

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
        assert!(token_embeddings(&[], &[1]).is_err());
    }
}
