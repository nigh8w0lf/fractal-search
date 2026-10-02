//! Native input preparation for the experimental dynamic-patch FG-CLIP2 export.
//! This module does not select a production provider or start a model service.

use anyhow::{ensure, Context, Result};
use image::{ColorType, DynamicImage, ImageDecoder, ImageFormat, ImageReader, RgbImage};
use std::{
    fs,
    io::{Cursor, Read},
    path::Path,
};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub const PATCH_SIZE: usize = 16;
pub const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * 3;
pub const SHORT_TEXT_LENGTH: usize = 64;
const PATCH_BUDGETS: [usize; 5] = [128, 256, 576, 784, 1024];
const MAX_PIXELS: u64 = 100_000_000;
const MAX_SIDE: u32 = 65_536;
const MAX_FILE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_TEXT_BYTES: usize = 64 * 1024;

pub struct ImageInputs {
    pub pixel_values: Vec<f32>,
    pub pixel_attention_mask: Vec<i32>,
    /// Patch grid [height, width] for each image, before padding.
    pub spatial_shapes: Vec<[usize; 2]>,
    pub max_patches: usize,
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    ensure!(
        width > 0 && height > 0 && width <= MAX_SIDE && height <= MAX_SIDE
            && u64::from(width) * u64::from(height) <= MAX_PIXELS,
        "FG-CLIP2 image dimensions {width}x{height} must contain 1..={MAX_PIXELS} pixels with sides <= {MAX_SIDE}"
    );
    Ok(())
}

pub fn patch_budget(width: u32, height: u32) -> Result<usize> {
    validate_dimensions(width, height)?;
    let count = u64::from(width / 16) * u64::from(height / 16);
    Ok(match count {
        0..=128 => 128,
        129..=256 => 256,
        257..=576 => 576,
        577..=784 => 784,
        _ => 1024,
    })
}

/// Match Siglip2's binary search, including rounding each side up to 16 pixels.
pub fn resized_dimensions(width: u32, height: u32, budget: usize) -> Result<(usize, usize)> {
    validate_dimensions(width, height)?;
    ensure!(
        PATCH_BUDGETS.contains(&budget),
        "Unsupported patch budget {budget}"
    );
    let scaled = |size: u32, scale: f64| {
        ((f64::from(size) * scale / PATCH_SIZE as f64).ceil() as usize).max(1) * PATCH_SIZE
    };
    let (mut lower, mut upper) = (0.000001_f64, 100.0_f64);
    while upper - lower >= 0.00001 {
        let scale = (lower + upper) / 2.0;
        let patches = (scaled(width, scale) / PATCH_SIZE) * (scaled(height, scale) / PATCH_SIZE);
        if patches <= budget {
            lower = scale;
        } else {
            upper = scale;
        }
    }
    let result = (scaled(width, lower), scaled(height, lower));
    ensure!(
        (result.0 / PATCH_SIZE) * (result.1 / PATCH_SIZE) <= budget,
        "Image aspect ratio cannot fit within {budget} patches"
    );
    Ok(result)
}

/// Detect content rather than trusting extensions; the app registers HEIC hooks at startup.
/// JPEG uses full-resolution libjpeg-turbo, matching Pillow's 8-bit RGB decode.
pub fn decode_image(path: &Path) -> Result<RgbImage> {
    let size = fs::metadata(path)
        .with_context(|| format!("Inspecting image {path:?}"))?
        .len();
    ensure!(
        size > 0 && size <= MAX_FILE_BYTES,
        "Image {path:?} must contain 1..={MAX_FILE_BYTES} bytes"
    );
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("Reading image {path:?}"))?;
    ensure!(
        bytes.len() as u64 <= MAX_FILE_BYTES,
        "Image {path:?} exceeded decode size limit"
    );
    let mut reader = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .with_context(|| format!("Detecting image format {path:?}"))?;
    if reader.format() == Some(ImageFormat::Jpeg) {
        let mut decoder = turbojpeg::Decompressor::new()?;
        let header = decoder
            .read_header(&bytes)
            .with_context(|| format!("Reading JPEG header {path:?}"))?;
        validate_dimensions(u32::try_from(header.width)?, u32::try_from(header.height)?)?;
        let decoded = turbojpeg::decompress(&bytes, turbojpeg::PixelFormat::RGB)
            .with_context(|| format!("Decoding JPEG {path:?}"))?;
        return RgbImage::from_raw(header.width as u32, header.height as u32, decoded.pixels)
            .context("JPEG decoder returned an invalid RGB buffer");
    }
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_PIXELS * 4);
    reader.limits(limits);
    let decoder = reader
        .into_decoder()
        .with_context(|| format!("Opening image decoder {path:?}"))?;
    let (width, height) = decoder.dimensions();
    validate_dimensions(width, height)?;
    ensure!(
        matches!(
            decoder.color_type(),
            ColorType::Rgb8 | ColorType::Rgba8 | ColorType::L8 | ColorType::La8
        ),
        "FG-CLIP2 requires an 8-bit image; {path:?} has {:?}",
        decoder.color_type()
    );
    // Like Pillow convert('RGB'), discard alpha without compositing or EXIF rotation.
    Ok(DynamicImage::from_decoder(decoder)
        .with_context(|| format!("Decoding image {path:?}"))?
        .to_rgb8())
}

struct AxisWeights {
    rows: Vec<Vec<(usize, i32)>>,
    precision: u32,
}

// PyTorch 2.9 CPU uint8 antialiased bilinear uses double weights quantized to
// int16 and rounds/clips after each separable pass. A float resize followed by
// one conversion to uint8 does not reproduce these pixels.
fn axis_weights(input: usize, output: usize) -> AxisWeights {
    let scale = input as f64 / output as f64;
    let support = scale.max(1.0);
    let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_size = support.ceil() as usize * 2 + 1;
    let mut max_weight = 0.0_f64;
    let floats: Vec<Vec<(usize, f64)>> = (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize)
                .min(input)
                .min(start + max_size);
            let mut row: Vec<_> = (start..end)
                .map(|j| {
                    (
                        j,
                        (1.0 - ((j as f64 - center + 0.5) * inverse_scale).abs()).max(0.0),
                    )
                })
                .collect();
            let total: f64 = row.iter().map(|(_, weight)| weight).sum();
            for (_, weight) in &mut row {
                *weight /= total;
                max_weight = max_weight.max(*weight);
            }
            row
        })
        .collect();
    let mut precision = 0;
    while precision < 22 {
        if (0.5 + max_weight * f64::from(1_u32 << (precision + 1))) as i32 >= 32768 {
            break;
        }
        precision += 1;
    }
    AxisWeights {
        rows: floats
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|(j, weight)| (j, (0.5 + weight * f64::from(1_u32 << precision)) as i32))
                    .collect()
            })
            .collect(),
        precision,
    }
}

fn resize_rgb(image: &RgbImage, width: usize, height: usize) -> Vec<u8> {
    let source_width = image.width() as usize;
    let source_height = image.height() as usize;
    let horizontal = if width == source_width {
        image.as_raw().clone()
    } else {
        let weights = axis_weights(source_width, width);
        let mut pixels = vec![0; source_height * width * 3];
        for y in 0..source_height {
            for (x, row) in weights.rows.iter().enumerate() {
                for c in 0..3 {
                    let sum =
                        row.iter()
                            .fold(1_i32 << (weights.precision - 1), |sum, &(j, weight)| {
                                sum + i32::from(image.as_raw()[(y * source_width + j) * 3 + c])
                                    * weight
                            });
                    pixels[(y * width + x) * 3 + c] =
                        (sum >> weights.precision).clamp(0, 255) as u8;
                }
            }
        }
        pixels
    };
    if height == source_height {
        return horizontal;
    }
    let weights = axis_weights(source_height, height);
    let mut pixels = vec![0; height * width * 3];
    for (y, row) in weights.rows.iter().enumerate() {
        for x in 0..width {
            for c in 0..3 {
                let sum = row
                    .iter()
                    .fold(1_i32 << (weights.precision - 1), |sum, &(j, weight)| {
                        sum + i32::from(horizontal[(j * width + x) * 3 + c]) * weight
                    });
                pixels[(y * width + x) * 3 + c] = (sum >> weights.precision).clamp(0, 255) as u8;
            }
        }
    }
    pixels
}

pub fn prepare_images(images: &[RgbImage]) -> Result<ImageInputs> {
    ensure!(
        (1..=8).contains(&images.len()),
        "FG-CLIP2 image batches must contain 1..=8 images"
    );
    let budget = images
        .iter()
        .map(|image| patch_budget(image.width(), image.height()))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("Empty image batch")?;
    let mut result = ImageInputs {
        pixel_values: vec![0.0; images.len() * budget * PATCH_DIM],
        pixel_attention_mask: vec![0; images.len() * budget],
        spatial_shapes: Vec::with_capacity(images.len()),
        max_patches: budget,
    };
    for (batch, image) in images.iter().enumerate() {
        let (width, height) = resized_dimensions(image.width(), image.height(), budget)?;
        let pixels = resize_rgb(image, width, height);
        let (rows, columns) = (height / PATCH_SIZE, width / PATCH_SIZE);
        result.spatial_shapes.push([rows, columns]);
        for row in 0..rows {
            for column in 0..columns {
                let patch = row * columns + column;
                result.pixel_attention_mask[batch * budget + patch] = 1;
                for y in 0..PATCH_SIZE {
                    let offset = ((row * PATCH_SIZE + y) * width + column * PATCH_SIZE) * 3;
                    for (i, &pixel) in pixels[offset..offset + PATCH_SIZE * 3].iter().enumerate() {
                        // Fused rescale/normalize in Siglip2ImageProcessorFast.
                        result.pixel_values
                            [(batch * budget + patch) * PATCH_DIM + y * PATCH_SIZE * 3 + i] =
                            (f32::from(pixel) - 127.5) / 127.5;
                    }
                }
            }
        }
    }
    Ok(result)
}

pub struct ShortTextTokenizer {
    tokenizer: Tokenizer,
}

impl ShortTextTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(path)
            .map_err(|error| anyhow::anyhow!("Loading FG-CLIP2 tokenizer {path:?}: {error}"))?;
        ensure!(
            tokenizer.get_vocab_size(true) == 256000,
            "Unexpected FG-CLIP2 tokenizer vocabulary size"
        );
        ensure!(
            tokenizer.token_to_id("<pad>") == Some(0) && tokenizer.token_to_id("<eos>") == Some(1),
            "Unexpected FG-CLIP2 pad/EOS tokens"
        );
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: SHORT_TEXT_LENGTH,
                ..Default::default()
            }))
            .map_err(|error| anyhow::anyhow!("Configuring FG-CLIP2 truncation: {error}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(SHORT_TEXT_LENGTH),
            direction: PaddingDirection::Right,
            pad_id: 0,
            pad_token: "<pad>".into(),
            ..Default::default()
        }));
        Ok(Self { tokenizer })
    }

    /// The current export accepts only one short query. Long mode is an error.
    pub fn encode(&self, text: &str, mode: &str) -> Result<Vec<i32>> {
        ensure!(
            mode == "short",
            "FG-CLIP2 ONNX supports only short text mode, requested {mode:?}"
        );
        ensure!(
            text.len() <= MAX_TEXT_BYTES,
            "FG-CLIP2 query exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
        );
        let encoding = self
            .tokenizer
            .encode(text.to_lowercase(), true)
            .map_err(|error| anyhow::anyhow!("Tokenizing FG-CLIP2 query: {error}"))?;
        ensure!(
            encoding.get_ids().len() == SHORT_TEXT_LENGTH,
            "Tokenizer did not return 64 tokens"
        );
        ensure!(
            encoding.get_ids().iter().all(|&id| id < 256000),
            "Tokenizer returned an invalid token ID"
        );
        let last = encoding
            .get_attention_mask()
            .iter()
            .rposition(|&value| value == 1)
            .context("Tokenizer returned no active tokens")?;
        ensure!(
            encoding.get_ids()[last] == 1,
            "Tokenizer did not append EOS before padding"
        );
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_policy_boundaries_and_invalid_inputs() {
        for (width, height, expected) in [
            (128, 256, 128),
            (144, 256, 256),
            (256, 256, 256),
            (272, 256, 576),
            (384, 384, 576),
            (400, 384, 784),
            (448, 448, 784),
            (464, 448, 1024),
        ] {
            assert_eq!(patch_budget(width, height).unwrap(), expected);
        }
        assert!(patch_budget(0, 10).is_err());
        assert!(patch_budget(100001, 1000).is_err());
        assert!(resized_dimensions(10, 10, 129).is_err());
        assert!(prepare_images(&[]).is_err());
    }

    #[test]
    fn upstream_grid_golden_values() {
        for (width, height, budget, expected) in [
            (736, 1181, 1024, (400, 640)),
            (1600, 900, 1024, (672, 384)),
            (96, 192, 128, (128, 256)),
            (1, 1, 128, (112, 112)),
        ] {
            assert_eq!(resized_dimensions(width, height, budget).unwrap(), expected);
        }
    }

    #[test]
    fn upstream_uint8_resize_golden_values() {
        let source = RgbImage::from_raw(5, 4, (0..60).collect()).unwrap();
        assert_eq!(
            resize_rgb(&source, 3, 2),
            [12, 13, 14, 17, 18, 19, 22, 23, 24, 35, 36, 37, 40, 41, 42, 45, 46, 47]
        );
        let source = RgbImage::from_raw(3, 2, (0..18).collect()).unwrap();
        assert_eq!(
            resize_rgb(&source, 5, 4),
            [
                0, 1, 2, 1, 2, 3, 3, 4, 5, 5, 6, 7, 6, 7, 8, 2, 3, 4, 3, 4, 5, 5, 6, 7, 7, 8, 9, 8,
                9, 10, 7, 8, 9, 8, 9, 10, 10, 11, 12, 12, 13, 14, 13, 14, 15, 9, 10, 11, 10, 11,
                12, 12, 13, 14, 14, 15, 16, 15, 16, 17,
            ]
        );
    }

    #[test]
    fn decoder_rejects_corruption_and_unsupported_depth() {
        let path = std::env::temp_dir().join(format!("fgclip2-input-{}.png", uuid::Uuid::new_v4()));
        fs::write(&path, b"invalid image").unwrap();
        assert!(decode_image(&path).is_err());
        let transparent = image::ImageBuffer::from_pixel(2, 2, image::Rgba([100_u8, 150, 200, 0]));
        transparent.save(&path).unwrap();
        assert_eq!(
            decode_image(&path).unwrap().get_pixel(0, 0).0,
            [100, 150, 200]
        );
        let image = image::ImageBuffer::from_pixel(2, 2, image::Luma([65535_u16]));
        image.save(&path).unwrap();
        assert!(decode_image(&path)
            .unwrap_err()
            .to_string()
            .contains("8-bit"));
        fs::remove_file(&path).unwrap();
        assert!(decode_image(&path).is_err());
    }

    #[test]
    fn unsupported_text_mode_and_oversize_queries_fail() {
        let tokenizer = ShortTextTokenizer {
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
        };
        assert!(tokenizer
            .encode("a photo", "long")
            .unwrap_err()
            .to_string()
            .contains("only short"));
        assert!(tokenizer
            .encode(&"a".repeat(MAX_TEXT_BYTES + 1), "short")
            .is_err());
    }

    #[test]
    fn patch_layout_normalization_and_padding() {
        let mut image = RgbImage::new(128, 256);
        image.put_pixel(0, 0, image::Rgb([0, 255, 128]));
        image.put_pixel(16, 0, image::Rgb([255, 0, 255]));
        let inputs = prepare_images(&[image]).unwrap();
        assert_eq!(inputs.spatial_shapes, [[16, 8]]);
        assert_eq!(inputs.pixel_values[..3], [-1.0, 1.0, 0.5 / 127.5]);
        assert_eq!(
            inputs.pixel_values[PATCH_DIM..PATCH_DIM + 3],
            [1.0, -1.0, 1.0]
        );
        let small = RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 128]));
        let inputs = prepare_images(&[small]).unwrap();
        assert_eq!(inputs.pixel_attention_mask.iter().sum::<i32>(), 49);
        assert!(inputs.pixel_attention_mask[49..]
            .iter()
            .all(|&value| value == 0));
        assert!(inputs.pixel_values[49 * PATCH_DIM..]
            .iter()
            .all(|&value| value == 0.0));
    }
}
