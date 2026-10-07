//! Pinned EmbeddingGemma 2 image and search-query preparation.
//! This module does not select a production provider or start a model service.

use crate::MAX_IMAGE_BATCH_SIZE;
use anyhow::{ensure, Context, Result};
use image::{ColorType, DynamicImage, ImageDecoder, ImageFormat, ImageReader, RgbImage};
use std::{
    fs,
    io::{Cursor, Read},
    path::Path,
};
use tokenizers::Tokenizer;

pub const PATCH_SIZE: usize = 16;
pub const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * 3;
pub const POOLING_KERNEL_SIZE: usize = 3;
pub const MAX_SOFT_TOKENS: usize = 280;
pub const VIDEO_MAX_SOFT_TOKENS: usize = 140;
pub const MAX_VIDEO_FRAMES: usize = 32;
pub const AUDIO_SAMPLE_RATE: usize = 16000;
pub const MAX_AUDIO_SAMPLES: usize = 30 * AUDIO_SAMPLE_RATE;
pub const AUDIO_FEATURE_DIM: usize = 128;
pub const MAX_PATCHES: usize = MAX_SOFT_TOKENS * POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE;
pub const MAX_SEQUENCE_LENGTH: usize = 8192;
pub const QUERY_PREFIX: &str = "task: search result | query: ";
pub const HIDDEN_DIM: usize = 512;
pub const IMAGE_TOKEN_ID: i64 = 258880;
pub const AUDIO_TOKEN_ID: i64 = 258881;
pub const VIDEO_TOKEN_ID: i64 = 258884;
const BOA_TOKEN_ID: i64 = 256000;
const EOA_TOKEN_ID: i64 = 258883;
const BOS_TOKEN_ID: i64 = 2;
const EOS_TOKEN_ID: i64 = 1;
const BOI_TOKEN_ID: i64 = 255999;
const EOI_TOKEN_ID: i64 = 258882;
const MAX_PIXELS: u64 = 100_000_000;
const MAX_SIDE: u32 = 65_536;
const MAX_FILE_BYTES: u64 = 128 * 1024 * 1024;
pub const MAX_TEXT_BYTES: usize = 4096;

pub struct ImageInputs {
    pub pixel_values: Vec<f32>,
    pub pixel_position_ids: Vec<i64>,
    pub num_soft_tokens: Vec<usize>,
    /// Patch grid [height, width] for each image, before padding.
    pub spatial_shapes: Vec<[usize; 2]>,
    pub max_patches: usize,
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    ensure!(
        width > 0 && height > 0 && width <= MAX_SIDE && height <= MAX_SIDE
            && u64::from(width) * u64::from(height) <= MAX_PIXELS,
        "EmbeddingGemma 2 image dimensions {width}x{height} must contain 1..={MAX_PIXELS} pixels with sides <= {MAX_SIDE}"
    );
    Ok(())
}

/// The pinned processor uses one fixed vision budget, independent of input size.
pub fn patch_budget(width: u32, height: u32) -> Result<usize> {
    validate_dimensions(width, height)?;
    Ok(MAX_PATCHES)
}

/// Gemma4 aspect-preserving resize, rounded down to patch/pooling multiples.
pub fn resized_dimensions(width: u32, height: u32, budget: usize) -> Result<(usize, usize)> {
    validate_dimensions(width, height)?;
    ensure!(
        [MAX_PATCHES, VIDEO_MAX_SOFT_TOKENS * 9].contains(&budget),
        "Unsupported patch budget {budget}"
    );
    let multiple = PATCH_SIZE * POOLING_KERNEL_SIZE;
    let factor =
        ((budget * PATCH_SIZE * PATCH_SIZE) as f64 / (f64::from(width) * f64::from(height))).sqrt();
    let mut target_width =
        (factor * f64::from(width) / multiple as f64).floor() as usize * multiple;
    let mut target_height =
        (factor * f64::from(height) / multiple as f64).floor() as usize * multiple;
    ensure!(
        target_width > 0 || target_height > 0,
        "Image cannot fit the vision budget"
    );
    let max_side = budget / (POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE) * multiple;
    if target_width == 0 {
        target_width = multiple;
        target_height = ((height / width) as usize * multiple).min(max_side);
    } else if target_height == 0 {
        target_height = multiple;
        target_width = ((width / height) as usize * multiple).min(max_side);
    }
    ensure!(
        target_width / PATCH_SIZE * (target_height / PATCH_SIZE) <= budget,
        "Image aspect ratio exceeds the vision budget"
    );
    Ok((target_width, target_height))
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
        "EmbeddingGemma 2 requires an 8-bit image; {path:?} has {:?}",
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

// Separable antialiased bicubic uint8 resize (a = -0.5, half-pixel centers).
// Widen the support for downsampling and round/clip after each pass, matching
// the pinned processor's uint8 path. This algorithm is part of MODEL_PROFILE.
fn cubic(value: f64) -> f64 {
    let x = value.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}
fn axis_weights(input: usize, output: usize) -> AxisWeights {
    let scale = input as f64 / output as f64;
    let support = 2.0 * scale.max(1.0);
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
                .map(|j| (j, cubic((j as f64 - center + 0.5) * inverse_scale)))
                .collect();
            let total: f64 = row.iter().map(|(_, weight)| weight).sum();
            for (_, weight) in &mut row {
                *weight /= total;
                max_weight = max_weight.max(weight.abs());
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
                    .map(|(j, weight)| (j, (weight * f64::from(1_u32 << precision)).round() as i32))
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
    prepare_visual_inputs(images, MAX_PATCHES, MAX_IMAGE_BATCH_SIZE)
}

pub fn prepare_video_frames(frames: &[RgbImage]) -> Result<ImageInputs> {
    prepare_visual_inputs(frames, VIDEO_MAX_SOFT_TOKENS * 9, MAX_VIDEO_FRAMES)
}

fn prepare_visual_inputs(
    images: &[RgbImage],
    budget: usize,
    max_batch: usize,
) -> Result<ImageInputs> {
    ensure!(
        (1..=max_batch).contains(&images.len()),
        "Visual batch must contain 1..={max_batch} frames"
    );
    let mut result = ImageInputs {
        pixel_values: vec![0.0; images.len() * budget * PATCH_DIM],
        pixel_position_ids: vec![-1; images.len() * budget * 2],
        num_soft_tokens: Vec::with_capacity(images.len()),
        spatial_shapes: Vec::with_capacity(images.len()),
        max_patches: budget,
    };
    for (batch, image) in images.iter().enumerate() {
        let (width, height) = resized_dimensions(image.width(), image.height(), budget)?;
        let pixels = resize_rgb(image, width, height);
        let (rows, columns) = (height / PATCH_SIZE, width / PATCH_SIZE);
        result.spatial_shapes.push([rows, columns]);
        result
            .num_soft_tokens
            .push(rows * columns / (POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE));
        for row in 0..rows {
            for column in 0..columns {
                let patch = row * columns + column;
                result.pixel_position_ids[(batch * budget + patch) * 2] = column as i64;
                result.pixel_position_ids[(batch * budget + patch) * 2 + 1] = row as i64;
                for y in 0..PATCH_SIZE {
                    let offset = ((row * PATCH_SIZE + y) * width + column * PATCH_SIZE) * 3;
                    for (i, &pixel) in pixels[offset..offset + PATCH_SIZE * 3].iter().enumerate() {
                        // Gemma4 rescales only: no CLIP mean/std normalization.
                        result.pixel_values
                            [(batch * budget + patch) * PATCH_DIM + y * PATCH_SIZE * 3 + i] =
                            f32::from(pixel) / 255.0;
                    }
                }
            }
        }
    }
    Ok(result)
}

/// Row-major token batch; padding is on the right with a zero attention mask.
pub struct TextInputs {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub sequence_length: usize,
}

/// One independent image per sample, with no text task prefix for media.
pub fn image_tokens(soft_tokens: &[usize]) -> Result<TextInputs> {
    ensure!(
        (1..=MAX_IMAGE_BATCH_SIZE).contains(&soft_tokens.len()),
        "Invalid image token batch size"
    );
    ensure!(
        soft_tokens
            .iter()
            .all(|n| (1..=MAX_SOFT_TOKENS).contains(n)),
        "Invalid image soft token count"
    );
    let sequence_length = soft_tokens.iter().max().copied().unwrap() + 4;
    let mut input_ids = vec![0; soft_tokens.len() * sequence_length];
    let mut attention_mask = vec![0; input_ids.len()];
    for (batch, &count) in soft_tokens.iter().enumerate() {
        let offset = batch * sequence_length;
        input_ids[offset] = BOS_TOKEN_ID;
        input_ids[offset + 1] = BOI_TOKEN_ID;
        input_ids[offset + 2..offset + 2 + count].fill(IMAGE_TOKEN_ID);
        input_ids[offset + 2 + count] = EOI_TOKEN_ID;
        input_ids[offset + 3 + count] = EOS_TOKEN_ID;
        attention_mask[offset..offset + count + 4].fill(1);
    }
    Ok(TextInputs {
        input_ids,
        attention_mask,
        sequence_length,
    })
}

/// One video sequence, preserving frame order and the pinned no-timestamps policy.
pub fn video_tokens(soft_tokens: &[usize]) -> Result<TextInputs> {
    ensure!(
        (1..=MAX_VIDEO_FRAMES).contains(&soft_tokens.len()),
        "Video requires 1..={MAX_VIDEO_FRAMES} sampled frames"
    );
    ensure!(
        soft_tokens
            .iter()
            .all(|n| (1..=VIDEO_MAX_SOFT_TOKENS).contains(n)),
        "Invalid video soft token count"
    );
    let mut input_ids = vec![BOS_TOKEN_ID];
    for &count in soft_tokens {
        input_ids.push(BOI_TOKEN_ID);
        input_ids.extend(std::iter::repeat_n(VIDEO_TOKEN_ID, count));
        input_ids.push(EOI_TOKEN_ID);
    }
    input_ids.push(EOS_TOKEN_ID);
    ensure!(
        input_ids.len() <= MAX_SEQUENCE_LENGTH,
        "Video exceeds the model context"
    );
    let sequence_length = input_ids.len();
    Ok(TextInputs {
        input_ids,
        attention_mask: vec![1; sequence_length],
        sequence_length,
    })
}

pub fn audio_tokens(soft_tokens: &[usize]) -> Result<TextInputs> {
    ensure!(
        (1..=MAX_IMAGE_BATCH_SIZE).contains(&soft_tokens.len()),
        "Invalid audio token batch size"
    );
    ensure!(
        soft_tokens.iter().all(|n| (1..=750).contains(n)),
        "Invalid audio soft token count for a 30-second segment"
    );
    let sequence_length = soft_tokens.iter().copied().max().unwrap() + 4;
    let mut input_ids = vec![0; soft_tokens.len() * sequence_length];
    let mut attention_mask = vec![0; input_ids.len()];
    for (batch, &count) in soft_tokens.iter().enumerate() {
        let offset = batch * sequence_length;
        input_ids[offset] = BOS_TOKEN_ID;
        input_ids[offset + 1] = BOA_TOKEN_ID;
        input_ids[offset + 2..offset + 2 + count].fill(AUDIO_TOKEN_ID);
        input_ids[offset + 2 + count] = EOA_TOKEN_ID;
        input_ids[offset + 3 + count] = EOS_TOKEN_ID;
        attention_mask[offset..offset + count + 4].fill(1);
    }
    Ok(TextInputs {
        input_ids,
        attention_mask,
        sequence_length,
    })
}

pub struct AudioInputs {
    pub input_features: Vec<f32>,
    pub input_features_mask: Vec<bool>,
    pub num_frames: usize,
    pub num_soft_tokens: usize,
}

// Pinned Gemma4 audio frontend: semicausal framing, periodic Hann, magnitude
// spectrum, HTK mel triangles and additive floor. No power spectrum, dither,
// preemphasis, DC removal or mean/std normalization. See docs/MODELS.md.
struct AudioKernel {
    window: [f32; 320],
    filters: Vec<Vec<(usize, f64)>>,
}
fn audio_kernel() -> &'static AudioKernel {
    static KERNEL: std::sync::OnceLock<AudioKernel> = std::sync::OnceLock::new();
    KERNEL.get_or_init(|| {
        let window = std::array::from_fn(|i| {
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 320.0).cos()) as f32
        });
        let maximum_mel = 2595.0 * (1.0_f64 + 8000.0 / 700.0).log10();
        let points: Vec<_> = (0..130)
            .map(|i| 700.0 * (10.0_f64.powf(maximum_mel * i as f64 / 129.0 / 2595.0) - 1.0))
            .collect();
        let filters = (0..AUDIO_FEATURE_DIM)
            .map(|mel| {
                (0..257)
                    .filter_map(|bin| {
                        let frequency = bin as f64 * 16000.0 / 512.0;
                        let weight = ((frequency - points[mel]) / (points[mel + 1] - points[mel]))
                            .min(
                                (points[mel + 2] - frequency) / (points[mel + 2] - points[mel + 1]),
                            )
                            .max(0.0);
                        (weight > 0.0).then_some((bin, weight))
                    })
                    .collect()
            })
            .collect();
        AudioKernel { window, filters }
    })
}

/// Small fixed radix-2 FFT keeps the production frontend dependency-free.
/// Inputs match NumPy's float32 window multiplication promoted to float64 FFT.
fn magnitude_fft(real: &mut [f64; 512]) -> [f64; 257] {
    let mut imaginary = [0.0_f64; 512];
    let mut reversed = 0;
    for i in 1..512 {
        let mut bit = 256;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if i < reversed {
            real.swap(i, reversed);
        }
    }
    let mut length = 2;
    while length <= 512 {
        let angle = -2.0 * std::f64::consts::PI / length as f64;
        let (step_imag, step_real) = angle.sin_cos();
        for base in (0..512).step_by(length) {
            let (mut wr, mut wi) = (1.0, 0.0);
            for offset in 0..length / 2 {
                let even = base + offset;
                let odd = even + length / 2;
                let tr = wr * real[odd] - wi * imaginary[odd];
                let ti = wr * imaginary[odd] + wi * real[odd];
                real[odd] = real[even] - tr;
                imaginary[odd] = imaginary[even] - ti;
                real[even] += tr;
                imaginary[even] += ti;
                let next_wr = wr * step_real - wi * step_imag;
                wi = wr * step_imag + wi * step_real;
                wr = next_wr;
            }
        }
        length *= 2;
    }
    std::array::from_fn(|i| real[i].hypot(imaginary[i]))
}

pub fn prepare_audio(samples: &[f32]) -> Result<AudioInputs> {
    ensure!((161..=MAX_AUDIO_SAMPLES).contains(&samples.len()), "Audio segments require 161..={MAX_AUDIO_SAMPLES} mono samples at 16000 Hz; split longer audio before embedding");
    ensure!(
        samples.iter().all(|v| v.is_finite()),
        "Audio contains non-finite samples"
    );
    let padded_length = samples.len().div_ceil(128) * 128;
    let num_frames = (padded_length + 160 - 321) / 160 + 1;
    let kernel = audio_kernel();
    let mut input_features = vec![0.0; num_frames * AUDIO_FEATURE_DIM];
    let input_features_mask: Vec<bool> = (0..num_frames)
        .map(|i| i * 160 + 320 < samples.len() + 160)
        .collect();
    for (frame, &valid) in input_features_mask.iter().enumerate() {
        if !valid {
            continue;
        }
        let mut real = [0.0_f64; 512];
        for (i, value) in real[..320].iter_mut().enumerate() {
            let padded_index = frame * 160 + i;
            if padded_index >= 160 {
                if let Some(&sample) = samples.get(padded_index - 160) {
                    *value = f64::from(sample * kernel.window[i]);
                }
            }
        }
        let magnitude = magnitude_fft(&mut real);
        for (mel, weights) in kernel.filters.iter().enumerate() {
            let total: f64 = weights
                .iter()
                .map(|&(bin, weight)| magnitude[bin] * weight)
                .sum();
            input_features[frame * AUDIO_FEATURE_DIM + mel] = (total + 0.001).ln() as f32;
        }
    }
    let num_soft_tokens = input_features_mask
        .iter()
        .step_by(4)
        .filter(|&&valid| valid)
        .count();
    ensure!(
        num_soft_tokens > 0,
        "Audio segment has no valid model tokens"
    );
    Ok(AudioInputs {
        input_features,
        input_features_mask,
        num_frames,
        num_soft_tokens,
    })
}

/// Validate plain document input before tokenization or lazy model loading.
pub fn validate_document_input(title: &str, text: &str) -> Result<()> {
    ensure!(
        !text.trim().is_empty(),
        "Document passage must not be empty"
    );
    ensure!(
        title.len() <= 1024 && text.len() <= MAX_TEXT_BYTES,
        "Document passage exceeds the bounded title/text limits; split it before embedding"
    );
    ensure!(
        !title.contains('\0') && !text.contains('\0'),
        "Document title and passage must not contain NUL"
    );
    Ok(())
}

pub struct QueryTokenizer {
    tokenizer: Tokenizer,
}

impl QueryTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(path).map_err(|error| {
            anyhow::anyhow!("Loading EmbeddingGemma 2 tokenizer {path:?}: {error}")
        })?;
        ensure!(
            tokenizer.get_vocab_size(true) == 262144,
            "Unexpected EmbeddingGemma 2 tokenizer vocabulary size"
        );
        for (token, id) in [
            ("<pad>", 0),
            ("<eos>", 1),
            ("<bos>", 2),
            ("<|image>", BOI_TOKEN_ID),
            ("<|image|>", IMAGE_TOKEN_ID),
            ("<image|>", EOI_TOKEN_ID),
        ] {
            ensure!(
                tokenizer.token_to_id(token) == Some(id as u32),
                "Unexpected token ID for {token}"
            );
        }
        // Reject excessive input explicitly rather than silently truncate a query.
        tokenizer
            .with_truncation(None)
            .map_err(|error| anyhow::anyhow!("Configuring tokenizer: {error}"))?;
        tokenizer.with_padding(None);
        // Literal media markers in plain text are content, never missing features.
        // The template still inserts BOS/EOS; media IDs are built separately.
        tokenizer.set_encode_special_tokens(true);
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<TextInputs> {
        ensure!(!text.trim().is_empty(), "Search query must not be empty");
        ensure!(
            text.len() <= MAX_TEXT_BYTES,
            "Model text input exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
        );
        self.encode_prefixed(format!("{QUERY_PREFIX}{text}"))
    }

    pub fn encode_document(&self, title: &str, text: &str) -> Result<TextInputs> {
        validate_document_input(title, text)?;
        let title = if title.trim().is_empty() {
            "none"
        } else {
            title
        };
        self.encode_prefixed(format!("title: {title} | text: {text}"))
    }

    fn encode_prefixed(&self, text: String) -> Result<TextInputs> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("Tokenizing EmbeddingGemma 2 text: {error}"))?;
        let ids = encoding.get_ids();
        ensure!(
            (2..=MAX_SEQUENCE_LENGTH).contains(&ids.len()),
            "Text exceeds model context of {MAX_SEQUENCE_LENGTH} tokens"
        );
        ensure!(
            ids.first() == Some(&(BOS_TOKEN_ID as u32))
                && ids.last() == Some(&(EOS_TOKEN_ID as u32)),
            "Tokenizer must wrap queries with BOS/EOS"
        );
        ensure!(
            ids.iter().all(|&id| id < 262144),
            "Tokenizer returned an invalid token ID"
        );
        // Defend the graph contract if tokenizer behavior changes.
        ensure!(
            !ids.iter().any(|&id| [258880, 258881, 258884].contains(&id)),
            "Media tokens are not allowed in plain text"
        );
        Ok(TextInputs {
            input_ids: ids.iter().map(|&id| i64::from(id)).collect(),
            attention_mask: encoding
                .get_attention_mask()
                .iter()
                .map(|&v| i64::from(v))
                .collect(),
            sequence_length: ids.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires FRACTAL_SEARCH_TEST_TOKENIZER pointing to the pinned tokenizer.json; no model weights"]
    fn plain_text_media_literals_are_not_feature_placeholders() {
        let path =
            std::env::var_os("FRACTAL_SEARCH_TEST_TOKENIZER").expect("Pinned tokenizer path");
        let tokenizer = QueryTokenizer::from_file(Path::new(&path)).unwrap();
        let literal = "Documenting <|image|> <|audio|> <|video|> and <bos> markers";
        for inputs in [
            tokenizer.encode(literal).unwrap(),
            tokenizer.encode_document("tokens.md", literal).unwrap(),
        ] {
            assert_eq!(inputs.input_ids.first(), Some(&BOS_TOKEN_ID));
            assert_eq!(inputs.input_ids.last(), Some(&EOS_TOKEN_ID));
            assert!(!inputs.input_ids.iter().any(|id| [
                IMAGE_TOKEN_ID,
                AUDIO_TOKEN_ID,
                VIDEO_TOKEN_ID
            ]
            .contains(id)));
        }
    }

    #[test]
    fn processor_geometry_and_bounds() {
        for (width, height, expected) in [
            (800, 600, (912, 672)),
            (600, 800, (672, 912)),
            (1, 1, (768, 768)),
            (1, 65536, (48, 13440)),
            (65536, 1, (13440, 48)),
            (1600, 900, (1056, 576)),
        ] {
            assert_eq!(patch_budget(width, height).unwrap(), MAX_PATCHES);
            assert_eq!(
                resized_dimensions(width, height, MAX_PATCHES).unwrap(),
                expected
            );
        }
        assert!(patch_budget(0, 10).is_err());
        assert!(patch_budget(100001, 1000).is_err());
        assert!(resized_dimensions(10, 10, 1024).is_err());
        assert!(prepare_images(&[]).is_err());
    }

    #[test]
    fn bicubic_uint8_golden_values() {
        let source = RgbImage::from_raw(5, 4, (0..60).collect()).unwrap();
        assert_eq!(
            resize_rgb(&source, 3, 2),
            [10, 11, 12, 15, 16, 17, 20, 21, 22, 37, 38, 39, 42, 43, 44, 47, 48, 49]
        );
        let image = RgbImage::from_pixel(1, 1, image::Rgb([0, 128, 255]));
        assert!(resize_rgb(&image, 7, 9)
            .chunks_exact(3)
            .all(|p| p == [0, 128, 255]));
    }

    #[test]
    fn decoder_rejects_corruption_and_unsupported_depth() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("input.png");
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
    }

    #[test]
    fn patch_layout_rescaling_and_positions() {
        // A full-budget geometry is not resized, allowing exact patch checks.
        let mut image = RgbImage::new(48, 13440);
        image.put_pixel(0, 0, image::Rgb([0, 255, 128]));
        image.put_pixel(16, 0, image::Rgb([255, 0, 255]));
        let inputs = prepare_images(&[image]).unwrap();
        assert_eq!(inputs.spatial_shapes, [[840, 3]]);
        assert_eq!(inputs.pixel_values[..3], [0.0, 1.0, 128.0 / 255.0]);
        assert_eq!(
            inputs.pixel_values[PATCH_DIM..PATCH_DIM + 3],
            [1.0, 0.0, 1.0]
        );
        assert_eq!(inputs.pixel_position_ids[..8], [0, 0, 1, 0, 2, 0, 0, 1]);
        assert_eq!(inputs.num_soft_tokens, [280]);
        let small = RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 128]));
        let inputs = prepare_images(&[small]).unwrap();
        assert_eq!(inputs.num_soft_tokens, [256]);
        assert!(inputs.pixel_position_ids[2304 * 2..]
            .iter()
            .all(|&v| v == -1));
        assert!(inputs.pixel_values[2304 * PATCH_DIM..]
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn independent_image_sequences_have_right_padding_and_matching_placeholders() {
        let inputs = image_tokens(&[2, 1]).unwrap();
        assert_eq!(inputs.sequence_length, 6);
        assert_eq!(
            inputs.input_ids,
            [2, 255999, 258880, 258880, 258882, 1, 2, 255999, 258880, 258882, 1, 0]
        );
        assert_eq!(inputs.attention_mask, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
        assert!(image_tokens(&[]).is_err());
        assert!(image_tokens(&[0]).is_err());
        assert!(image_tokens(&[281]).is_err());
    }

    #[test]
    fn audio_framing_masks_tokens_and_silence_are_exact() {
        for (samples, expected_frames, expected_soft) in [
            (161, 1, 1),
            (16000, 99, 25),
            (32000, 199, 50),
            (480000, 2999, 750),
        ] {
            let inputs = prepare_audio(&vec![0.0; samples]).unwrap();
            assert_eq!(inputs.num_frames, expected_frames);
            assert_eq!(inputs.num_soft_tokens, expected_soft);
            for (frame, &valid) in inputs.input_features_mask.iter().enumerate() {
                for &value in &inputs.input_features
                    [frame * AUDIO_FEATURE_DIM..(frame + 1) * AUDIO_FEATURE_DIM]
                {
                    assert!((value - if valid { 0.001_f32.ln() } else { 0.0 }).abs() < 1e-6);
                }
            }
        }
        assert!(prepare_audio(&[0.0; 160]).is_err());
        assert!(prepare_audio(&vec![0.0; MAX_AUDIO_SAMPLES + 1]).is_err());
        assert!(prepare_audio(&[f32::NAN; 161]).is_err());
        let tokens = audio_tokens(&[2, 1]).unwrap();
        assert_eq!(
            tokens.input_ids,
            [2, 256000, 258881, 258881, 258883, 1, 2, 256000, 258881, 258883, 1, 0]
        );
        assert!(audio_tokens(&[0]).is_err());
    }

    #[test]
    fn fft_uses_magnitude_and_video_uses_its_own_budget() {
        let mut impulse = [0.0; 512];
        impulse[0] = 2.0;
        assert!(magnitude_fft(&mut impulse)
            .iter()
            .all(|v| (v - 2.0).abs() < 1e-10));
        let frames = prepare_video_frames(&[RgbImage::new(1, 1)]).unwrap();
        assert_eq!(frames.max_patches, 1260);
        assert_eq!(frames.num_soft_tokens, [121]);
        let tokens = video_tokens(&[2, 1]).unwrap();
        assert_eq!(
            tokens.input_ids,
            [2, 255999, 258884, 258884, 258882, 255999, 258884, 258882, 1]
        );
        assert!(video_tokens(&[141]).is_err());
        assert!(video_tokens(&vec![1; 33]).is_err());
    }

    #[test]
    fn document_validation_bounds_utf8_and_rejects_nul_without_models() {
        assert!(validate_document_input("", "passage").is_ok());
        assert!(validate_document_input("  ", "passage").is_ok());
        assert!(validate_document_input("<|image|>", "literal <|audio|> text").is_ok());
        assert!(validate_document_input(&"a".repeat(1024), &"b".repeat(MAX_TEXT_BYTES)).is_ok());
        assert!(validate_document_input(&"é".repeat(512), &"猫".repeat(1365)).is_ok());
        assert!(validate_document_input(&"é".repeat(513), "passage").is_err());
        assert!(validate_document_input("title", &"猫".repeat(1366)).is_err());
        for (title, text) in [
            ("title", ""),
            ("title", " \n\t"),
            ("bad\0title", "passage"),
            ("title", "bad\0passage"),
        ] {
            assert!(validate_document_input(title, text).is_err());
        }
    }

    #[test]
    fn document_tokenizer_uses_shared_validation_before_tokenization() {
        let tokenizer = QueryTokenizer {
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
        };
        for (title, text) in [
            ("title".to_owned(), String::new()),
            ("a".repeat(1025), "passage".to_owned()),
            ("title".to_owned(), "b".repeat(MAX_TEXT_BYTES + 1)),
            ("bad\0title".to_owned(), "passage".to_owned()),
            ("title".to_owned(), "bad\0passage".to_owned()),
        ] {
            let expected = validate_document_input(&title, &text)
                .unwrap_err()
                .to_string();
            let actual = tokenizer
                .encode_document(&title, &text)
                .err()
                .expect("invalid input must fail")
                .to_string();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn invalid_queries_fail_before_tokenization() {
        let tokenizer = QueryTokenizer {
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
        };
        assert!(tokenizer.encode(" ").is_err());
        assert!(tokenizer.encode(&"a".repeat(MAX_TEXT_BYTES + 1)).is_err());
    }
}
