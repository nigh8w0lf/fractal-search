//! Offline parity against scripts/prepare-embeddinggemma2-fixtures.py.
//! cargo run -p fractal-search-core --example validate_embeddinggemma2 -- [--preprocess-only] [--native-output] MODEL_DIR FIXTURES_JSON...
use anyhow::{bail, ensure, Context, Result};
use fractal_search_core::{
    preprocess, runtime::NativeRuntime, EMBEDDING_DIM, MODEL_PROFILE, NATIVE_EMBEDDING_DIM,
    NATIVE_MODEL_PROFILE,
};
use serde::Deserialize;
use std::{
    fs,
    path::{Component, Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    name: String,
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    audio_file: Option<String>,
    #[serde(default)]
    audio_samples: usize,
    #[serde(default)]
    audio_mask_file: Option<String>,
    #[serde(default)]
    image_files: Vec<String>,
    reference: Vec<Vec<f32>>,
    input_file: String,
    shape: Vec<usize>,
    attention_file: String,
    #[serde(default)]
    spatial_shapes: Vec<[usize; 2]>,
    #[serde(default)]
    position_reference_file: Option<String>,
    #[serde(default)]
    num_soft_tokens: Vec<usize>,
    #[serde(default)]
    token_ids_file: Option<String>,
    #[serde(default)]
    token_shape: Vec<usize>,
    #[serde(rename = "reference_median_ms")]
    _reference_median_ms: f64,
}

fn fixture_path(directory: &Path, name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    ensure!(
        !name.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_))),
        "Fixture paths must be relative and stay in their directory"
    );
    let path = directory.join(path).canonicalize()?;
    ensure!(
        path.starts_with(directory.canonicalize()?),
        "Fixture path escapes its directory"
    );
    Ok(path)
}

fn read_tensor(directory: &Path, name: &str, count: usize, width: usize) -> Result<Vec<u8>> {
    let size = count
        .checked_mul(width)
        .context("Fixture tensor is too large")?;
    ensure!(size <= 512 * 1024 * 1024, "Fixture tensor exceeds 512 MiB");
    let path = fixture_path(directory, name)?;
    ensure!(
        fs::metadata(&path)?.len() == size as u64,
        "Invalid fixture tensor size: {name}"
    );
    let bytes = fs::read(path)?;
    ensure!(
        bytes.len() == size,
        "Fixture tensor changed while reading: {name}"
    );
    Ok(bytes)
}

fn compare_integers(directory: &Path, name: &str, actual: &[i64]) -> Result<()> {
    let bytes = read_tensor(directory, name, actual.len(), 8)?;
    for (i, (actual, expected)) in actual.iter().zip(bytes.chunks_exact(8)).enumerate() {
        let expected = i64::from_le_bytes(expected.try_into().expect("eight bytes"));
        ensure!(
            *actual == expected,
            "Fixture {name} integer mismatch at {i}: {actual} != {expected}"
        );
    }
    Ok(())
}

fn compare_floats(directory: &Path, name: &str, actual: &[f32], tolerance: f32) -> Result<f32> {
    let bytes = read_tensor(directory, name, actual.len(), 4)?;
    let mut maximum = 0.0_f32;
    for (actual, expected) in actual.iter().zip(bytes.chunks_exact(4)) {
        let expected = f32::from_le_bytes(expected.try_into().expect("four bytes"));
        ensure!(
            actual.is_finite() && expected.is_finite(),
            "Non-finite fixture pixels"
        );
        maximum = maximum.max((actual - expected).abs());
    }
    ensure!(
        maximum <= tolerance,
        "Fixture {name} tensor drift {maximum} exceeds {tolerance}"
    );
    Ok(maximum)
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1).peekable();
    let mut preprocess_only = false;
    let mut native_output = false;
    while let Some(arg) = args.peek() {
        if arg == "--preprocess-only" {
            preprocess_only = true;
        } else if arg == "--native-output" {
            native_output = true;
        } else {
            break;
        }
        args.next();
    }
    let embedding_dimension = if native_output {
        NATIVE_EMBEDDING_DIM
    } else {
        EMBEDDING_DIM
    };
    let model = PathBuf::from(args.next().context("Provide the pinned model directory")?);
    let files: Vec<PathBuf> = args.map(PathBuf::from).collect();
    ensure!(!files.is_empty(), "Provide native-input fixture JSON files");
    let tokenizer = preprocess::QueryTokenizer::from_file(&model.join("tokenizer.json"))?;
    let runtime = NativeRuntime::new(model, 6)?;
    fractal_search_core::initialize_image_decoders();
    let mut minimum_cosine = 1.0_f64;
    let mut case_count = 0;
    let mut embedding_count = 0;
    let mut results = Vec::new();
    for file in files {
        ensure!(
            fs::metadata(&file)?.len() <= 4 * 1024 * 1024,
            "Fixture JSON exceeds 4 MiB"
        );
        let fixtures: Vec<Fixture> = serde_json::from_slice(&fs::read(&file)?)?;
        let directory = file.parent().context("Fixture directory missing")?;
        for fixture in fixtures {
            eprintln!("[Parity] {}", fixture.name);
            let mut preprocessing_max_abs_error = 0.0_f32;
            let actual = match fixture.kind.as_str() {
                "text" | "document" => {
                    let query = fixture
                        .text
                        .as_deref()
                        .context("Text fixture is missing raw text")?;
                    let title = fixture.title.as_deref().unwrap_or("none");
                    let inputs = if fixture.kind == "document" {
                        tokenizer.encode_document(title, query)?
                    } else {
                        tokenizer.encode(query)?
                    };
                    ensure!(
                        fixture.shape == [1, inputs.sequence_length],
                        "Text input shape mismatch"
                    );
                    compare_integers(directory, &fixture.input_file, &inputs.input_ids)?;
                    compare_integers(directory, &fixture.attention_file, &inputs.attention_mask)?;
                    if preprocess_only {
                        Vec::new()
                    } else if fixture.kind == "document" {
                        vec![if native_output {
                            runtime.embed_document_native(title, query)?
                        } else {
                            runtime.embed_document(title, query)?
                        }]
                    } else {
                        vec![if native_output {
                            runtime.embed_text_native(query)?
                        } else {
                            runtime.embed_text(query)?
                        }]
                    }
                }
                "image" | "video" => {
                    let images = fixture
                        .image_files
                        .iter()
                        .map(|p| preprocess::decode_image(&fixture_path(directory, p)?))
                        .collect::<Result<Vec<_>>>()?;
                    ensure!(!images.is_empty(), "Image fixture has no source images");
                    let prepared = if fixture.kind == "video" {
                        preprocess::prepare_video_frames(&images)?
                    } else {
                        preprocess::prepare_images(&images)?
                    };
                    ensure!(
                        fixture.shape
                            == [images.len(), prepared.max_patches, preprocess::PATCH_DIM],
                        "Image tensor shape mismatch"
                    );
                    ensure!(
                        prepared.spatial_shapes == fixture.spatial_shapes,
                        "Image grid mismatch"
                    );
                    ensure!(
                        prepared.num_soft_tokens == fixture.num_soft_tokens,
                        "Image soft-token count mismatch"
                    );
                    preprocessing_max_abs_error = compare_floats(
                        directory,
                        &fixture.input_file,
                        &prepared.pixel_values,
                        1.0 / 255.0 + 1e-6,
                    )?;
                    compare_integers(
                        directory,
                        fixture
                            .position_reference_file
                            .as_deref()
                            .context("Image fixture lacks positions")?,
                        &prepared.pixel_position_ids,
                    )?;
                    let inputs = if fixture.kind == "video" {
                        preprocess::video_tokens(&prepared.num_soft_tokens)?
                    } else {
                        preprocess::image_tokens(&prepared.num_soft_tokens)?
                    };
                    let token_batch = if fixture.kind == "video" {
                        1
                    } else {
                        images.len()
                    };
                    ensure!(
                        fixture.token_shape == [token_batch, inputs.sequence_length],
                        "Media token shape mismatch"
                    );
                    compare_integers(
                        directory,
                        fixture
                            .token_ids_file
                            .as_deref()
                            .context("Image fixture lacks token IDs")?,
                        &inputs.input_ids,
                    )?;
                    compare_integers(directory, &fixture.attention_file, &inputs.attention_mask)?;
                    if preprocess_only {
                        Vec::new()
                    } else if fixture.kind == "video" {
                        vec![if native_output {
                            runtime.embed_video_native(&images)?
                        } else {
                            runtime.embed_video(&images)?
                        }]
                    } else {
                        if native_output {
                            runtime.embed_images_native(&images)?
                        } else {
                            runtime.embed_images(&images)?
                        }
                    }
                }
                "audio" => {
                    ensure!(
                        (161..=480_000).contains(&fixture.audio_samples),
                        "Audio fixture sample count is out of bounds"
                    );
                    let bytes = read_tensor(
                        directory,
                        fixture
                            .audio_file
                            .as_deref()
                            .context("Audio fixture lacks PCM samples")?,
                        fixture.audio_samples,
                        4,
                    )?;
                    let samples = bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
                        .collect::<Vec<_>>();
                    ensure!(
                        samples.iter().all(|v| v.is_finite()),
                        "Audio fixture contains non-finite PCM"
                    );
                    let prepared = preprocess::prepare_audio(&samples)?;
                    ensure!(
                        fixture.shape == [1, prepared.num_frames, 128],
                        "Audio feature shape mismatch"
                    );
                    ensure!(
                        fixture.num_soft_tokens == [prepared.num_soft_tokens],
                        "Audio soft-token count mismatch"
                    );
                    preprocessing_max_abs_error = compare_floats(
                        directory,
                        &fixture.input_file,
                        &prepared.input_features,
                        1e-4,
                    )?;
                    let masks = read_tensor(
                        directory,
                        fixture
                            .audio_mask_file
                            .as_deref()
                            .context("Audio fixture lacks frame mask")?,
                        prepared.input_features_mask.len(),
                        1,
                    )?;
                    ensure!(
                        masks
                            .iter()
                            .zip(&prepared.input_features_mask)
                            .all(|(expected, actual)| *expected == u8::from(*actual)),
                        "Audio frame-mask mismatch"
                    );
                    let inputs = preprocess::audio_tokens(&[prepared.num_soft_tokens])?;
                    ensure!(
                        fixture.token_shape == [1, inputs.sequence_length],
                        "Audio token shape mismatch"
                    );
                    compare_integers(
                        directory,
                        fixture
                            .token_ids_file
                            .as_deref()
                            .context("Audio fixture lacks token IDs")?,
                        &inputs.input_ids,
                    )?;
                    compare_integers(directory, &fixture.attention_file, &inputs.attention_mask)?;
                    if preprocess_only {
                        Vec::new()
                    } else {
                        vec![if native_output {
                            runtime.embed_audio_native(&samples)?
                        } else {
                            runtime.embed_audio(&samples)?
                        }]
                    }
                }
                other => bail!("Unknown fixture kind {other:?}"),
            };
            if !preprocess_only {
                ensure!(
                    !fixture.reference.is_empty(),
                    "Fixture {} has no inference reference; use --preprocess-only",
                    fixture.name
                );
                ensure!(
                    actual.len() == fixture.reference.len(),
                    "Fixture {} output count mismatch",
                    fixture.name
                );
            }
            let mut cosines = Vec::new();
            let mut max_abs_errors = Vec::new();
            let mut l2_errors = Vec::new();
            let mut maximum_norm_error = 0.0_f64;
            for (actual, expected) in actual.iter().zip(&fixture.reference) {
                ensure!(
                    actual.len() == embedding_dimension
                        && expected.len() == embedding_dimension
                        && actual.iter().chain(expected).all(|v| v.is_finite()),
                    "Invalid actual or reference embedding in {}",
                    fixture.name
                );
                let dot: f64 = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                let norm = |v: &[f32]| v.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
                let norm_a = norm(actual);
                let norm_b = norm(expected);
                ensure!(
                    (norm_a - 1.0).abs() < 1e-5 && (norm_b - 1.0).abs() < 1e-5,
                    "Non-unit embedding in {}",
                    fixture.name
                );
                let cosine = dot / (norm_a * norm_b);
                ensure!(
                    cosine.is_finite() && cosine >= 0.9999,
                    "Fixture {} cosine {cosine} is below 0.9999",
                    fixture.name
                );
                minimum_cosine = minimum_cosine.min(cosine);
                cosines.push(cosine);
                max_abs_errors.push(
                    actual
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                        .fold(0.0_f64, f64::max),
                );
                l2_errors.push(
                    actual
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                        .sum::<f64>()
                        .sqrt(),
                );
                maximum_norm_error = maximum_norm_error.max((norm_a - 1.0).abs());
                embedding_count += 1;
            }
            results.push(serde_json::json!({
                "name": fixture.name, "kind": fixture.kind, "cosines": cosines,
                "maxAbsErrors": max_abs_errors, "l2Errors": l2_errors,
                "maximumNormError": maximum_norm_error,
                "preprocessingMaxAbsError": preprocessing_max_abs_error,
            }));
            case_count += 1;
        }
    }
    ensure!(case_count > 0, "No fixtures were validated");
    println!(
        "{}",
        serde_json::json!({
            "fixtures":case_count, "embeddings":embedding_count,
            "output_mode":if native_output { "native" } else { "indexed" },
            "embedding_dimension":embedding_dimension,
            "model_profile":if native_output { NATIVE_MODEL_PROFILE } else { MODEL_PROFILE },
            "minimumCosine":if embedding_count > 0 { Some(minimum_cosine) } else { None },
            "inferenceValidated":!preprocess_only, "results":results,
            "scope":"Synthetic image/document/audio/sampled-video HF preprocessing and pinned ONNX parity, not decoder, original-checkpoint or semantic-quality validation"
        })
    );
    Ok(())
}
