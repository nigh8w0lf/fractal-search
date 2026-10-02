//! Manual parity check against fixtures from scripts/prepare-fgclip2-onnx-fixtures.py.
//! Usage: cargo run -p fractal-search-core --example validate_fgclip2 -- MODEL_DIR FIXTURES_JSON...
use anyhow::{bail, ensure, Context, Result};
use fractal_search_core::{preprocess, runtime::NativeRuntime, Embedder, EMBEDDING_DIM};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    name: String,
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image_files: Vec<String>,
    reference: Vec<Vec<f32>>,
    #[serde(rename = "input_file")]
    _input_file: String,
    #[serde(rename = "shape")]
    _shape: Vec<usize>,
    #[serde(default, rename = "mask_file")]
    _mask_file: Option<String>,
    #[serde(default, rename = "spatial_shapes")]
    _spatial_shapes: Vec<[usize; 2]>,
    #[serde(default, rename = "position_reference_file")]
    _position_reference_file: Option<String>,
    #[serde(rename = "reference_median_ms")]
    _reference_median_ms: f64,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model = PathBuf::from(args.next().context("Provide the pinned model directory")?);
    let files: Vec<PathBuf> = args.map(PathBuf::from).collect();
    ensure!(!files.is_empty(), "Provide native-input fixture JSON files");
    let mut runtime = NativeRuntime::new(model, 6)?;
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
            let actual = match fixture.kind.as_str() {
                "text" => vec![runtime.embed_text(
                    fixture
                        .text
                        .as_deref()
                        .context("Native text fixture is missing raw text")?,
                )?],
                "image" => {
                    let images = fixture
                        .image_files
                        .iter()
                        .map(|p| preprocess::decode_image(&directory.join(p)))
                        .collect::<Result<Vec<_>>>()?;
                    ensure!(
                        !images.is_empty(),
                        "Native image fixture has no source images"
                    );
                    runtime.embed_images(&images)?
                }
                other => bail!("Unknown fixture kind {other:?}"),
            };
            ensure!(
                actual.len() == fixture.reference.len(),
                "Fixture {} output count mismatch",
                fixture.name
            );
            let mut cosines = Vec::new();
            let mut max_abs_errors = Vec::new();
            let mut l2_errors = Vec::new();
            let mut maximum_norm_error = 0.0_f64;
            for (actual, expected) in actual.iter().zip(&fixture.reference) {
                ensure!(
                    actual.len() == EMBEDDING_DIM
                        && expected.len() == EMBEDDING_DIM
                        && actual.iter().chain(expected).all(|v| v.is_finite()),
                    "Invalid actual or reference embedding in {}",
                    fixture.name
                );
                let dot: f64 = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                let norm_a = actual
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm_b = expected
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt();
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
                "name": fixture.name,
                "kind": fixture.kind,
                "cosines": cosines,
                "maxAbsErrors": max_abs_errors,
                "l2Errors": l2_errors,
                "maximumNormError": maximum_norm_error,
            }));
            case_count += 1;
        }
    }
    println!(
        "{}",
        serde_json::json!({"fixtures":case_count,"embeddings":embedding_count,"minimumCosine":minimum_cosine,"results":results})
    );
    Ok(())
}
