//! Synthetic CPU/CUDA parity, warmed latency and actual GPU graph-placement check.
//! cargo run --release -p fractal-search-core --features cuda --example validate_cuda -- MODEL_DIR OUTPUT_DIR [DEVICE_ID]
use anyhow::{ensure, Context, Result};
use fractal_search_core::{
    preprocess,
    runtime::{ExecutionProvider, NativeRuntime},
    Embedder, EMBEDDING_DIM,
};
use image::{Rgb, RgbImage};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};

struct Measurements {
    images: Vec<Vec<f32>>,
    text: Vec<Vec<f32>>,
    documents: Vec<Vec<f32>>,
    audio: Vec<Vec<f32>>,
    video: Vec<Vec<f32>>,
    image_ms: f64,
    text_ms: f64,
    profiles: Vec<PathBuf>,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn measure(
    models: &std::path::Path,
    output: &std::path::Path,
    provider: ExecutionProvider,
    images: &[RgbImage],
    queries: &[&str],
) -> Result<Measurements> {
    let mut runtime = NativeRuntime::with_provider(models.into(), 6, provider)?;
    runtime.enable_profiling(output.into())?;
    // First calls include verification, session startup and GPU memory allocation.
    // Warmed medians below intentionally exclude those one-time costs.
    let mut image_vectors = images
        .iter()
        .map(|image| runtime.embed_image(image))
        .collect::<Result<Vec<_>>>()?;
    image_vectors.extend(runtime.embed_images(images)?);
    let text = queries
        .iter()
        .map(|query| runtime.embed_text(query))
        .collect::<Result<Vec<_>>>()?;
    let documents =
        vec![runtime.embed_document("notes.md", "A colorful landscape beside the water")?];
    let samples: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.25)
        .collect();
    let audio = vec![runtime.embed_audio(&samples)?];
    let video = vec![runtime.embed_video(images)?];
    // Changed shapes/modalities exercise fresh allocations and buffer reuse.
    // Finiteness alone is insufficient: a corrupted first call can also return
    // a normalized but different vector. Compare repeats with the first calls.
    compare(
        &image_vectors[images.len()..],
        &runtime.embed_images(images)?,
    )
    .context("Image batch changed after audio/video inference")?;
    compare(&audio, &[runtime.embed_audio(&samples)?])
        .context("Audio changed after video inference")?;
    compare(&video, &[runtime.embed_video(images)?]).context("Repeated video inference changed")?;
    let mut image_times = Vec::new();
    let mut text_times = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let repeated_image = runtime.embed_image(&images[0])?;
        image_times.push(start.elapsed().as_secs_f64() * 1000.0);
        compare(&image_vectors[..1], &[repeated_image])
            .context("Warmed image differs from first inference")?;
        let start = Instant::now();
        let repeated_text = runtime.embed_text(queries[0])?;
        text_times.push(start.elapsed().as_secs_f64() * 1000.0);
        compare(&text[..1], &[repeated_text])
            .context("Warmed query differs from first inference")?;
    }
    let profiles = runtime.finish_profiling()?;
    Ok(Measurements {
        images: image_vectors,
        text,
        documents,
        audio,
        video,
        image_ms: median(image_times),
        text_ms: median(text_times),
        profiles,
    })
}

fn compare(cpu: &[Vec<f32>], cuda: &[Vec<f32>]) -> Result<Value> {
    ensure!(cpu.len() == cuda.len(), "Embedding output counts differ");
    let mut cosines = Vec::new();
    let mut maximum_error = 0.0_f64;
    for (a, b) in cpu.iter().zip(cuda) {
        ensure!(
            a.len() == EMBEDDING_DIM
                && b.len() == EMBEDDING_DIM
                && a.iter().chain(b).all(|v| v.is_finite()),
            "Invalid provider embedding"
        );
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let norm_a = norm(a);
        let norm_b = norm(b);
        ensure!(
            (norm_a - 1.0).abs() < 1e-5 && (norm_b - 1.0).abs() < 1e-5,
            "Embedding is not normalized"
        );
        let cosine = dot / (norm_a * norm_b);
        ensure!(
            cosine >= 0.9999,
            "Embedding cosine {cosine} is below 0.9999"
        );
        cosines.push(cosine);
        for (a, b) in a.iter().zip(b) {
            maximum_error = maximum_error.max((f64::from(*a) - f64::from(*b)).abs());
        }
    }
    Ok(json!({"cosines":cosines,"maximumAbsoluteError":maximum_error}))
}

fn check_cuda_nodes(profiles: &[PathBuf]) -> Result<Value> {
    ensure!(
        profiles.len() == 3,
        "Expected vision, audio and shared text CUDA profiles"
    );
    let mut summaries = Vec::new();
    for path in profiles {
        let events: Vec<Value> = serde_json::from_slice(&fs::read(path)?)?;
        let mut counts = BTreeMap::<String, usize>::new();
        let mut cuda_matrix_nodes = 0;
        let mut cpu_matrix_nodes = 0;
        for event in events {
            let Some(provider) = event["args"]["provider"].as_str() else {
                continue;
            };
            *counts.entry(provider.into()).or_default() += 1;
            let operation = event["args"]["op_name"]
                .as_str()
                .context("Kernel profile event is missing op_name")?;
            if operation.contains("MatMul")
                || operation == "Gemm"
                || operation.contains("Attention")
            {
                if provider == "CUDAExecutionProvider" {
                    cuda_matrix_nodes += 1;
                } else {
                    cpu_matrix_nodes += 1;
                }
            }
        }
        ensure!(
            cuda_matrix_nodes > 0,
            "No CUDA matrix/attention kernels executed in {path:?}"
        );
        ensure!(
            cpu_matrix_nodes == 0,
            "Model matrix/attention kernels unexpectedly ran on CPU in {path:?}"
        );
        summaries.push(json!({"path":path,"kernelEventsByProvider":counts,"cudaMatrixOrAttentionEvents":cuda_matrix_nodes,"cpuMatrixOrAttentionEvents":cpu_matrix_nodes}));
    }
    Ok(json!(summaries))
}

fn main() -> Result<()> {
    ensure!(
        cfg!(feature = "cuda"),
        "Build this example with --features cuda"
    );
    let mut args = std::env::args_os().skip(1);
    let models = PathBuf::from(args.next().context("Provide model directory")?);
    let output = PathBuf::from(
        args.next()
            .context("Provide a temporary output directory")?,
    );
    let device_id = args
        .next()
        .map(|s| {
            s.into_string()
                .map_err(|_| anyhow::anyhow!("Device ID must be UTF-8"))
        })
        .transpose()?
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(0);
    ensure!(args.next().is_none(), "Unexpected extra argument");
    ensure!(
        !output.exists(),
        "Output directory must be new to protect existing artifacts"
    );
    fs::create_dir_all(&output)?;
    fractal_search_core::initialize_image_decoders();
    let mut images = Vec::new();
    for (index, (width, height)) in [(512, 384), (256, 512), (768, 192)].into_iter().enumerate() {
        let image = RgbImage::from_fn(width, height, |x, y| {
            Rgb([
                ((x + index as u32 * 40) % 256) as u8,
                ((y * 2) % 256) as u8,
                ((x / 8 + y / 8) % 256) as u8,
            ])
        });
        let path = output.join(format!("synthetic-{index}.png"));
        image.save(&path)?;
        images.push(preprocess::decode_image(&path)?);
    }
    let queries = [
        "a red bicycle",
        "blue sky over a green field",
        "colourful geometric patterns",
    ];
    let cpu = measure(&models, &output, ExecutionProvider::Cpu, &images, &queries)?;
    let cuda = measure(
        &models,
        &output,
        ExecutionProvider::Cuda { device_id },
        &images,
        &queries,
    )?;
    let report = json!({
        "deviceId":device_id,
        "embeddingDimension":EMBEDDING_DIM,
        "modelProfile":fractal_search_core::MODEL_PROFILE,
        "outputMode":"indexed",
        "images":compare(&cpu.images, &cuda.images)?,
        "text":compare(&cpu.text, &cuda.text)?,
        "documents":compare(&cpu.documents, &cuda.documents)?,
        "audio":compare(&cpu.audio, &cuda.audio)?,
        "video":compare(&cpu.video, &cuda.video)?,
        "cpuMedianImageMs":cpu.image_ms,"cudaMedianImageMs":cuda.image_ms,
        "cpuMedianTextMs":cpu.text_ms,"cudaMedianTextMs":cuda.text_ms,
        "imageSpeedup":cpu.image_ms/cuda.image_ms,"textSpeedup":cpu.text_ms/cuda.text_ms,
        "cudaProfiles":check_cuda_nodes(&cuda.profiles)?,
        "cpuProfiles":cpu.profiles,
        "timingNote":"Five warmed calls; includes preprocessing and diagnostic profiling overhead; synthetic inputs only"
    });
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
