//! Resident FG-CLIP2 ONNX sessions with explicit CPU or CUDA selection.
use crate::{embeddings, preprocess, Embedder, EMBEDDING_DIM, MODEL_PROFILE};
use anyhow::{bail, ensure, Context, Result};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
    value::Tensor,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

/// CUDA is an explicit opt-in. Model assets and embedding profiles are identical
/// across providers; selecting CUDA never retries a failed session on the CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    Cuda {
        device_id: i32,
    },
}

impl ExecutionProvider {
    pub fn from_selection(provider: &str, device_id: Option<i32>) -> Result<Self> {
        let provider = match provider {
            "cpu" => {
                ensure!(device_id.is_none(), "--device-id requires --provider cuda");
                Self::Cpu
            }
            "cuda" => Self::Cuda {
                device_id: device_id.unwrap_or(0),
            },
            other => bail!("Unknown inference provider {other:?}; choose cpu or cuda"),
        };
        provider.validate()?;
        Ok(provider)
    }

    fn validate(self) -> Result<()> {
        if let Self::Cuda { device_id } = self {
            ensure!(device_id >= 0, "CUDA device ID must be non-negative");
            ensure!(cfg!(feature = "cuda"), "CUDA was requested but this binary was built without CUDA support; install a CUDA bundle or build with --features cuda");
        }
        Ok(())
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda { .. } => "CUDA",
        }
    }

    pub fn device_id(self) -> Option<i32> {
        match self {
            Self::Cpu => None,
            Self::Cuda { device_id } => Some(device_id),
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    repository: String,
    revision: String,
    files: Vec<ModelFile>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelFile {
    path: String,
    bytes: u64,
    sha256: String,
}

/// Model verification is performed once per process before either session loads.
pub struct NativeRuntime {
    model_dir: PathBuf,
    threads: usize,
    provider: ExecutionProvider,
    profiling_directory: Option<PathBuf>,
    verified: bool,
    vision: Option<Session>,
    text: Option<Session>,
    positions: Vec<f32>,
    tokens: Vec<u8>,
    tokenizer: Option<preprocess::ShortTextTokenizer>,
}

impl NativeRuntime {
    pub fn new(model_dir: PathBuf, threads: usize) -> Result<Self> {
        Self::with_provider(model_dir, threads, ExecutionProvider::Cpu)
    }

    /// Validate configuration without loading model assets or initializing a GPU.
    pub fn with_provider(
        model_dir: PathBuf,
        threads: usize,
        provider: ExecutionProvider,
    ) -> Result<Self> {
        ensure!(
            (1..=64).contains(&threads),
            "ONNX threads must be between 1 and 64"
        );
        provider.validate()?;
        Ok(Self {
            model_dir,
            threads,
            provider,
            profiling_directory: None,
            verified: false,
            vision: None,
            text: None,
            positions: Vec::new(),
            tokens: Vec::new(),
            tokenizer: None,
        })
    }
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }
    /// Opt-in diagnostics for developer benchmarks; never enabled by desktop startup.
    pub fn enable_profiling(&mut self, directory: PathBuf) -> Result<()> {
        ensure!(
            !self.loaded(),
            "Enable profiling before loading model sessions"
        );
        ensure!(
            directory.is_dir(),
            "Profiling directory must already exist: {directory:?}"
        );
        self.profiling_directory = Some(directory);
        Ok(())
    }
    pub fn finish_profiling(&mut self) -> Result<Vec<PathBuf>> {
        ensure!(
            self.profiling_directory.is_some(),
            "Runtime profiling is not enabled"
        );
        let mut files = Vec::new();
        for session in [&mut self.vision, &mut self.text].into_iter().flatten() {
            files.push(PathBuf::from(session.end_profiling()?));
        }
        self.profiling_directory = None;
        Ok(files)
    }
    pub fn loaded(&self) -> bool {
        self.vision.is_some() || self.text.is_some()
    }
    pub fn text_loaded(&self) -> bool {
        self.text.is_some()
    }

    pub fn verify(&mut self) -> Result<()> {
        if self.verified {
            return Ok(());
        }
        let manifest: Manifest = serde_json::from_str(include_str!("../models/fgclip2.json"))?;
        eprintln!(
            "[Model] Verifying {}/{} profile={MODEL_PROFILE} at {:?}",
            manifest.repository, manifest.revision, self.model_dir
        );
        for file in manifest.files {
            let path = self.model_dir.join(&file.path);
            let size = fs::metadata(&path).with_context(|| format!("Missing FG-CLIP2 asset {path:?}; use Fractal Search’s Set up model button or the development scripts/fetch-fgclip2-onnx-probe.py downloader"))?.len();
            ensure!(
                size == file.bytes,
                "Invalid model asset size for {path:?}: expected {}, found {size}",
                file.bytes
            );
            let mut source = fs::File::open(&path)?;
            let mut hash = Sha256::new();
            let mut buffer = vec![0; 1024 * 1024];
            loop {
                let count = source.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
            ensure!(
                format!("{:x}", hash.finalize()) == file.sha256,
                "SHA-256 mismatch for model asset {path:?}"
            );
        }
        self.verified = true;
        Ok(())
    }
    fn session_builder(&self) -> Result<SessionBuilder> {
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.threads)?
            .with_inter_threads(1)?
            .with_parallel_execution(false)?;
        match self.provider {
            ExecutionProvider::Cpu => {
                Ok(
                    builder.with_execution_providers([CPUExecutionProvider::default()
                        .build()
                        .error_on_failure()])?,
                )
            }
            ExecutionProvider::Cuda { device_id } => {
                #[cfg(feature = "cuda")]
                {
                    use ort::execution_providers::CUDAExecutionProvider;
                    // TF32 changes FP32 precision on Ampere and newer GPUs. Keep
                    // the existing catalog profile's precision on every GPU.
                    Ok(builder.with_execution_providers([
                        CUDAExecutionProvider::default()
                            .with_device_id(device_id)
                            .with_tf32(false)
                            .build()
                            .error_on_failure(),
                    ]).with_context(|| format!("Initialize CUDA device {device_id}; the CUDA bundle, NVIDIA driver, CUDA 12 and cuDNN 9 libraries must be available. CPU retry is disabled"))?)
                }
                #[cfg(not(feature = "cuda"))]
                bail!("CUDA device {device_id} requested in a CPU-only build")
            }
        }
    }
    fn session(&self, file: &str, inputs: &[&str], output: &str) -> Result<Session> {
        let start = Instant::now();
        let mut builder = self.session_builder()?;
        if let Some(directory) = &self.profiling_directory {
            builder = builder
                .with_profiling(directory.join(format!("{}-{file}", self.provider.name())))?;
        }
        let session = builder
            .commit_from_file(self.model_dir.join(file))
            .with_context(|| {
                format!(
                    "Loading native FG-CLIP2 {file} with {} device {:?}; no provider retry",
                    self.provider.name(),
                    self.provider.device_id()
                )
            })?;
        ensure!(
            session
                .inputs
                .iter()
                .map(|i| i.name.as_str())
                .collect::<Vec<_>>()
                == inputs,
            "Unexpected FG-CLIP2 input contract for {file}"
        );
        ensure!(
            session.outputs.len() == 1 && session.outputs[0].name == output,
            "Unexpected FG-CLIP2 output contract for {file}"
        );
        eprintln!(
            "[Model] Loaded {file} provider={} device={:?} threads={} load_ms={:.1}",
            self.provider.name(),
            self.provider.device_id(),
            self.threads,
            start.elapsed().as_secs_f64() * 1000.0
        );
        if matches!(self.provider, ExecutionProvider::Cuda { .. }) {
            // ORT deliberately assigns shape calculations to CPU to avoid
            // expensive tiny transfers; this is not a failed CUDA registration.
            eprintln!("[Model] CUDA executes supported graph operations; ONNX Runtime may place shape/control operations on CPU. Image decoding, tokenization and catalog operations remain on CPU");
        }
        Ok(session)
    }
    fn load_vision(&mut self) -> Result<()> {
        if self.vision.is_some() {
            return Ok(());
        }
        self.verify()?;
        let bytes = fs::read(self.model_dir.join("vision_pos_embedding.bin"))?;
        self.positions = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
            .collect();
        ensure!(
            self.positions.len() == 16 * 16 * EMBEDDING_DIM
                && self.positions.iter().all(|x| x.is_finite()),
            "Invalid vision position table"
        );
        self.vision = Some(self.session(
            "visual.onnx",
            &["pixel_values", "pixel_attention_mask", "pos_embed"],
            "image_features",
        )?);
        Ok(())
    }
    /// Allow callers to report startup separately from query inference.
    pub fn prepare_text(&mut self) -> Result<()> {
        if self.text.is_some() {
            return Ok(());
        }
        self.verify()?;
        self.tokenizer = Some(preprocess::ShortTextTokenizer::from_file(
            &self.model_dir.join("tokenizer.json"),
        )?);
        self.tokens = fs::read(self.model_dir.join("text_token_embedding.bin"))?;
        self.text = Some(self.session("text.onnx", &["token_embeds"], "text_features")?);
        Ok(())
    }
    pub fn embed_images(&mut self, images: &[image::RgbImage]) -> Result<Vec<Vec<f32>>> {
        let start = Instant::now();
        let inputs = preprocess::prepare_images(images)?;
        self.load_vision()?;
        let batch = images.len();
        let patches = inputs.max_patches;
        let positions =
            embeddings::position_embeddings(&self.positions, &inputs.spatial_shapes, patches)?;
        let pixels = Tensor::from_array((
            [batch, patches, EMBEDDING_DIM],
            inputs.pixel_values.into_boxed_slice(),
        ))?;
        let mask = Tensor::from_array((
            [batch, patches],
            inputs.pixel_attention_mask.into_boxed_slice(),
        ))?;
        let positions = Tensor::from_array((
            [batch, patches, EMBEDDING_DIM],
            positions.into_boxed_slice(),
        ))?;
        let prepared_ms = start.elapsed().as_secs_f64() * 1000.0;
        let inference = Instant::now();
        let outputs = self.vision.as_mut().context("Vision session not initialized")?.run(ort::inputs!["pixel_values" => pixels, "pixel_attention_mask" => mask, "pos_embed" => positions])?;
        let (shape, values) = outputs["image_features"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [batch as i64, EMBEDDING_DIM as i64],
            "Unexpected FG-CLIP2 vision output shape {shape:?}"
        );
        let result = values
            .chunks_exact(EMBEDDING_DIM)
            .map(embeddings::normalize)
            .collect::<Result<_>>()?;
        eprintln!("[Inference] provider={} images={batch} patches={patches} prepare_and_load_ms={prepared_ms:.1} inference_ms={:.1}", self.provider.name(), inference.elapsed().as_secs_f64()*1000.0);
        Ok(result)
    }
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
    pub fn assets_present(&self) -> bool {
        [
            "visual.onnx",
            "text.onnx",
            "vision_pos_embedding.bin",
            "text_token_embedding.bin",
            "tokenizer.json",
        ]
        .iter()
        .all(|file| self.model_dir.join(file).is_file())
    }
}

impl Embedder for NativeRuntime {
    fn embed_images(&mut self, images: &[image::RgbImage]) -> Result<Vec<Vec<f32>>> {
        NativeRuntime::embed_images(self, images)
    }
    fn embed_image(&mut self, image: &image::RgbImage) -> Result<Vec<f32>> {
        self.embed_images(std::slice::from_ref(image))?
            .into_iter()
            .next()
            .context("Vision returned no embedding")
    }
    fn embed_text(&mut self, query: &str) -> Result<Vec<f32>> {
        ensure!(
            query.len() <= 4096,
            "Model text input exceeds 4096 UTF-8 bytes"
        );
        self.prepare_text()?;
        let ids = self
            .tokenizer
            .as_ref()
            .context("Text tokenizer not initialized")?
            .encode(query, "short")?;
        let values = embeddings::token_embeddings(&self.tokens, &ids)?;
        let tensor = Tensor::from_array((
            [1, preprocess::SHORT_TEXT_LENGTH, EMBEDDING_DIM],
            values.into_boxed_slice(),
        ))?;
        let start = Instant::now();
        let outputs = self
            .text
            .as_mut()
            .context("Text session not initialized")?
            .run(ort::inputs!["token_embeds" => tensor])?;
        let (shape, values) = outputs["text_features"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, EMBEDDING_DIM as i64],
            "Unexpected FG-CLIP2 text output shape {shape:?}"
        );
        let result = embeddings::normalize(values)?;
        eprintln!(
            "[Inference] provider={} query inference_ms={:.1}",
            self.provider.name(),
            start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires FRACTAL_SEARCH_BATCH_TEST_MODELS and host CUDA access"]
    fn image_batches_preserve_single_image_embeddings_on_cpu_and_cuda() {
        let models = PathBuf::from(
            std::env::var_os("FRACTAL_SEARCH_BATCH_TEST_MODELS")
                .expect("Provide an existing verified model cache"),
        );
        for provider in [
            ExecutionProvider::Cpu,
            ExecutionProvider::Cuda { device_id: 0 },
        ] {
            let mut runtime = NativeRuntime::with_provider(models.clone(), 6, provider).unwrap();
            for large in [false, true] {
                let images = (0..crate::MAX_IMAGE_BATCH_SIZE)
                    .map(|index| {
                        let (width, height) = if large {
                            if index % 2 == 0 {
                                (640, 480)
                            } else {
                                (480, 640)
                            }
                        } else if index % 2 == 0 {
                            (48, 32)
                        } else {
                            (1, 1024)
                        };
                        image::RgbImage::from_fn(width, height, |x, y| {
                            image::Rgb([
                                ((x + y + index as u32 * 23) % 256) as u8,
                                ((x * 3 + index as u32 * 11) % 256) as u8,
                                ((y * 7 + index as u32 * 19) % 256) as u8,
                            ])
                        })
                    })
                    .collect::<Vec<_>>();
                let singles = images
                    .iter()
                    .map(|image| runtime.embed_image(image).unwrap())
                    .collect::<Vec<_>>();
                for size in 1..=crate::MAX_IMAGE_BATCH_SIZE {
                    let vectors = runtime.embed_images(&images[..size]).unwrap();
                    assert_eq!(vectors.len(), size);
                    for (index, (actual, expected)) in vectors.iter().zip(&singles).enumerate() {
                        assert_eq!(actual.len(), EMBEDDING_DIM);
                        assert!(actual.iter().all(|value| value.is_finite()));
                        let dot: f64 = actual
                            .iter()
                            .zip(expected)
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum();
                        let norm =
                            |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                        assert!((norm(actual) - 1.0).abs() < 0.001);
                        let cosine = dot / (norm(actual) * norm(expected));
                        assert!(cosine >= 0.9999, "provider={provider:?} large={large} batch={size} image={index} cosine={cosine}");
                    }
                }
            }
        }
    }

    #[test]
    fn provider_options_are_explicit_and_validate_device_ids() {
        assert_eq!(
            ExecutionProvider::from_selection("cpu", None).unwrap(),
            ExecutionProvider::Cpu
        );
        assert!(ExecutionProvider::from_selection("cpu", Some(0)).is_err());
        assert!(ExecutionProvider::from_selection("auto", None).is_err());
        assert!(ExecutionProvider::from_selection("cuda", Some(-1)).is_err());
        assert_eq!(ExecutionProvider::Cpu.device_id(), None);
        #[cfg(feature = "cuda")]
        {
            assert_eq!(
                ExecutionProvider::from_selection("cuda", None).unwrap(),
                ExecutionProvider::Cuda { device_id: 0 }
            );
            assert_eq!(
                ExecutionProvider::from_selection("cuda", Some(3))
                    .unwrap()
                    .device_id(),
                Some(3)
            );
        }
        #[cfg(not(feature = "cuda"))]
        assert!(ExecutionProvider::from_selection("cuda", None)
            .unwrap_err()
            .to_string()
            .contains("without CUDA support"));
    }

    #[test]
    fn construction_is_lazy_and_preserves_cpu_default() {
        let runtime = NativeRuntime::new(PathBuf::from("/missing-model-assets"), 1).unwrap();
        assert_eq!(runtime.provider(), ExecutionProvider::Cpu);
        assert!(!runtime.loaded());
        assert!(NativeRuntime::new(PathBuf::new(), 0).is_err());
        assert!(NativeRuntime::new(PathBuf::new(), 65).is_err());
        assert!(NativeRuntime::with_provider(
            PathBuf::new(),
            1,
            ExecutionProvider::Cuda { device_id: -1 }
        )
        .is_err());
        #[cfg(not(feature = "cuda"))]
        assert!(NativeRuntime::with_provider(
            PathBuf::new(),
            1,
            ExecutionProvider::Cuda { device_id: 0 }
        )
        .is_err());
    }
}
