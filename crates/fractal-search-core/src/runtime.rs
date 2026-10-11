//! Resident EmbeddingGemma 2 ONNX sessions with explicit CPU or CUDA selection.
use crate::{embeddings, preprocess, Embedder, EMBEDDING_DIM, MODEL_PROFILE};
use anyhow::{bail, ensure, Context, Result};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        run_options::{OutputSelector, RunOptions},
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

// ort rc.10 rejects zero dimensions in from_array; allocate real empty tensors.
fn empty_media_features() -> Result<Tensor<f32>> {
    Ok(Tensor::new(
        &ort::memory::Allocator::default(),
        [0usize, preprocess::HIDDEN_DIM],
    )?)
}

fn media_feature_tensor(ids: &[i64], token: i64, features: Vec<f32>) -> Result<Tensor<f32>> {
    let count = ids.iter().filter(|&&id| id == token).count();
    ensure!(
        features.len() == count * preprocess::HIDDEN_DIM && features.iter().all(|v| v.is_finite()),
        "Media token/feature count mismatch for token {token}"
    );
    if count == 0 {
        empty_media_features()
    } else {
        Ok(Tensor::from_array((
            [count, preprocess::HIDDEN_DIM],
            features.into_boxed_slice(),
        ))?)
    }
}

/// Model verification is performed once per process before any session loads.
pub struct NativeRuntime {
    model_dir: PathBuf,
    threads: usize,
    provider: ExecutionProvider,
    profiling_directory: Option<PathBuf>,
    verified: bool,
    vision: Option<Session>,
    audio: Option<Session>,
    text: Option<Session>,
    tokenizer: Option<preprocess::QueryTokenizer>,
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
            audio: None,
            text: None,
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
        for session in [&mut self.vision, &mut self.audio, &mut self.text]
            .into_iter()
            .flatten()
        {
            files.push(PathBuf::from(session.end_profiling()?));
        }
        self.profiling_directory = None;
        Ok(files)
    }
    pub fn loaded(&self) -> bool {
        self.vision.is_some() || self.audio.is_some() || self.text.is_some()
    }
    pub fn text_loaded(&self) -> bool {
        self.text.is_some()
    }

    pub fn verify(&mut self) -> Result<()> {
        if self.verified {
            return Ok(());
        }
        let manifest: Manifest =
            serde_json::from_str(include_str!("../models/embeddinggemma2.json"))?;
        eprintln!(
            "[Model] Verifying {}/{} profile={MODEL_PROFILE} at {:?}",
            manifest.repository, manifest.revision, self.model_dir
        );
        for file in manifest.files {
            let path = self.model_dir.join(&file.path);
            let size = fs::metadata(&path).with_context(|| format!("Missing EmbeddingGemma 2 asset {path:?}; use Fractal Search’s Set up model button or the development scripts/fetch-embeddinggemma2.py downloader"))?.len();
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
                    // CPU shape/control work still runs in a CUDA session, but
                    // its worker pools must sleep while waiting for GPU work.
                    let builder = builder
                        .with_intra_op_spinning(false)?
                        .with_inter_op_spinning(false)?;
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
    fn session(&self, file: &str, inputs: &[&str], outputs: &[&str]) -> Result<Session> {
        let start = Instant::now();
        let mut builder = self.session_builder()?;
        if let Some(directory) = &self.profiling_directory {
            builder = builder.with_profiling(directory.join(format!(
                "{}-{}",
                self.provider.name(),
                file.replace('/', "-")
            )))?;
        }
        let session = builder
            .commit_from_file(self.model_dir.join(file))
            .with_context(|| {
                format!(
                    "Loading native EmbeddingGemma 2 {file} with {} device {:?}; no provider retry",
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
            "Unexpected EmbeddingGemma 2 input contract for {file}"
        );
        ensure!(
            session
                .outputs
                .iter()
                .map(|o| o.name.as_str())
                .collect::<Vec<_>>()
                == outputs,
            "Unexpected EmbeddingGemma 2 output contract for {file}"
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
        self.vision = Some(self.session(
            "onnx/vision_encoder.onnx",
            &["pixel_values", "pixel_position_ids"],
            &["image_features"],
        )?);
        Ok(())
    }
    fn load_audio(&mut self) -> Result<()> {
        if self.audio.is_some() {
            return Ok(());
        }
        self.verify()?;
        self.audio = Some(self.session(
            "onnx/audio_encoder.onnx",
            &["input_features", "input_features_mask"],
            &["audio_features"],
        )?);
        Ok(())
    }
    /// Warm only the shared text backbone; the vision encoder remains lazy.
    pub fn prepare_text(&mut self) -> Result<()> {
        if self.text.is_some() {
            return Ok(());
        }
        self.verify()?;
        self.tokenizer = Some(preprocess::QueryTokenizer::from_file(
            &self.model_dir.join("tokenizer.json"),
        )?);
        self.text = Some(self.session(
            "onnx/model.onnx",
            &[
                "input_ids",
                "attention_mask",
                "image_features",
                "video_features",
                "audio_features",
            ],
            &["last_hidden_state", "sentence_embedding"],
        )?);
        Ok(())
    }

    /// The exported backbone already performs masked mean pooling, projection
    /// and L2 normalization. Use sentence_embedding, never last_hidden_state.
    fn embed_tokens(
        &mut self,
        inputs: preprocess::TextInputs,
        image_features: Vec<f32>,
        video_features: Vec<f32>,
        audio_features: Vec<f32>,
    ) -> Result<Vec<Vec<f32>>> {
        let sequence = inputs.sequence_length;
        ensure!(
            sequence > 0 && inputs.input_ids.len() % sequence == 0,
            "Invalid token batch shape"
        );
        let batch = inputs.input_ids.len() / sequence;
        ensure!(
            batch > 0 && inputs.attention_mask.len() == inputs.input_ids.len(),
            "Invalid attention mask shape"
        );
        let images = media_feature_tensor(
            &inputs.input_ids,
            preprocess::IMAGE_TOKEN_ID,
            image_features,
        )?;
        let video = media_feature_tensor(
            &inputs.input_ids,
            preprocess::VIDEO_TOKEN_ID,
            video_features,
        )?;
        let audio = media_feature_tensor(
            &inputs.input_ids,
            preprocess::AUDIO_TOKEN_ID,
            audio_features,
        )?;
        let ids = Tensor::from_array(([batch, sequence], inputs.input_ids.into_boxed_slice()))?;
        let mask =
            Tensor::from_array(([batch, sequence], inputs.attention_mask.into_boxed_slice()))?;
        // Fetch only the retrieval vector. Returning last_hidden_state would
        // copy the unused token matrix back to CPU on every CUDA inference.
        let options = RunOptions::new()?
            .with_outputs(OutputSelector::no_default().with("sentence_embedding"));
        let outputs = self
            .text
            .as_mut()
            .context("Text session not initialized")?
            .run_with_options(
                ort::inputs![
                    "input_ids" => ids, "attention_mask" => mask, "image_features" => images,
                    "video_features" => video, "audio_features" => audio
                ],
                &options,
            )?;
        let (shape, values) = outputs["sentence_embedding"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [batch as i64, EMBEDDING_DIM as i64],
            "Unexpected EmbeddingGemma 2 sentence_embedding shape {shape:?}"
        );
        values
            .chunks_exact(EMBEDDING_DIM)
            .map(embeddings::normalize)
            .collect()
    }

    pub fn embed_images(&mut self, images: &[image::RgbImage]) -> Result<Vec<Vec<f32>>> {
        let start = Instant::now();
        let inputs = preprocess::prepare_images(images)?;
        let tokens = preprocess::image_tokens(&inputs.num_soft_tokens)?;
        let feature_count: usize = inputs.num_soft_tokens.iter().sum();
        self.prepare_text()?;
        self.load_vision()?;
        let batch = images.len();
        let patches = inputs.max_patches;
        let prepared_ms = start.elapsed().as_secs_f64() * 1000.0;
        let inference = Instant::now();
        let features = self.encode_visual(inputs, batch, feature_count)?;
        let result = self.embed_tokens(tokens, features, Vec::new(), Vec::new())?;
        eprintln!("[Inference] provider={} images={batch} patches={patches} prepare_and_load_ms={prepared_ms:.1} inference_ms={:.1}", self.provider.name(), inference.elapsed().as_secs_f64()*1000.0);
        Ok(result)
    }
    fn encode_visual(
        &mut self,
        inputs: preprocess::ImageInputs,
        batch: usize,
        feature_count: usize,
    ) -> Result<Vec<f32>> {
        self.load_vision()?;
        let patches = inputs.max_patches;
        let pixels = Tensor::from_array((
            [batch, patches, preprocess::PATCH_DIM],
            inputs.pixel_values.into_boxed_slice(),
        ))?;
        let positions = Tensor::from_array((
            [batch, patches, 2],
            inputs.pixel_position_ids.into_boxed_slice(),
        ))?;
        let outputs = self
            .vision
            .as_mut()
            .context("Vision session not initialized")?
            .run(ort::inputs![
                "pixel_values" => pixels, "pixel_position_ids" => positions
            ])?;
        let (shape, values) = outputs["image_features"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [feature_count as i64, preprocess::HIDDEN_DIM as i64],
            "Unexpected EmbeddingGemma 2 vision output shape {shape:?}"
        );
        Ok(values.to_vec())
    }

    pub fn embed_document(&mut self, title: &str, text: &str) -> Result<Vec<f32>> {
        self.prepare_text()?;
        let inputs = self
            .tokenizer
            .as_ref()
            .context("Text tokenizer not initialized")?
            .encode_document(title, text)?;
        self.embed_tokens(inputs, Vec::new(), Vec::new(), Vec::new())?
            .into_iter()
            .next()
            .context("Model returned no document embedding")
    }

    pub fn embed_audio(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let inputs = preprocess::prepare_audio(samples)?;
        let tokens = preprocess::audio_tokens(&[inputs.num_soft_tokens])?;
        self.prepare_text()?;
        self.load_audio()?;
        let features = {
            let data = Tensor::from_array((
                [1, inputs.num_frames, preprocess::AUDIO_FEATURE_DIM],
                inputs.input_features.into_boxed_slice(),
            ))?;
            let mask = Tensor::from_array((
                [1, inputs.num_frames],
                inputs.input_features_mask.into_boxed_slice(),
            ))?;
            let output = self
                .audio
                .as_mut()
                .context("Audio session not initialized")?
                .run(ort::inputs![
                    "input_features" => data, "input_features_mask" => mask
                ])?;
            let (shape, values) = output["audio_features"].try_extract_tensor::<f32>()?;
            ensure!(
                shape.as_ref() == [inputs.num_soft_tokens as i64, preprocess::HIDDEN_DIM as i64],
                "Unexpected audio encoder feature shape {shape:?}"
            );
            values.to_vec()
        };
        self.embed_tokens(tokens, Vec::new(), Vec::new(), features)?
            .into_iter()
            .next()
            .context("Model returned no audio embedding")
    }

    pub fn embed_video(&mut self, frames: &[image::RgbImage]) -> Result<Vec<f32>> {
        ensure!(
            (1..=preprocess::MAX_VIDEO_FRAMES).contains(&frames.len()),
            "Video requires 1..=32 sampled frames"
        );
        self.prepare_text()?;
        let mut features = Vec::new();
        let mut soft_tokens = Vec::with_capacity(frames.len());
        // Encode one frame at a time to keep encoder memory independent of clip
        // length. The shared backbone still embeds the whole temporal sequence.
        for frame in frames {
            let inputs = preprocess::prepare_video_frames(std::slice::from_ref(frame))?;
            let count = inputs.num_soft_tokens[0];
            soft_tokens.push(count);
            features.extend(self.encode_visual(inputs, 1, count)?);
        }
        let tokens = preprocess::video_tokens(&soft_tokens)?;
        self.embed_tokens(tokens, Vec::new(), features, Vec::new())?
            .into_iter()
            .next()
            .context("Model returned no video embedding")
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
    pub fn assets_present(&self) -> bool {
        let manifest: Manifest =
            match serde_json::from_str(include_str!("../models/embeddinggemma2.json")) {
                Ok(manifest) => manifest,
                Err(_) => return false,
            };
        manifest
            .files
            .iter()
            .all(|file| self.model_dir.join(&file.path).is_file())
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
    fn embed_document(&mut self, title: &str, text: &str) -> Result<Vec<f32>> {
        NativeRuntime::embed_document(self, title, text)
    }
    fn embed_audio(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        NativeRuntime::embed_audio(self, samples)
    }
    fn embed_video(&mut self, frames: &[image::RgbImage]) -> Result<Vec<f32>> {
        NativeRuntime::embed_video(self, frames)
    }
    fn embed_text(&mut self, query: &str) -> Result<Vec<f32>> {
        ensure!(!query.trim().is_empty(), "Search query must not be empty");
        ensure!(
            query.len() <= preprocess::MAX_TEXT_BYTES,
            "Model text input exceeds {} UTF-8 bytes",
            preprocess::MAX_TEXT_BYTES
        );
        self.prepare_text()?;
        let inputs = self
            .tokenizer
            .as_ref()
            .context("Text tokenizer not initialized")?
            .encode(query)?;
        let start = Instant::now();
        let result = self
            .embed_tokens(inputs, Vec::new(), Vec::new(), Vec::new())?
            .into_iter()
            .next()
            .context("Model returned no query embedding")?;
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

    #[test]
    fn absent_modalities_have_real_zero_row_tensors_without_model_weights() {
        let tensor = empty_media_features().unwrap();
        match tensor.dtype() {
            ort::value::ValueType::Tensor { ty, shape, .. } => {
                assert_eq!(*ty, ort::tensor::TensorElementType::Float32);
                assert_eq!(shape.as_ref(), &[0, preprocess::HIDDEN_DIM as i64]);
            }
            _ => panic!("Expected tensor"),
        }
    }

    #[test]
    fn media_feature_counts_and_nonfinite_outputs_are_rejected() {
        assert!(media_feature_tensor(
            &[preprocess::AUDIO_TOKEN_ID],
            preprocess::AUDIO_TOKEN_ID,
            Vec::new()
        )
        .is_err());
        assert!(media_feature_tensor(&[], preprocess::VIDEO_TOKEN_ID, vec![1.0; 512]).is_err());
        assert!(media_feature_tensor(
            &[preprocess::IMAGE_TOKEN_ID],
            preprocess::IMAGE_TOKEN_ID,
            vec![f32::NAN; 512]
        )
        .is_err());
    }

    #[test]
    #[ignore = "requires FRACTAL_SEARCH_BATCH_TEST_MODELS; CUDA builds additionally need host CUDA access"]
    fn image_batches_preserve_single_image_embeddings_on_cpu_and_cuda() {
        let models = PathBuf::from(
            std::env::var_os("FRACTAL_SEARCH_BATCH_TEST_MODELS")
                .expect("Provide an existing verified model cache"),
        );
        let providers = vec![ExecutionProvider::Cpu];
        #[cfg(feature = "cuda")]
        let providers = {
            let mut providers = providers;
            providers.push(ExecutionProvider::Cuda { device_id: 0 });
            providers
        };
        for provider in providers {
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
