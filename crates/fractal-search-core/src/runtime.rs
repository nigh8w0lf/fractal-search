//! Resident EmbeddingGemma 2 ONNX sessions with explicit CPU or CUDA selection.
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
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, Instant},
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

fn verify_model_file(model_dir: &Path, file: &ModelFile) -> Result<()> {
    let path = model_dir.join(&file.path);
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
    Ok(())
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

#[path = "runtime_owner.rs"]
mod owner;
pub use owner::{IdlePolicy, RuntimeIdlePolicy};
use owner::{Resources, RuntimeOwner};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModalityDiagnostics {
    pub loaded: bool,
    pub idle_timeout_seconds: Option<f64>,
    pub idle_for_ms: Option<f64>,
    pub load_count: u64,
    pub release_count: u64,
    pub last_load_duration_ms: Option<f64>,
    pub last_reload_duration_ms: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeDiagnostics {
    pub text: ModalityDiagnostics,
    pub vision: ModalityDiagnostics,
    pub audio: ModalityDiagnostics,
}

#[derive(Clone, Copy, Default)]
struct SessionLoad {
    count: u64,
    releases: u64,
    last_duration: Option<Duration>,
    last_reload_duration: Option<Duration>,
}

impl SessionLoad {
    fn record(&mut self, duration: Duration) {
        if self.count != 0 {
            self.last_reload_duration = Some(duration);
        }
        self.count += 1;
        self.last_duration = Some(duration);
    }

    fn diagnostics(
        self,
        loaded: bool,
        policy: IdlePolicy,
        last_used: Option<Instant>,
        now: Instant,
    ) -> ModalityDiagnostics {
        ModalityDiagnostics {
            loaded,
            idle_timeout_seconds: match policy {
                IdlePolicy::KeepWarm => None,
                IdlePolicy::ReleaseAfter(duration) => Some(duration.as_secs_f64()),
            },
            idle_for_ms: last_used
                .map(|last| now.saturating_duration_since(last).as_secs_f64() * 1000.0),
            load_count: self.count,
            release_count: self.releases,
            last_load_duration_ms: self
                .last_duration
                .map(|duration| duration.as_secs_f64() * 1000.0),
            last_reload_duration_ms: self
                .last_reload_duration
                .map(|duration| duration.as_secs_f64() * 1000.0),
        }
    }
}

/// Owns serial admission, lazy initialization, inference, and idle release. The
/// reaper drops sessions only after every active and admitted operation drains.
/// It never owns catalog/settings data. Metadata construction stays model-free.
pub struct NativeRuntime {
    model_dir: PathBuf,
    provider: ExecutionProvider,
    owner: RuntimeOwner<RuntimeSessions>,
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
        let sessions = RuntimeSessions::with_provider(model_dir.clone(), threads, provider)?;
        Ok(Self {
            model_dir,
            provider,
            owner: RuntimeOwner::new(sessions, RuntimeIdlePolicy::default())?,
        })
    }

    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn set_idle_policy(&self, policy: RuntimeIdlePolicy) -> Result<()> {
        self.owner.set_policy(policy)
    }

    pub fn idle_policy(&self) -> Result<RuntimeIdlePolicy> {
        self.owner.inspect(|_, policy, _| policy)
    }

    pub fn diagnostics(&self) -> Result<RuntimeDiagnostics> {
        self.owner.inspect(|sessions, policy, last_used| {
            let loaded = Resources::loaded(sessions);
            let now = Instant::now();
            RuntimeDiagnostics {
                text: sessions.loads[0].diagnostics(loaded[0], policy.text, last_used[0], now),
                vision: sessions.loads[1].diagnostics(loaded[1], policy.vision, last_used[1], now),
                audio: sessions.loads[2].diagnostics(loaded[2], policy.audio, last_used[2], now),
            }
        })
    }

    pub fn loaded(&self) -> bool {
        self.owner
            .inspect(|sessions, _, _| sessions.loaded())
            .expect("Runtime state poisoned; restart the service")
    }

    pub fn text_loaded(&self) -> bool {
        self.owner
            .inspect(|sessions, _, _| sessions.text_loaded())
            .expect("Runtime state poisoned; restart the service")
    }

    pub fn assets_present(&self) -> bool {
        self.owner
            .inspect(|sessions, _, _| sessions.assets_present())
            .expect("Runtime state poisoned; restart the service")
    }

    pub fn verify(&self) -> Result<()> {
        self.owner.execute(owner::NONE, RuntimeSessions::verify)
    }

    /// Profiling keeps sessions resident until finish_profiling is called, so an
    /// idle timer cannot discard a profile before the caller collects its path.
    pub fn enable_profiling(&self, directory: PathBuf) -> Result<()> {
        self.owner
            .execute(owner::NONE, |sessions| sessions.enable_profiling(directory))
    }

    pub fn finish_profiling(&self) -> Result<Vec<PathBuf>> {
        self.owner
            .execute(owner::NONE, RuntimeSessions::finish_profiling)
    }

    /// Warm only the shared text backbone; both media encoders remain lazy.
    pub fn prepare_text(&self) -> Result<()> {
        self.owner
            .execute(owner::TEXT, RuntimeSessions::prepare_text)
    }

    /// Exact formatted token count, requiring only tokenizer.json and no ONNX
    /// model session. Counting does not extend any loaded model's idle timer.
    pub fn document_token_count(&self, title: &str, text: &str) -> Result<usize> {
        self.owner.execute(owner::NONE, |sessions| {
            sessions.prepare_tokenizer()?;
            sessions
                .tokenizer
                .as_ref()
                .context("Text tokenizer not initialized")?
                .document_token_count(title, text)
        })
    }

    pub fn embed_text(&self, query: &str) -> Result<Vec<f32>> {
        self.owner
            .execute(owner::TEXT, |sessions| sessions.embed_text(query))
    }

    pub fn embed_document(&self, title: &str, text: &str) -> Result<Vec<f32>> {
        self.owner
            .execute(owner::TEXT, |sessions| sessions.embed_document(title, text))
    }

    pub fn embed_images(&self, images: &[image::RgbImage]) -> Result<Vec<Vec<f32>>> {
        self.owner
            .execute(owner::VISION, |sessions| sessions.embed_images(images))
    }

    pub fn embed_audio(&self, samples: &[f32]) -> Result<Vec<f32>> {
        self.owner
            .execute(owner::AUDIO, |sessions| sessions.embed_audio(samples))
    }

    pub fn embed_video(&self, frames: &[image::RgbImage]) -> Result<Vec<f32>> {
        self.owner
            .execute(owner::VISION, |sessions| sessions.embed_video(frames))
    }
}

impl Embedder for NativeRuntime {
    fn embed_image(&mut self, image: &image::RgbImage) -> Result<Vec<f32>> {
        NativeRuntime::embed_images(self, std::slice::from_ref(image))?
            .into_iter()
            .next()
            .context("Vision returned no embedding")
    }
    fn embed_images(&mut self, images: &[image::RgbImage]) -> Result<Vec<Vec<f32>>> {
        NativeRuntime::embed_images(self, images)
    }
    fn embed_text(&mut self, query: &str) -> Result<Vec<f32>> {
        NativeRuntime::embed_text(self, query)
    }
    fn embed_document(&mut self, title: &str, text: &str) -> Result<Vec<f32>> {
        NativeRuntime::embed_document(self, title, text)
    }
    fn document_token_count(&mut self, title: &str, text: &str) -> Result<usize> {
        NativeRuntime::document_token_count(self, title, text)
    }
    fn embedding_profile(&self) -> Option<String> {
        Some(MODEL_PROFILE.into())
    }
    fn embed_audio(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        NativeRuntime::embed_audio(self, samples)
    }
    fn embed_video(&mut self, frames: &[image::RgbImage]) -> Result<Vec<f32>> {
        NativeRuntime::embed_video(self, frames)
    }
}

/// Verification is retained across unloads. Only this owner-accessed object can
/// hold or mutate ONNX sessions; callers can never retain a session reference.
struct RuntimeSessions {
    model_dir: PathBuf,
    threads: usize,
    provider: ExecutionProvider,
    profiling_directory: Option<PathBuf>,
    verified: bool,
    vision: Option<Session>,
    audio: Option<Session>,
    text: Option<Session>,
    tokenizer: Option<preprocess::QueryTokenizer>,
    loads: [SessionLoad; 3],
}

impl RuntimeSessions {
    fn with_provider(
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
            loads: [SessionLoad::default(); 3],
        })
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
            verify_model_file(&self.model_dir, &file)?;
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
        let start = Instant::now();
        self.vision = Some(self.session(
            "onnx/vision_encoder.onnx",
            &["pixel_values", "pixel_position_ids"],
            &["image_features"],
        )?);
        self.loads[1].record(start.elapsed());
        Ok(())
    }
    fn load_audio(&mut self) -> Result<()> {
        if self.audio.is_some() {
            return Ok(());
        }
        self.verify()?;
        let start = Instant::now();
        self.audio = Some(self.session(
            "onnx/audio_encoder.onnx",
            &["input_features", "input_features_mask"],
            &["audio_features"],
        )?);
        self.loads[2].record(start.elapsed());
        Ok(())
    }
    fn prepare_tokenizer(&mut self) -> Result<()> {
        if self.tokenizer.is_none() {
            if !self.verified {
                // Token counting needs only the pinned tokenizer, never weight
                // files or native sessions. Verify that asset before using it.
                let manifest: Manifest =
                    serde_json::from_str(include_str!("../models/embeddinggemma2.json"))?;
                let file = manifest
                    .files
                    .iter()
                    .find(|file| file.path == "tokenizer.json")
                    .context("Pinned model manifest does not contain tokenizer.json")?;
                verify_model_file(&self.model_dir, file)?;
            }
            self.tokenizer = Some(preprocess::QueryTokenizer::from_file(
                &self.model_dir.join("tokenizer.json"),
            )?);
        }
        Ok(())
    }

    /// Warm only the shared text backbone; the vision encoder remains lazy.
    pub fn prepare_text(&mut self) -> Result<()> {
        if self.text.is_some() {
            return Ok(());
        }
        self.verify()?;
        self.prepare_tokenizer()?;
        let start = Instant::now();
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
        self.loads[0].record(start.elapsed());
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
        let outputs = self
            .text
            .as_mut()
            .context("Text session not initialized")?
            .run(ort::inputs![
                "input_ids" => ids, "attention_mask" => mask, "image_features" => images,
                "video_features" => video, "audio_features" => audio
            ])?;
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

impl Resources for RuntimeSessions {
    fn loaded(&self) -> [bool; 3] {
        [
            self.text.is_some(),
            self.vision.is_some(),
            self.audio.is_some(),
        ]
    }

    fn release(&mut self, modality: usize) {
        let name = match modality {
            0 => {
                drop(self.text.take());
                "text"
            }
            1 => {
                drop(self.vision.take());
                "vision"
            }
            2 => {
                drop(self.audio.take());
                "audio"
            }
            _ => unreachable!("Unknown model modality"),
        };
        // Reload reopens model files. Revalidate pinned hashes rather than
        // trusting bytes that may have changed while the session was absent.
        self.verified = false;
        self.loads[modality].releases += 1;
        eprintln!(
            "[Model] Released idle {name} session provider={} device={:?}",
            self.provider.name(),
            self.provider.device_id()
        );
    }

    fn release_enabled(&self) -> bool {
        self.profiling_directory.is_none()
    }
}

impl Embedder for RuntimeSessions {
    fn embed_images(&mut self, images: &[image::RgbImage]) -> Result<Vec<Vec<f32>>> {
        RuntimeSessions::embed_images(self, images)
    }
    fn embed_image(&mut self, image: &image::RgbImage) -> Result<Vec<f32>> {
        self.embed_images(std::slice::from_ref(image))?
            .into_iter()
            .next()
            .context("Vision returned no embedding")
    }
    fn embed_document(&mut self, title: &str, text: &str) -> Result<Vec<f32>> {
        RuntimeSessions::embed_document(self, title, text)
    }
    fn embed_audio(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        RuntimeSessions::embed_audio(self, samples)
    }
    fn embed_video(&mut self, frames: &[image::RgbImage]) -> Result<Vec<f32>> {
        RuntimeSessions::embed_video(self, frames)
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
    fn idle_configuration_and_diagnostics_do_not_load_models() {
        let runtime = NativeRuntime::new(PathBuf::from("/missing-model-assets"), 1).unwrap();
        let initial = runtime.diagnostics().unwrap();
        assert!(!initial.text.loaded && !initial.vision.loaded && !initial.audio.loaded);
        assert_eq!(initial.text.idle_timeout_seconds, Some(900.0));
        assert_eq!(initial.vision.idle_timeout_seconds, Some(300.0));
        assert_eq!(initial.audio.idle_timeout_seconds, Some(300.0));
        assert_eq!(initial.text.load_count, 0);
        assert_eq!(initial.text.last_reload_duration_ms, None);
        runtime
            .set_idle_policy(RuntimeIdlePolicy::KEEP_WARM)
            .unwrap();
        assert_eq!(runtime.idle_policy().unwrap(), RuntimeIdlePolicy::KEEP_WARM);
        assert_eq!(
            runtime.diagnostics().unwrap().text.idle_timeout_seconds,
            None
        );
        assert!(!runtime.loaded());
        let error = runtime
            .document_token_count("title", "passage")
            .unwrap_err();
        assert!(error.to_string().contains("tokenizer.json"));
        assert!(!runtime.loaded());
    }

    #[test]
    fn reload_diagnostics_distinguish_first_load_and_reload() {
        let mut load = SessionLoad::default();
        load.record(Duration::from_millis(12));
        assert_eq!(load.count, 1);
        assert_eq!(load.last_reload_duration, None);
        load.releases += 1;
        load.record(Duration::from_millis(7));
        let diagnostic = load.diagnostics(true, IdlePolicy::KeepWarm, None, Instant::now());
        assert_eq!(diagnostic.load_count, 2);
        assert_eq!(diagnostic.release_count, 1);
        assert_eq!(diagnostic.last_load_duration_ms, Some(7.0));
        assert_eq!(diagnostic.last_reload_duration_ms, Some(7.0));
        assert!(serde_json::to_value(diagnostic).unwrap()["lastReloadDurationMs"].is_number());
    }

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
    #[test]
    fn released_sessions_require_asset_reverification_before_reload() {
        let mut sessions = RuntimeSessions::with_provider(
            PathBuf::from("/missing-model-assets"),
            1,
            ExecutionProvider::Cpu,
        )
        .unwrap();
        sessions.verified = true;
        owner::Resources::release(&mut sessions, 0);
        assert!(!sessions.verified);
        assert!(sessions
            .prepare_text()
            .unwrap_err()
            .to_string()
            .contains("Missing EmbeddingGemma 2 asset"));
    }
}
