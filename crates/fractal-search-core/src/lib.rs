//! Shared native multimodal search code. No desktop framework or face models.
#[cfg(feature = "catalog")]
pub mod agent;
#[cfg(feature = "catalog")]
pub mod catalog;
pub mod embeddings;
#[cfg(feature = "media")]
pub mod media;
pub mod preprocess;
#[cfg(feature = "runtime")]
pub mod runtime;

/// Native pinned FP16 graph output width, retained for upstream parity.
pub const NATIVE_EMBEDDING_DIM: usize = 768;
/// Default indexed/query Matryoshka dimension. Not a user-selectable profile.
pub const EMBEDDING_DIM: usize = 512;
pub const MAX_IMAGE_BATCH_SIZE: usize = 8;
pub const NATIVE_MODEL_PROFILE: &str = "embeddinggemma2-daa72c51-fp16-multimodal-v2";
/// Catalog and derived-cache identity includes the leading-dimension projection.
pub const MODEL_PROFILE: &str = "embeddinggemma2-daa72c51-fp16-multimodal-mrl512-v3";
pub const EMBEDDING_PROJECTION: &str = "native768-leading512-l2-v1";
/// A path-free compatibility error that the read-only agent may safely expose.
#[cfg(feature = "catalog")]
#[derive(Debug)]
pub(crate) struct CatalogContractMismatch;
#[cfg(feature = "catalog")]
impl std::fmt::Display for CatalogContractMismatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Incompatible catalog schema, model profile or embedding contract: this build requires FP16-model 512-dimensional Matryoshka vectors; preserve this catalog, use a fresh data directory and reindex in Fractal Search")
    }
}
#[cfg(feature = "catalog")]
impl std::error::Error for CatalogContractMismatch {}

/// Schema-4 lexical field/tokenizer/RRF identity. The legacy name includes the
/// original byte-window parameters; new extraction has its own independent
/// media::TEXT_EXTRACTION_PROFILE and never changes this on-disk schema guard.
pub const TEXT_SEARCH_PROFILE: &str = "filename-newline-passage-v1-4096b-overlap256-icu-split-lower-fold-nostem-nostop-max4096-rrf60-v1";

/// Retrieval ablations use the same catalog, tokenizer, filters and candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextSearchMode {
    Bm25,
    Dense,
    Hybrid,
}

/// Every modality and query returns one finite, unit-length EMBEDDING_DIM vector.
/// Native full-width reference outputs are never part of this storage contract.
pub trait Embedder {
    /// Explicit native tensor preparation capability; custom embedders retain
    /// their raw-input contract. Unsupported prepared calls fail, never retry.
    fn uses_prepared_inputs(&self) -> bool {
        false
    }
    fn preparation_tokenizer(
        &mut self,
    ) -> anyhow::Result<std::sync::Arc<preprocess::QueryTokenizer>> {
        anyhow::bail!("This embedder does not expose a preparation tokenizer")
    }
    fn embed_prepared_images(
        &mut self,
        _inputs: preprocess::ImageInputs,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        anyhow::bail!("Prepared image inference is not implemented")
    }
    fn embed_prepared_audio(
        &mut self,
        _inputs: preprocess::AudioInputs,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("Prepared audio inference is not implemented")
    }
    fn embed_prepared_video(
        &mut self,
        _inputs: Vec<preprocess::ImageInputs>,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("Prepared video inference is not implemented")
    }
    fn embed_prepared_document(
        &mut self,
        _inputs: preprocess::TextInputs,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("Prepared document inference is not implemented")
    }
    /// A verified implementation identity. None disables persistent vector reuse.
    fn embedding_profile(&self) -> Option<String> {
        None
    }
    /// Count the actual formatted document input, including BOS/EOS and title.
    fn document_token_count(&mut self, _title: &str, _text: &str) -> anyhow::Result<usize> {
        anyhow::bail!("Document token counting is not implemented by this embedder")
    }
    fn embed_image(&mut self, image: &image::RgbImage) -> anyhow::Result<Vec<f32>>;
    fn embed_images(&mut self, images: &[image::RgbImage]) -> anyhow::Result<Vec<Vec<f32>>>;
    fn embed_text(&mut self, query: &str) -> anyhow::Result<Vec<f32>>;
    fn embed_document(&mut self, _title: &str, _text: &str) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("Document embedding is not implemented by this embedder")
    }
    fn embed_audio(&mut self, _samples: &[f32]) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("Audio embedding is not implemented by this embedder")
    }
    fn embed_video(&mut self, _frames: &[image::RgbImage]) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("Video embedding is not implemented by this embedder")
    }
}

pub fn initialize_image_decoders() {
    static INITIALIZE: std::sync::Once = std::sync::Once::new();
    INITIALIZE.call_once(libheif_rs::integration::image::register_all_decoding_hooks);
}

#[cfg(feature = "catalog")]
mod index_state;
#[cfg(feature = "catalog")]
mod store;

#[cfg(feature = "catalog")]
mod preparation_pool;
