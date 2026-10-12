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

pub const EMBEDDING_DIM: usize = 768;
pub const MAX_IMAGE_BATCH_SIZE: usize = 8;
pub const MODEL_PROFILE: &str = "embeddinggemma2-daa72c51-fp32-multimodal-v1";
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

pub trait Embedder {
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
