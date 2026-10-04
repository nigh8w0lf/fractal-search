//! Shared native image search code. No desktop framework or face models.
#[cfg(feature = "catalog")]
pub mod agent;
#[cfg(feature = "catalog")]
pub mod catalog;
pub mod embeddings;
pub mod preprocess;
#[cfg(feature = "runtime")]
pub mod runtime;

pub const EMBEDDING_DIM: usize = 768;
pub const MODEL_PROFILE: &str = "fgclip2-base-33f3b9d4-dynamic-torch291-short64-v1";

pub trait Embedder {
    fn embed_image(&mut self, image: &image::RgbImage) -> anyhow::Result<Vec<f32>>;
    fn embed_text(&mut self, query: &str) -> anyhow::Result<Vec<f32>>;
}

pub fn initialize_image_decoders() {
    static INITIALIZE: std::sync::Once = std::sync::Once::new();
    INITIALIZE.call_once(libheif_rs::integration::image::register_all_decoding_hooks);
}

#[cfg(feature = "catalog")]
mod store;
