//! LanceDB catalog and incremental indexing. Originals are never modified.
mod preparation;
use crate::index_state::{
    hash_file, hash_file_with_cancel, is_racy, FileStamp as Stamp, FileState, IndexState,
    PassageLocation,
};
use crate::media::{self, MediaKind, PreparedMedia};
use crate::preparation_pool::PreparationPool;
#[cfg(test)]
use crate::store::MAX_TEXT_EXCERPT_BYTES;
use crate::store::{asset_filter, quote, Record, Store};
use crate::{
    preprocess::decode_image, Embedder, EMBEDDING_DIM, MAX_IMAGE_BATCH_SIZE, MODEL_PROFILE,
};
use anyhow::{bail, ensure, Context, Result};
use image::{codecs::webp::WebPEncoder, ExtendedColorType, RgbImage};
use preparation::{FatalPreparation, FileData, FileTicket, NativeInput};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::{Instant, UNIX_EPOCH},
};
use url::Url;
use uuid::Uuid;
use walkdir::WalkDir;

const MAX_FILES: usize = 250_000;
const MAX_RESULTS: usize = 1000;
const MAX_REPORTED_FAILURES: usize = 100;
const MAX_FAILURE_MESSAGE_BYTES: usize = 2048;

/// Rebuilding text is explicit; it replaces each complete file atomically.
#[derive(Clone, Copy, Debug, Default)]
pub struct IndexOptions {
    pub reindex_text: bool,
    /// Read and hash even stable timestamps; detects deliberately restored mtimes.
    pub verify_content: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    pub id: i64,
    pub directory: String,
    pub asset_count: usize,
    pub file_count: usize,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogStatus {
    pub asset_count: usize,
    pub file_count: usize,
    pub sources: Vec<Source>,
    pub model_profile: &'static str,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Asset {
    pub asset_id: String,
    pub filename: String,
    pub path: String,
    pub thumbnail_url: String,
    pub width: u32,
    pub height: u32,
    pub media_kind: String,
    pub segment_index: u32,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub text_excerpt: Option<String>,
    pub similarity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_byte: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_byte: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub additional_matches: Vec<Asset>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Page {
    pub items: Vec<Asset>,
    pub total: usize,
    pub offset: usize,
    pub limit: usize,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexFailure {
    pub path: String,
    pub error: String,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexProgress {
    pub stage: &'static str,
    pub directory: String,
    pub current: usize,
    pub total: usize,
    pub indexed: usize,
    pub unchanged: usize,
    pub failed: usize,
    pub filename: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<IndexFailure>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexResult {
    pub indexed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub cancelled: bool,
    pub failed: usize,
    pub failures: Vec<IndexFailure>,
    pub failures_omitted: usize,
}

impl IndexProgress {
    fn record_failure(
        &mut self,
        path: &Path,
        error: anyhow::Error,
        failures: &mut Vec<IndexFailure>,
    ) {
        let mut message = format!("{error:#}");
        eprintln!("[Index] Skipped file {path:?}: {message}");
        if message.len() > MAX_FAILURE_MESSAGE_BYTES {
            let mut end = MAX_FAILURE_MESSAGE_BYTES;
            while !message.is_char_boundary(end) {
                end -= 1;
            }
            message.truncate(end);
            message.push_str("… [message truncated; full error in backend log]");
        }
        self.current += 1;
        self.failed += 1;
        self.failure = Some(IndexFailure {
            path: path.display().to_string(),
            error: message,
        });
        if failures.len() < MAX_REPORTED_FAILURES {
            failures.push(self.failure.as_ref().expect("recorded failure").clone());
        }
    }

    fn result(&self, removed: usize, cancelled: bool, failures: Vec<IndexFailure>) -> IndexResult {
        IndexResult {
            indexed: self.indexed,
            unchanged: self.unchanged,
            removed,
            cancelled,
            failed: self.failed,
            failures_omitted: self.failed - failures.len(),
            failures,
        }
    }
}

struct PendingImage {
    path: PathBuf,
    before: Stamp,
    existing: Option<Record>,
    decode_ms: f64,
    start: Instant,
    content_hash: String,
    embedding_profile: Option<String>,
    verify_after: bool,
    thumbnail: Vec<u8>,
}

struct IndexBatch {
    source_id: i64,
    images: Vec<RgbImage>,
    inputs: Vec<crate::preprocess::ImageInputs>,
    pending: Vec<PendingImage>,
}

impl IndexBatch {
    fn flush<E: Embedder>(
        &mut self,
        catalog: &Catalog,
        embedder: &mut E,
        state: &mut IndexProgress,
        failures: &mut Vec<IndexFailure>,
        cancelled: &impl Fn() -> bool,
        progress: &mut impl FnMut(&IndexProgress) -> Result<()>,
    ) -> Result<bool> {
        if self.images.is_empty() {
            return Ok(cancelled());
        }
        if cancelled() {
            return Ok(true);
        }
        let inferred = if embedder.uses_prepared_inputs() {
            embedder.embed_prepared_images(preparation::combine_images(std::mem::take(
                &mut self.inputs,
            ))?)
        } else {
            embedder.embed_images(&self.images)
        };
        let vectors = inferred.with_context(|| {
            format!(
                "Embed image batch of {} starting at {:?}",
                self.images.len(),
                self.pending[0].path
            )
        })?;
        ensure!(
            vectors.len() == self.images.len(),
            "Image batch returned {} embeddings for {} images",
            vectors.len(),
            self.images.len()
        );
        // Validate the entire inference result before committing any of it.
        for vector in &vectors {
            validate_embedding(vector)?;
        }
        let images = std::mem::take(&mut self.images);
        let pending = std::mem::take(&mut self.pending);
        for ((pending, image), vector) in pending.into_iter().zip(images).zip(vectors) {
            state.failure = None;
            state.filename = pending
                .path
                .file_name()
                .context("Image filename missing")?
                .to_string_lossy()
                .into_owned();
            catalog.finish_index_image(self.source_id, pending, &image, vector, state, failures)?;
            progress(state)?;
            // Native inference cannot be interrupted. Finish one image commit
            // even if cancelled during inference, then discard the rest of the
            // computed batch and never prune unvisited entries.
            if cancelled() {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[derive(Debug)]
pub struct Catalog {
    store: Store,
    thumbnails: PathBuf,
    previews: PathBuf,
    index_state: IndexState,
}

fn text_path(path: &Path) -> Result<&str> {
    path.to_str()
        .context("Media search requires UTF-8 filesystem paths")
}
// Schema 4's private modified string is opaque to readers. A per-file commit
// nonce binds derived state to authoritative rows without a schema migration.
fn raw_modified(value: &str) -> &str {
    value.split_once('@').map_or(value, |(stamp, _)| stamp)
}
fn generation_modified(stamp: &Stamp) -> String {
    format!("{}@{}", stamp.modified, Uuid::new_v4())
}
fn file_url(path: &Path) -> Result<String> {
    Url::from_file_path(path)
        .map(|u| u.to_string())
        .map_err(|_| anyhow::anyhow!("Cannot create file URL for {path:?}"))
}
fn stamp(path: &Path) -> Result<Stamp> {
    let metadata = fs::metadata(path).with_context(|| format!("Inspect media {path:?}"))?;
    ensure!(metadata.is_file(), "Media is not a regular file: {path:?}");
    let time = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .context("Media modification time predates Unix epoch")?;
    Ok(Stamp {
        bytes: metadata.len(),
        modified: format!("{}.{}", time.as_secs(), time.subsec_nanos()),
    })
}
fn validate_page(offset: usize, limit: usize) -> Result<()> {
    ensure!(
        (1..=100).contains(&limit) && offset <= MAX_FILES,
        "Page limit must be 1..=100 and offset <= {MAX_FILES}"
    );
    Ok(())
}
pub(crate) fn validate_embedding(values: &[f32]) -> Result<()> {
    ensure!(
        values.len() == EMBEDDING_DIM,
        "Search embedding must have {EMBEDDING_DIM} dimensions, got {}",
        values.len()
    );
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "Search embedding contains a non-finite value"
    );
    let norm: f64 = values
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        (norm - 1.0).abs() < 0.0001,
        "Search embedding must be normalized (norm={norm})"
    );
    Ok(())
}
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!("{}.part", Uuid::new_v4()));
    let result = (|| {
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        output.write_all(bytes)?;
        output.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        if let Err(error) = fs::remove_file(&temporary) {
            if error.kind() != std::io::ErrorKind::NotFound {
                eprintln!("[Cache] Cannot remove failed temporary {temporary:?}: {error}");
            }
        }
    }
    result
}
fn bounded_dimensions(width: u32, height: u32, max_side: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= max_side {
        return (width, height);
    }
    let scale = |side| (u64::from(side) * u64::from(max_side) / u64::from(longest)).max(1) as u32;
    (scale(width), scale(height))
}

fn cache_is_current(row: &Record) -> bool {
    row.thumbnail.as_ref().is_none_or(|cache| {
        image::image_dimensions(cache)
            .is_ok_and(|dimensions| dimensions == bounded_dimensions(row.width, row.height, 384))
    })
}

fn bounded_thumbnail(image: &RgbImage, max_side: u32) -> RgbImage {
    let (width, height) = bounded_dimensions(image.width(), image.height(), max_side);
    // imageops::thumbnail takes exact output dimensions; it does not fit an
    // image into a bounding box or preserve its aspect ratio automatically.
    image::imageops::thumbnail(image, width, height)
}

fn thumbnail(image: &RgbImage) -> Result<Vec<u8>> {
    let pixels = bounded_thumbnail(image, 384);
    let mut bytes = Vec::new();
    WebPEncoder::new_lossless(&mut bytes).encode(
        pixels.as_raw(),
        pixels.width(),
        pixels.height(),
        ExtendedColorType::Rgb8,
    )?;
    Ok(bytes)
}

impl Catalog {
    pub fn open(data_dir: &Path) -> Result<Self> {
        fs::create_dir_all(data_dir)
            .with_context(|| format!("Create search data directory {data_dir:?}"))?;
        let data_dir = data_dir.canonicalize()?;
        let store = Store::open(&data_dir, false)?;
        let thumbnails = data_dir.join("thumbnails");
        let previews = data_dir.join("previews");
        fs::create_dir_all(&thumbnails)?;
        fs::create_dir_all(&previews)?;
        Ok(Self {
            store,
            thumbnails,
            previews,
            index_state: IndexState::new(&data_dir),
        })
    }
    pub fn status(&self) -> Result<CatalogStatus> {
        let (asset_count, file_count, rows) = self.store.status()?;
        let mut sources = rows
            .into_iter()
            .map(|(row, asset_count, file_count)| Source {
                id: row.source_id,
                directory: row.path,
                asset_count,
                file_count,
            })
            .collect::<Vec<_>>();
        sources.sort_by(|a, b| a.directory.cmp(&b.directory));
        Ok(CatalogStatus {
            asset_count,
            file_count,
            sources,
            model_profile: MODEL_PROFILE,
        })
    }
    fn asset(row: Record) -> (Asset, Option<PathBuf>) {
        (
            Asset {
                asset_id: row.asset_id,
                filename: row.filename,
                path: row.path,
                thumbnail_url: String::new(),
                width: row.width,
                height: row.height,
                media_kind: row.media_kind,
                segment_index: row.segment_index,
                start_ms: row.start_ms,
                end_ms: row.end_ms,
                text_excerpt: row.text_excerpt,
                similarity: None,
                hybrid_score: None,
                start_byte: None,
                end_byte: None,
                start_line: None,
                end_line: None,
                additional_matches: Vec::new(),
            },
            row.thumbnail.map(PathBuf::from),
        )
    }
    fn located_asset(&self, row: Record) -> Result<(Asset, Option<PathBuf>)> {
        let location = self.index_state.location(&row)?;
        let (mut asset, thumbnail) = Self::asset(row);
        if let Some(location) = location {
            asset.start_byte = location.start_byte;
            asset.end_byte = location.end_byte;
            asset.start_line = location.start_line;
            asset.end_line = location.end_line;
        }
        Ok((asset, thumbnail))
    }
    fn complete_asset(&self, (mut asset, thumbnail): (Asset, Option<PathBuf>)) -> Result<Asset> {
        let Some(thumbnail) = thumbnail else {
            return Ok(asset);
        };
        ensure!(
            thumbnail.parent() == Some(self.thumbnails.as_path()) && thumbnail.is_file(),
            "Thumbnail is unavailable for {}; rescan its folder",
            asset.filename
        );
        let dimensions = image::image_dimensions(&thumbnail)
            .with_context(|| format!("Read cached thumbnail dimensions {thumbnail:?}"))?;
        ensure!(
            dimensions == bounded_dimensions(asset.width, asset.height, 384),
            "Cached thumbnail dimensions are invalid; rescan its source folder"
        );
        asset.thumbnail_url = file_url(&thumbnail)?;
        Ok(asset)
    }
    pub fn browse(&self, offset: usize, limit: usize) -> Result<Page> {
        self.browse_source(offset, limit, None)
    }
    pub fn source_directory(&self, source_id: i64) -> Result<PathBuf> {
        ensure!(source_id > 0, "Source ID must be positive");
        let record = self
            .store
            .get(
                &format!("kind = 'source' AND source_id = {source_id}"),
                false,
            )?
            .context("Indexed folder does not exist")?;
        Ok(PathBuf::from(record.path))
    }
    pub fn browse_source(
        &self,
        offset: usize,
        limit: usize,
        source_id: Option<i64>,
    ) -> Result<Page> {
        self.browse_filtered(offset, limit, source_id, None)
    }
    pub fn browse_filtered(
        &self,
        offset: usize,
        limit: usize,
        source_id: Option<i64>,
        media_kind: Option<MediaKind>,
    ) -> Result<Page> {
        validate_page(offset, limit)?;
        if let Some(id) = source_id {
            self.source_directory(id)?;
        }
        let (total, rows) = self
            .store
            .browse_filtered(offset, limit, source_id, media_kind)?;
        let items = rows
            .into_iter()
            .map(|r| self.complete_asset(self.located_asset(r)?))
            .collect::<Result<_>>()?;
        Ok(Page {
            items,
            total,
            offset,
            limit,
        })
    }
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<Asset>> {
        self.search_source(query_embedding, limit, None)
    }
    pub fn search_source(
        &self,
        query_embedding: &[f32],
        limit: usize,
        source_id: Option<i64>,
    ) -> Result<Vec<Asset>> {
        self.search_results("", query_embedding, limit, source_id, None)
    }
    pub fn search_filtered(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        source_id: Option<i64>,
        media_kind: Option<MediaKind>,
    ) -> Result<Vec<Asset>> {
        crate::agent::validate_query(query)?;
        self.search_results(query, query_embedding, limit, source_id, media_kind)
    }
    pub fn search_filtered_grouped(
        &self,
        query: &str,
        embedding: &[f32],
        limit: usize,
        source_id: Option<i64>,
        media_kind: Option<MediaKind>,
        group_text_by_file: bool,
    ) -> Result<Vec<Asset>> {
        crate::agent::validate_query(query)?;
        ensure!(
            (1..=MAX_RESULTS).contains(&limit),
            "Invalid search result limit"
        );
        if let Some(id) = source_id {
            self.source_directory(id)?;
        }
        self.store
            .search_filtered_grouped(
                query,
                embedding,
                limit,
                source_id,
                media_kind,
                group_text_by_file,
            )?
            .into_iter()
            .map(|hit| self.result_asset(hit))
            .collect()
    }
    fn result_asset(&self, hit: crate::store::SearchHit) -> Result<Asset> {
        let mut asset = self.complete_asset(self.located_asset(hit.row)?)?;
        asset.similarity = Some((1.0 - hit.distance).clamp(-1.0, 1.0));
        asset.hybrid_score = hit.hybrid_score;
        asset.additional_matches = hit
            .additional_matches
            .into_iter()
            .map(|other| self.result_asset(other))
            .collect::<Result<_>>()?;
        Ok(asset)
    }

    /// Model-free BM25, exact dense, and unexpanded RRF ablations.
    pub fn search_text_evaluation(
        &self,
        query: &str,
        embedding: &[f32],
        limit: usize,
        source_id: Option<i64>,
        mode: crate::TextSearchMode,
    ) -> Result<Vec<Asset>> {
        crate::agent::validate_query(query)?;
        ensure!(
            (1..=MAX_RESULTS).contains(&limit),
            "Invalid evaluation result limit"
        );
        if let Some(id) = source_id {
            self.source_directory(id)?;
        }
        self.store
            .search_text_evaluation(query, embedding, limit, source_id, mode)?
            .into_iter()
            .map(|hit| {
                let mut asset = self.complete_asset(self.located_asset(hit.row)?)?;
                asset.similarity = if mode == crate::TextSearchMode::Bm25 {
                    None
                } else {
                    Some((1.0 - hit.distance).clamp(-1.0, 1.0))
                };
                asset.hybrid_score = hit.hybrid_score;
                Ok(asset)
            })
            .collect()
    }

    fn search_results(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        source_id: Option<i64>,
        media_kind: Option<MediaKind>,
    ) -> Result<Vec<Asset>> {
        ensure!(
            (1..=MAX_RESULTS).contains(&limit),
            "Search result limit must be between 1 and {MAX_RESULTS}"
        );
        if let Some(id) = source_id {
            self.source_directory(id)?;
        }
        let hits =
            self.store
                .search_filtered(query, query_embedding, limit, source_id, media_kind)?;
        hits.into_iter()
            .map(|hit| {
                let mut asset = self.complete_asset(self.located_asset(hit.row)?)?;
                asset.similarity = Some((1.0 - hit.distance).clamp(-1.0, 1.0));
                asset.hybrid_score = hit.hybrid_score;
                Ok(asset)
            })
            .collect()
    }
    /// Remove the source and its images in one Lance commit, then delete caches.
    pub fn remove_source(&mut self, source_id: i64) -> Result<usize> {
        self.source_directory(source_id)?;
        let rows = self
            .store
            .list(&asset_filter(Some(source_id)), MAX_FILES + 1, false)?;
        ensure!(rows.len() <= MAX_FILES, "Source exceeds segment limit");
        self.store.delete(&format!("source_id = {source_id}"))?;
        for row in &rows {
            if let Some(cache) = &row.thumbnail {
                self.remove_cache(Path::new(cache))?;
            }
            self.remove_cache(&self.previews.join(format!("{}.png", row.asset_id)))?;
        }
        self.store.checkpoint()?;
        Ok(rows.len())
    }
    fn source(&mut self, directory: &Path) -> Result<i64> {
        let sources = self.store.list("kind = 'source'", MAX_FILES + 1, false)?;
        ensure!(sources.len() < MAX_FILES, "Catalog source limit reached");
        let mut id = 1;
        for source in sources {
            let other = Path::new(&source.path);
            if directory == other {
                return Ok(source.source_id);
            }
            ensure!(!directory.starts_with(other) && !other.starts_with(directory), "Folder overlaps an existing source: {}. Rescan the existing source or choose a separate folder", source.path);
            ensure!(
                source.source_id > 0 && source.source_id < i64::from(i32::MAX),
                "Invalid or exhausted catalog source IDs"
            );
            id = id.max(source.source_id + 1);
        }
        self.store.upsert(&Record {
            kind: "source".into(),
            asset_id: format!("source_{}", Uuid::new_v4()),
            source_id: id,
            path: text_path(directory)?.to_owned(),
            ..Record::default()
        })?;
        Ok(id)
    }
    pub fn index<E: Embedder>(
        &mut self,
        directory: &Path,
        embedder: &mut E,
        cancelled: impl Fn() -> bool,
        progress: impl FnMut(&IndexProgress) -> Result<()>,
    ) -> Result<IndexResult> {
        self.index_with_batch_size(directory, embedder, 1, cancelled, progress)
    }

    pub fn index_with_batch_size<E: Embedder>(
        &mut self,
        directory: &Path,
        embedder: &mut E,
        batch_size: usize,
        cancelled: impl Fn() -> bool,
        progress: impl FnMut(&IndexProgress) -> Result<()>,
    ) -> Result<IndexResult> {
        self.index_with_options(
            directory,
            embedder,
            batch_size,
            IndexOptions::default(),
            cancelled,
            progress,
        )
    }

    pub fn index_with_options<E: Embedder>(
        &mut self,
        directory: &Path,
        embedder: &mut E,
        batch_size: usize,
        options: IndexOptions,
        cancelled: impl Fn() -> bool,
        mut progress: impl FnMut(&IndexProgress) -> Result<()>,
    ) -> Result<IndexResult> {
        ensure!(
            (1..=MAX_IMAGE_BATCH_SIZE).contains(&batch_size),
            "Image batch size must be between 1 and 8"
        );
        let directory = directory
            .canonicalize()
            .with_context(|| format!("Resolve selected folder {directory:?}"))?;
        ensure!(
            directory.is_dir(),
            "Selected source is not a directory: {directory:?}"
        );
        let data_dir = self
            .thumbnails
            .parent()
            .context("Catalog data path missing")?;
        ensure!(
            !data_dir.starts_with(&directory) && !directory.starts_with(data_dir),
            "Choose a media folder outside the search application's data directory"
        );
        let source_id = self.source(&directory)?;
        let mut state = IndexProgress {
            stage: "scanning",
            directory: text_path(&directory)?.to_owned(),
            current: 0,
            total: 0,
            indexed: 0,
            unchanged: 0,
            failed: 0,
            filename: String::new(),
            failure: None,
        };
        let mut failures = Vec::new();
        progress(&state)?;
        let mut paths = Vec::new();
        for entry in WalkDir::new(&directory).follow_links(false) {
            if cancelled() {
                return Ok(state.result(0, true, failures));
            }
            let entry = entry.with_context(|| format!("Scan folder {directory:?}"))?;
            if !entry.file_type().is_file() {
                continue;
            }
            if media::kind_for_path(entry.path()).is_none() {
                continue;
            }
            paths.push(entry.path().canonicalize()?);
            ensure!(
                paths.len() <= MAX_FILES,
                "Folder scan exceeded {MAX_FILES} files; select a smaller folder"
            );
        }
        paths.sort();
        state.stage = "indexing";
        state.total = paths.len();
        progress(&state)?;
        // One pinned, paged source snapshot. No per-file Lance reconnect/query.
        let previous = self.store.source_manifest(source_id)?;
        let mut manifest: HashMap<String, Vec<Record>> = HashMap::new();
        for row in previous {
            manifest.entry(row.path.clone()).or_default().push(row);
        }
        let mut batch = IndexBatch {
            source_id,
            images: Vec::with_capacity(batch_size),
            inputs: Vec::with_capacity(batch_size),
            pending: Vec::with_capacity(batch_size),
        };
        let mut pool = PreparationPool::new()?;
        let native = embedder.uses_prepared_inputs();
        let mut tokenizer = None;
        let mut prefetched: HashMap<PathBuf, FileTicket> = HashMap::new();
        for (position, path) in paths.iter().enumerate() {
            pool.check()?;
            if cancelled() {
                return Ok(state.result(0, true, failures));
            }
            // Prepare both images of the next common batch while this batch
            // infers. At a modality boundary retain only one following file:
            // two completed image reservations would otherwise starve the
            // current media file's segment jobs of their preparation budget.
            let lookahead = if batch_size > 1
                && paths.get(position..position + 3).is_some_and(|window| {
                    window
                        .iter()
                        .all(|path| media::kind_for_path(path) == Some(MediaKind::Image))
                }) {
                3
            } else {
                2
            };
            // Stat-identical files stay on the cheap scan path; admission below
            // schedules repairs explicitly.
            for (offset, candidate) in paths.iter().skip(position).take(lookahead).enumerate() {
                // If this file was skipped by the cheap prefetch check, reserve
                // room for a possible repair discovered during full admission.
                if offset == 2 && !prefetched.contains_key(path) {
                    break;
                }
                if prefetched.contains_key(candidate) {
                    continue;
                }
                let old = manifest.get(text_path(candidate)?);
                // Admission below reports source errors. Do not launch speculative
                // decoding when the initial stat/type check itself has failed.
                let candidate_stamp = match stamp(candidate) {
                    Ok(stamp) => stamp,
                    Err(_) => continue,
                };
                let unchanged_stat = old.is_some_and(|rows| {
                    !rows.is_empty()
                        && rows.iter().all(|row| {
                            row.bytes == candidate_stamp.bytes
                                && raw_modified(&row.modified) == candidate_stamp.modified
                                && cache_is_current(row)
                        })
                });
                if unchanged_stat && !options.reindex_text {
                    continue;
                }
                match media::is_typescript_source(candidate) {
                    Ok(false) => (),
                    Ok(true) | Err(_) => continue,
                }
                if native
                    && media::kind_for_path(candidate) == Some(MediaKind::Text)
                    && tokenizer.is_none()
                {
                    tokenizer = Some(embedder.preparation_tokenizer()?);
                }
                prefetched.insert(
                    candidate.clone(),
                    preparation::submit_file(&pool, candidate, native, tokenizer.clone())?,
                );
            }
            let ticket = prefetched.remove(path);
            let start = Instant::now();
            let path_text = text_path(path)?;
            let before = match media::is_typescript_source(path).and_then(|unsupported| {
                if unsupported {
                    return Ok(None);
                }
                let stamp = stamp(path)?;
                let max_bytes =
                    match media::kind_for_path(path).context("Unsupported indexed file")? {
                        MediaKind::Image => crate::preprocess::MAX_FILE_BYTES,
                        MediaKind::Text => media::MAX_TEXT_FILE_BYTES,
                        MediaKind::Audio | MediaKind::Video => media::MAX_MEDIA_FILE_BYTES,
                    };
                ensure!(
                    stamp.bytes > 0 && stamp.bytes <= max_bytes,
                    "Source {path:?} must contain 1..={max_bytes} bytes"
                );
                Ok(Some(stamp))
            }) {
                Ok(Some(stamp)) => stamp,
                Ok(None) => {
                    state.current += 1;
                    progress(&state)?;
                    continue;
                }
                Err(error) => {
                    if batch.flush(
                        self,
                        embedder,
                        &mut state,
                        &mut failures,
                        &cancelled,
                        &mut progress,
                    )? {
                        return Ok(state.result(0, true, failures));
                    }
                    state.filename = path
                        .file_name()
                        .context("Image filename missing")?
                        .to_string_lossy()
                        .into_owned();
                    state.record_failure(path, error, &mut failures);
                    progress(&state)?;
                    continue;
                }
            };
            let existing_rows = manifest.remove(path_text).unwrap_or_default();
            let saved = self
                .index_state
                .load(source_id, path_text, &existing_rows)?;
            let is_text = media::kind_for_path(path) == Some(MediaKind::Text);
            let forced = options.reindex_text && is_text;
            if is_text && !forced {
                ensure!(saved.as_ref().is_none_or(|state| state.extraction_profile == media::TEXT_EXTRACTION_PROFILE),
                    "Text extraction profile changed for {path:?}; request reindexText explicitly. Existing rows remain readable");
            }
            let same_stat = !existing_rows.is_empty()
                && saved.as_ref().map_or_else(
                    || {
                        existing_rows.iter().all(|row| {
                            row.bytes == before.bytes
                                && raw_modified(&row.modified) == before.modified
                        })
                    },
                    |state| state.stamp == before,
                );
            let same_profile = saved.as_ref().is_none_or(|state| {
                state.matches_profile(
                    embedder.embedding_profile().as_deref(),
                    is_text.then_some(media::TEXT_EXTRACTION_PROFILE),
                )
            });
            let verify_after = options.verify_content || is_racy(&before);
            // A nonce marks our new derived-state contract. Its missing/mismatched
            // sidecar must be reconstructed; true legacy rows may keep old chunks.
            let derived_state_missing =
                saved.is_none() && existing_rows.iter().any(|row| row.modified.contains('@'));
            let cheap_unchanged =
                same_stat && same_profile && !forced && !derived_state_missing && !verify_after;
            let caches_current = existing_rows.iter().all(cache_is_current);
            let content_hash = if cheap_unchanged && (saved.is_some() || caches_current) {
                saved.as_ref().map(|state| state.content_sha256.clone())
            } else {
                match hash_file_with_cancel(path, &before, &cancelled).and_then(|hash| {
                    ensure!(
                        stamp(path)? == before,
                        "File changed while hashing: {path:?}"
                    );
                    Ok(hash)
                }) {
                    Ok(hash) => Some(hash),
                    Err(error) => {
                        if cancelled() {
                            return Ok(state.result(0, true, failures));
                        }
                        if batch.flush(
                            self,
                            embedder,
                            &mut state,
                            &mut failures,
                            &cancelled,
                            &mut progress,
                        )? {
                            return Ok(state.result(0, true, failures));
                        }
                        state.record_failure(path, error, &mut failures);
                        progress(&state)?;
                        continue;
                    }
                }
            };
            let content_unchanged = !forced
                && same_profile
                && saved
                    .as_ref()
                    .is_some_and(|old| content_hash.as_ref() == Some(&old.content_sha256));
            let unchanged = cheap_unchanged || content_unchanged;
            if unchanged && caches_current {
                if batch.flush(
                    self,
                    embedder,
                    &mut state,
                    &mut failures,
                    &cancelled,
                    &mut progress,
                )? {
                    return Ok(state.result(0, true, failures));
                }
                if let Some(mut saved) = saved.filter(|saved| saved.stamp != before) {
                    saved.stamp = before.clone();
                    self.index_state.save(source_id, path_text, &saved)?;
                }
                state.failure = None;
                state.filename = path
                    .file_name()
                    .context("Media filename missing")?
                    .to_string_lossy()
                    .into_owned();
                state.unchanged += 1;
                state.current += 1;
                progress(&state)?;
                continue;
            }
            let content_hash = content_hash
                .context("Changed or cache-repaired file must have a checked content hash")?;
            // Flush a partial image batch while the next media file is being
            // prepared, rather than waiting for its first segment before inference.
            if media::kind_for_path(path) != Some(MediaKind::Image) {
                if batch.flush(
                    self,
                    embedder,
                    &mut state,
                    &mut failures,
                    &cancelled,
                    &mut progress,
                )? {
                    return Ok(state.result(0, true, failures));
                }
            }
            state.filename = path
                .file_name()
                .context("Media filename missing")?
                .to_string_lossy()
                .into_owned();
            let ticket = match ticket {
                Some(ticket) => ticket,
                None => {
                    if native && is_text && tokenizer.is_none() {
                        tokenizer = Some(embedder.preparation_tokenizer()?);
                    }
                    preparation::submit_file(&pool, path, native, tokenizer.clone())?
                }
            };
            let Some(prepared) = ticket.wait(&cancelled)? else {
                return Ok(state.result(0, true, failures));
            };
            let prepared = match prepared.and_then(|prepared| {
                ensure!(
                    prepared.stamp == before,
                    "File changed before preparation: {path:?}"
                );
                Ok(prepared.data)
            }) {
                Ok(prepared) => prepared,
                Err(error) if error.is::<FatalPreparation>() => return Err(error),
                Err(error) => {
                    if batch.flush(
                        self,
                        embedder,
                        &mut state,
                        &mut failures,
                        &cancelled,
                        &mut progress,
                    )? {
                        return Ok(state.result(0, true, failures));
                    }
                    state.filename = path
                        .file_name()
                        .context("Media filename missing")?
                        .to_string_lossy()
                        .into_owned();
                    state.record_failure(path, error, &mut failures);
                    progress(&state)?;
                    continue;
                }
            };
            if media::kind_for_path(path) != Some(MediaKind::Image) {
                state.failure = None;
                state.filename = path
                    .file_name()
                    .context("Media filename missing")?
                    .to_string_lossy()
                    .into_owned();
                if self.index_media_file(
                    &pool,
                    prepared,
                    tokenizer.clone(),
                    source_id,
                    path,
                    before,
                    existing_rows,
                    unchanged,
                    content_hash,
                    verify_after,
                    embedder,
                    &mut state,
                    &mut failures,
                    &cancelled,
                )? {
                    return Ok(state.result(0, true, failures));
                }
                progress(&state)?;
                continue;
            }
            ensure!(
                existing_rows.len() <= 1,
                "Image has multiple indexed segments"
            );
            let existing = existing_rows.into_iter().next();
            let FileData::Image {
                image,
                inputs,
                thumbnail,
            } = prepared
            else {
                bail!("Image preparation returned a different media type");
            };
            let pending = PendingImage {
                path: path.clone(),
                before,
                existing,
                decode_ms: start.elapsed().as_secs_f64() * 1000.0,
                start,
                content_hash,
                embedding_profile: embedder.embedding_profile(),
                verify_after,
                thumbnail,
            };
            if unchanged {
                if batch.flush(
                    self,
                    embedder,
                    &mut state,
                    &mut failures,
                    &cancelled,
                    &mut progress,
                )? {
                    return Ok(state.result(0, true, failures));
                }
                let id = &pending
                    .existing
                    .as_ref()
                    .context("Unchanged image record is missing")?
                    .asset_id;
                let vector = self
                    .store
                    .asset(id, true)?
                    .embedding
                    .context("Indexed image is missing its embedding")?;
                state.failure = None;
                state.filename = path
                    .file_name()
                    .context("Image filename missing")?
                    .to_string_lossy()
                    .into_owned();
                self.finish_index_image(
                    source_id,
                    pending,
                    &image,
                    vector,
                    &mut state,
                    &mut failures,
                )?;
                progress(&state)?;
                continue;
            }
            // EmbeddingGemma2 pads every image to the same 2520-patch budget,
            // so consecutive images can share a batch regardless of dimensions.
            if let Some(inputs) = inputs {
                batch.inputs.push(inputs);
            }
            batch.images.push(image);
            batch.pending.push(pending);
            if batch.images.len() == batch_size
                && batch.flush(
                    self,
                    embedder,
                    &mut state,
                    &mut failures,
                    &cancelled,
                    &mut progress,
                )?
            {
                return Ok(state.result(0, true, failures));
            }
        }
        if batch.flush(
            self,
            embedder,
            &mut state,
            &mut failures,
            &cancelled,
            &mut progress,
        )? {
            return Ok(state.result(0, true, failures));
        }
        pool.finish()?;
        // Only a completed traversal may prune missing files. Known files that
        // failed decoding stay in `seen`, preserving their last good entries.
        let seen: HashSet<_> = paths
            .iter()
            .map(|p| text_path(p).map(str::to_owned))
            .collect::<Result<_>>()?;
        let stale = manifest
            .into_values()
            .flatten()
            .filter(|row| !seen.contains(&row.path))
            .collect::<Vec<_>>();
        if cancelled() {
            return Ok(state.result(0, true, failures));
        }
        let mut removed = 0;
        for rows in stale.chunks(1000) {
            let ids = rows
                .iter()
                .map(|row| quote(&row.asset_id))
                .collect::<Vec<_>>()
                .join(",");
            self.store
                .delete(&format!("kind = 'asset' AND asset_id IN ({ids})"))?;
            for row in rows {
                if let Some(cache) = &row.thumbnail {
                    self.remove_cache(Path::new(cache))?;
                }
                self.remove_cache(&self.previews.join(format!("{}.png", row.asset_id)))?;
                removed += 1;
            }
        }
        // Only completed traversals refresh the derived text index. Individual
        // file commits (including interrupted scans) are searchable immediately.
        self.store.checkpoint()?;
        state.stage = "complete";
        state.failure = None;
        progress(&state)?;
        Ok(state.result(removed, false, failures))
    }
    #[allow(clippy::too_many_arguments)]
    fn index_media_file<E: Embedder>(
        &self,
        pool: &PreparationPool,
        prepared: FileData,
        tokenizer: Option<std::sync::Arc<crate::preprocess::QueryTokenizer>>,
        source_id: i64,
        path: &Path,
        before: Stamp,
        existing: Vec<Record>,
        unchanged: bool,
        content_hash: String,
        verify_after: bool,
        embedder: &mut E,
        state: &mut IndexProgress,
        failures: &mut Vec<IndexFailure>,
        cancelled: &impl Fn() -> bool,
    ) -> Result<bool> {
        let FileData::Media {
            info,
            specs,
            mut first,
        } = prepared
        else {
            bail!("Media preparation returned an image");
        };
        let mut tokenizer_failed = false;
        let segmented = match specs {
            Some(specs) => Ok(specs),
            None => media::segments_with_token_counter(path, &info, |title, text| {
                let result = embedder.document_token_count(title, text);
                tokenizer_failed |= result.is_err();
                result
            }),
        };
        let specs = match segmented {
            Ok(specs) if !specs.is_empty() => specs,
            Ok(_) => {
                state.record_failure(
                    path,
                    anyhow::anyhow!("File contains no indexable segments"),
                    failures,
                );
                return Ok(cancelled());
            }
            Err(error) if tokenizer_failed => {
                return Err(error).context("Count real model document tokens")
            }
            Err(error) => {
                state.record_failure(path, error, failures);
                return Ok(cancelled());
            }
        };
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("Media has no UTF-8 filename")?;
        let modified = generation_modified(&before);
        let digest = format!("{:x}", Sha256::digest(modified.as_bytes()));
        let mut rows = Vec::with_capacity(specs.len());
        let mut thumbnails = Vec::new();
        let mut locations = Vec::new();
        // Long documents/clips should not scan every old segment for each new
        // segment. Borrow the existing metadata once; retain it for cleanup.
        let mut existing_by_segment = HashMap::with_capacity(existing.len());
        for row in &existing {
            ensure!(
                existing_by_segment
                    .insert((row.media_kind.as_str(), row.segment_index), row)
                    .is_none(),
                "Indexed file contains duplicate segment identities: {path:?}"
            );
        }
        let mut tickets = std::collections::VecDeque::new();
        let mut scheduled = usize::from(first.is_some());
        for spec in &specs {
            let ready = first.take();
            let queued_limit = if ready.is_some() { 1 } else { 2 };
            while scheduled < specs.len() && tickets.len() < queued_limit {
                tickets.push_back(preparation::submit_segment(
                    pool,
                    path,
                    &info,
                    &specs[scheduled],
                    embedder.uses_prepared_inputs(),
                    tokenizer.clone(),
                )?);
                scheduled += 1;
            }
            if cancelled() {
                return Ok(true);
            }
            let result = if let Some(ready) = ready {
                Ok(ready)
            } else {
                let Some(result) = tickets
                    .pop_front()
                    .context("Missing prepared segment ticket")?
                    .wait(cancelled)?
                else {
                    return Ok(true);
                };
                result
            };
            let preparation::PreparedSegment {
                raw: prepared,
                input,
                thumbnail,
            } = match result {
                Ok(prepared) => prepared,
                Err(error) if error.is::<FatalPreparation>() => return Err(error),
                Err(error) => {
                    state.record_failure(path, error, failures);
                    return Ok(cancelled());
                }
            };
            if cancelled() {
                return Ok(true);
            }
            // Modality-local indices survive soundtrack additions/removals and
            // slight tail-boundary changes when a file grows or shrinks.
            let old = existing_by_segment
                .get(&(spec.kind.as_str(), spec.index))
                .copied();
            let asset_id = old
                .map(|row| row.asset_id.clone())
                .unwrap_or_else(|| format!("asset_{}", Uuid::new_v4()));
            let embedding = if unchanged && old.is_some() {
                self.store
                    .asset(&old.expect("matched unchanged segment").asset_id, true)?
                    .embedding
                    .context("Indexed segment is missing its embedding")?
            } else {
                match &prepared {
                    PreparedMedia::Text { title, text } => {
                        let key = embedder.embedding_profile().map(|profile| {
                            self.index_state.document_key(
                                &profile,
                                media::TEXT_EXTRACTION_PROFILE,
                                title,
                                text,
                            )
                        });
                        if let Some(cached) = key
                            .as_deref()
                            .map(|key| self.index_state.vector(key))
                            .transpose()?
                            .flatten()
                        {
                            Ok(cached)
                        } else {
                            let vector = match input {
                                NativeInput::Text(tokens) => {
                                    embedder.embed_prepared_document(tokens)
                                }
                                NativeInput::Raw => embedder.embed_document(title, text),
                                _ => bail!("Prepared text input type mismatch"),
                            }?;
                            validate_embedding(&vector)?;
                            if let Some(key) = key {
                                self.index_state.save_vector(&key, &vector)?;
                            }
                            Ok(vector)
                        }
                    }
                    PreparedMedia::Audio { samples } => match input {
                        NativeInput::Audio(inputs) => embedder.embed_prepared_audio(inputs),
                        NativeInput::Raw => embedder.embed_audio(samples),
                        _ => bail!("Prepared audio input type mismatch"),
                    },
                    PreparedMedia::Video { frames } => match input {
                        NativeInput::Video(inputs) => embedder.embed_prepared_video(inputs),
                        NativeInput::Raw => embedder.embed_video(frames),
                        _ => bail!("Prepared video input type mismatch"),
                    },
                }
                .with_context(|| {
                    format!(
                        "Embed {} segment {} of {path:?}",
                        spec.kind.as_str(),
                        spec.index
                    )
                })?
            };
            validate_embedding(&embedding)?;
            // A multi-segment file is one unit: cancellation never publishes a
            // partial document, clip, or soundtrack, or removes its old rows.
            if cancelled() {
                return Ok(true);
            }
            let (width, height, cache) = match prepared {
                PreparedMedia::Video { frames } => {
                    let frame = frames
                        .first()
                        .context("Decoded video segment has no frames")?;
                    let cache = self
                        .thumbnails
                        .join(format!("{asset_id}-{}.webp", &digest[..16]));
                    thumbnails.push((
                        cache.clone(),
                        thumbnail.context("Prepared video thumbnail missing")?,
                    ));
                    (
                        frame.width(),
                        frame.height(),
                        Some(text_path(&cache)?.to_owned()),
                    )
                }
                PreparedMedia::Text { .. } | PreparedMedia::Audio { .. } => (0, 0, None),
            };
            // Full passage is private index content; the display excerpt stays
            // separately bounded and never includes the filename or directory.
            let indexed_text = spec.text.as_ref().map(|text| format!("{filename}\n{text}"));
            let text_excerpt = spec
                .text
                .as_deref()
                .map(crate::store::text_excerpt)
                .map(str::to_owned);
            locations.push(PassageLocation {
                asset_id: asset_id.clone(),
                row_generation: String::new(),
                start_byte: spec.start_byte,
                end_byte: spec.end_byte,
                start_line: spec.start_line,
                end_line: spec.end_line,
            });
            rows.push(Record {
                kind: "asset".into(),
                asset_id,
                source_id,
                path: text_path(path)?.to_owned(),
                filename: filename.to_owned(),
                width,
                height,
                bytes: before.bytes,
                modified: modified.clone(),
                thumbnail: cache,
                media_kind: spec.kind.as_str().into(),
                segment_index: spec.index,
                start_ms: spec.start_ms,
                end_ms: spec.end_ms,
                text_excerpt,
                indexed_text,
                embedding: Some(embedding),
            });
        }
        let check = || {
            stamp(path).and_then(|after| {
                ensure!(
                    after == before,
                    "File changed during indexing: {path:?}; rescan after it finishes changing"
                );
                if verify_after {
                    ensure!(hash_file_with_cancel(path, &before, cancelled)? == content_hash,
                    "File bytes changed during indexing despite an unchanged timestamp: {path:?}");
                }
                Ok(())
            })
        };
        if let Err(error) = check() {
            if cancelled() {
                return Ok(true);
            }
            state.record_failure(path, error, failures);
            return Ok(cancelled());
        }
        if cancelled() {
            return Ok(true);
        }
        // Cache files are staged only after every segment has decoded and passed
        // the vector contract. Old cache names remain valid until the commit.
        for (cache, bytes) in &thumbnails {
            if let Err(error) = write_atomic(cache, bytes) {
                self.discard_staged_thumbnails(&thumbnails, &existing)?;
                return Err(error).context("Write indexed segment thumbnail");
            }
        }
        if cancelled() {
            self.discard_staged_thumbnails(&thumbnails, &existing)?;
            return Ok(true);
        }
        if let Err(error) = check() {
            self.discard_staged_thumbnails(&thumbnails, &existing)?;
            if cancelled() {
                return Ok(true);
            }
            state.record_failure(path, error, failures);
            return Ok(false);
        }
        if let Err(error) = self.store.replace_file(source_id, text_path(path)?, &rows) {
            self.discard_staged_thumbnails(&thumbnails, &existing)?;
            return Err(error);
        }
        let extraction = if info.kind == MediaKind::Text {
            media::TEXT_EXTRACTION_PROFILE
        } else {
            "native-media-segments-v1"
        };
        self.index_state.save(
            source_id,
            text_path(path)?,
            &FileState::new(
                &rows,
                before,
                content_hash,
                extraction,
                embedder.embedding_profile(),
                locations,
            ),
        )?;
        let retained = rows
            .iter()
            .filter_map(|row| row.thumbnail.as_deref())
            .collect::<HashSet<_>>();
        for old in existing {
            if let Some(cache) = old
                .thumbnail
                .as_deref()
                .filter(|cache| !retained.contains(cache))
            {
                self.remove_cache(Path::new(cache))?;
            }
            self.remove_cache(&self.previews.join(format!("{}.png", old.asset_id)))?;
        }
        state.indexed += 1;
        state.current += 1;
        Ok(cancelled())
    }
    fn discard_staged_thumbnails(
        &self,
        staged: &[(PathBuf, Vec<u8>)],
        existing: &[Record],
    ) -> Result<()> {
        for (path, _) in staged {
            if !existing.iter().any(|row| {
                row.thumbnail
                    .as_deref()
                    .is_some_and(|old| Path::new(old) == path)
            }) {
                self.remove_cache(path)?;
            }
        }
        Ok(())
    }
    fn finish_index_image(
        &self,
        source_id: i64,
        pending: PendingImage,
        image: &RgbImage,
        embedding: Vec<f32>,
        state: &mut IndexProgress,
        failures: &mut Vec<IndexFailure>,
    ) -> Result<()> {
        validate_embedding(&embedding)?;
        let PendingImage {
            path,
            before,
            existing,
            decode_ms,
            start,
            content_hash,
            embedding_profile,
            verify_after,
            thumbnail,
        } = pending;
        let source_check = stamp(&path).and_then(|after| {
            ensure!(
                before == after,
                "Image changed during indexing: {path:?}; rescan after it finishes changing"
            );
            if verify_after {
                ensure!(
                    hash_file(&path, &before)? == content_hash,
                    "Image bytes changed despite an unchanged timestamp: {path:?}"
                );
            }
            Ok(())
        });
        if let Err(error) = source_check {
            state.record_failure(&path, error, failures);
            return Ok(());
        }
        let asset_id = existing
            .as_ref()
            .map(|r| r.asset_id.clone())
            .unwrap_or_else(|| format!("asset_{}", Uuid::new_v4()));
        let modified = generation_modified(&before);
        let digest = format!("{:x}", Sha256::digest(modified.as_bytes()));
        let cache = self
            .thumbnails
            .join(format!("{asset_id}-{}.webp", &digest[..16]));
        let thumb_start = Instant::now();
        write_atomic(&cache, &thumbnail)?;
        if let Err(error) = stamp(&path).and_then(|after| {
            ensure!(
                before == after,
                "Image changed while preparing its thumbnail; rescan its folder"
            );
            if verify_after {
                ensure!(
                    hash_file(&path, &before)? == content_hash,
                    "Image bytes changed while preparing its thumbnail; rescan its folder"
                );
            }
            Ok(())
        }) {
            if existing.as_ref().and_then(|row| row.thumbnail.as_deref())
                != Some(text_path(&cache)?)
            {
                self.remove_cache(&cache)?;
            }
            state.record_failure(&path, error, failures);
            return Ok(());
        }
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("Image has no UTF-8 filename")?;
        let row = Record {
            kind: "asset".into(),
            media_kind: "image".into(),
            asset_id: asset_id.clone(),
            source_id,
            path: text_path(&path)?.to_owned(),
            filename: filename.to_owned(),
            width: image.width(),
            height: image.height(),
            bytes: before.bytes,
            modified: modified.clone(),
            thumbnail: Some(text_path(&cache)?.to_owned()),
            embedding: Some(embedding),
            ..Record::default()
        };
        self.store
            .replace_file(source_id, text_path(&path)?, std::slice::from_ref(&row))?;
        self.index_state.save(
            source_id,
            text_path(&path)?,
            &FileState::new(
                std::slice::from_ref(&row),
                before.clone(),
                content_hash,
                "native-image-v1",
                embedding_profile,
                Vec::new(),
            ),
        )?;
        if let Some(old) = existing.as_ref().filter(|r| {
            r.thumbnail.as_deref() != Some(text_path(&cache).expect("validated cache path"))
        }) {
            if let Some(cache) = &old.thumbnail {
                self.remove_cache(Path::new(cache))?;
            }
        }
        self.remove_cache(&self.previews.join(format!("{asset_id}.png")))?;
        state.indexed += 1;
        state.current += 1;
        let reused = existing.as_ref().is_some_and(|row| {
            row.bytes == before.bytes && raw_modified(&row.modified) == before.modified
        });
        eprintln!("[Index] path={path:?} dimensions={}x{} decode_ms={decode_ms:.1} thumbnail_and_store_ms={:.1} total_ms={:.1} embedding_reused={reused}", image.width(), image.height(), thumb_start.elapsed().as_secs_f64()*1000.0, start.elapsed().as_secs_f64()*1000.0);
        Ok(())
    }
    fn remove_cache(&self, path: &Path) -> Result<()> {
        ensure!(
            path.starts_with(&self.thumbnails) || path.starts_with(&self.previews),
            "Refusing to delete cache outside search data: {path:?}"
        );
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e).with_context(|| format!("Remove outdated cache {path:?}")),
        }
    }
    pub fn preview(&self, asset_id: &str) -> Result<String> {
        Uuid::parse_str(
            asset_id
                .strip_prefix("asset_")
                .context("Invalid asset identifier")?,
        )
        .context("Invalid asset UUID")?;
        let record = self.store.asset(asset_id, false)?;
        ensure!(record.media_kind == "image", "Full previews are available only for images; use the video thumbnail or open the original file");
        let observed = self
            .index_state
            .load(
                record.source_id,
                &record.path,
                std::slice::from_ref(&record),
            )?
            .map(|state| state.stamp)
            .unwrap_or_else(|| Stamp {
                bytes: record.bytes,
                modified: raw_modified(&record.modified).to_owned(),
            });
        let (path, width, height) = (record.path, record.width, record.height);
        let path = Path::new(&path);
        let indexed_stamp = observed;
        ensure!(
            stamp(path)? == indexed_stamp,
            "Image changed since indexing; rescan its folder before previewing it"
        );
        let preview = self.previews.join(format!("{asset_id}.png"));
        if preview.is_file() {
            let dimensions = image::image_dimensions(&preview)
                .with_context(|| format!("Read cached preview dimensions {preview:?}"))?;
            ensure!(
                dimensions == bounded_dimensions(width, height, 1600),
                "Cached preview dimensions are invalid; rescan its source folder"
            );
        } else {
            let image = decode_image(path)?;
            let pixels = bounded_thumbnail(&image, 1600);
            let mut bytes = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(pixels).write_to(&mut bytes, image::ImageFormat::Png)?;
            ensure!(
                stamp(path)? == indexed_stamp,
                "Image changed while generating its preview; rescan its folder"
            );
            write_atomic(&preview, &bytes.into_inner())?;
        }
        file_url(&preview)
    }
    pub fn original_url(&self, asset_id: &str) -> Result<String> {
        crate::agent::validate_asset_id(asset_id)?;
        let path = self.store.asset(asset_id, false)?.path;
        ensure!(
            Path::new(&path).is_file(),
            "Original media file is missing: {path}"
        );
        file_url(Path::new(&path))
    }
    pub fn checkpoint(&self) -> Result<()> {
        self.store.checkpoint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::patch_budget;
    use anyhow::bail;
    struct FakeEmbedder {
        images: usize,
    }
    impl Embedder for FakeEmbedder {
        fn embed_images(&mut self, images: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
            images.iter().map(|image| self.embed_image(image)).collect()
        }
        fn embed_image(&mut self, image: &RgbImage) -> Result<Vec<f32>> {
            self.images += 1;
            let mut v = vec![0.; EMBEDDING_DIM];
            v[usize::from(image.get_pixel(0, 0)[2] > image.get_pixel(0, 0)[0])] = 1.;
            Ok(v)
        }
        fn embed_text(&mut self, _: &str) -> Result<Vec<f32>> {
            bail!("test embedder has no text model")
        }
    }
    fn save(path: &Path, color: [u8; 3]) {
        RgbImage::from_pixel(48, 32, image::Rgb(color))
            .save(path)
            .unwrap();
    }
    struct BatchEmbedder<'a> {
        model: FakeEmbedder,
        calls: Vec<Vec<usize>>,
        cancel: Option<&'a std::cell::Cell<bool>>,
        invalid_output: bool,
        invalid_embedding: Option<Vec<f32>>,
    }
    impl Embedder for BatchEmbedder<'_> {
        fn embed_image(&mut self, _: &RgbImage) -> Result<Vec<f32>> {
            bail!("Indexer must call batch inference")
        }
        fn embed_images(&mut self, images: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
            let budgets = images
                .iter()
                .map(|image| patch_budget(image.width(), image.height()))
                .collect::<Result<Vec<_>>>()?;
            assert!(budgets.iter().all(|budget| *budget == 2520));
            self.calls.push(budgets);
            if let Some(cancel) = self.cancel {
                cancel.set(true);
            }
            let mut vectors = self.model.embed_images(images)?;
            if self.invalid_output {
                vectors.pop();
            }
            if let Some(vector) = &self.invalid_embedding {
                *vectors.last_mut().expect("nonempty batch") = vector.clone();
            }
            Ok(vectors)
        }
        fn embed_text(&mut self, _: &str) -> Result<Vec<f32>> {
            bail!("test embedder has no text model")
        }
    }
    fn batch_embedder() -> BatchEmbedder<'static> {
        BatchEmbedder {
            model: FakeEmbedder { images: 0 },
            calls: vec![],
            cancel: None,
            invalid_output: false,
            invalid_embedding: None,
        }
    }
    #[test]
    fn image_lookahead_leaves_room_for_a_same_stat_sidecar_repair() {
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        let data = root.path().join("data");
        fs::create_dir(&photos).unwrap();
        save(&photos.join("a.png"), [255, 0, 0]);
        let mut catalog = Catalog::open(&data).unwrap();
        let mut model = batch_embedder();
        catalog
            .index_with_batch_size(&photos, &mut model, 2, || false, |_| Ok(()))
            .unwrap();
        // Source stat and thumbnail still match, but full admission must rebuild
        // this file's missing generation-bound state. Two later outputs must not
        // occupy all reservations before that repair job can start.
        fs::remove_dir_all(data.join("index-state-v1/files")).unwrap();
        save(&photos.join("b.png"), [0, 255, 0]);
        save(&photos.join("c.png"), [0, 0, 255]);
        let start = Instant::now();
        let result = catalog
            .index_with_batch_size(
                &photos,
                &mut model,
                2,
                || start.elapsed() > std::time::Duration::from_secs(10),
                |_| Ok(()),
            )
            .unwrap();
        assert!(
            !result.cancelled,
            "Lookahead starved the current repair job"
        );
        assert_eq!((result.indexed, result.failed), (3, 0));
        assert_eq!(catalog.status().unwrap().asset_count, 3);
    }

    #[test]
    fn indexing_batches_respect_the_limit_flush_tails_and_reuse_embeddings() {
        for size in [1, 2, 3, 8] {
            let root = tempfile::tempdir().unwrap();
            let photos = root.path().join("photos");
            fs::create_dir(&photos).unwrap();
            for index in 0..10 {
                save(&photos.join(format!("{index:02}.png")), [255, 0, 0]);
            }
            let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
            let mut model = batch_embedder();
            let result = catalog
                .index_with_batch_size(&photos, &mut model, size, || false, |_| Ok(()))
                .unwrap();
            assert_eq!(result.indexed, 10);
            assert_eq!(
                model.calls.iter().map(Vec::len).collect::<Vec<_>>(),
                (0..10)
                    .collect::<Vec<_>>()
                    .chunks(size)
                    .map(<[_]>::len)
                    .collect::<Vec<_>>()
            );
            let calls = model.calls.len();
            let first = catalog.browse(0, 10).unwrap().items[0].clone();
            let thumbnail = Url::parse(&first.thumbnail_url)
                .unwrap()
                .to_file_path()
                .unwrap();
            fs::remove_file(thumbnail).unwrap();
            let repaired = catalog
                .index_with_batch_size(&photos, &mut model, size, || false, |_| Ok(()))
                .unwrap();
            assert_eq!((repaired.indexed, repaired.unchanged), (1, 9));
            assert_eq!(model.calls.len(), calls);
            assert_eq!(
                catalog.browse(0, 10).unwrap().items[0].asset_id,
                first.asset_id
            );
        }
    }
    #[test]
    fn indexing_batches_mix_dimensions_with_fixed_patch_budget_and_report_corrupt_files() {
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        for (index, width, height) in [
            (0, 48, 32),
            (1, 1, 1024),
            (2, 200, 200),
            (3, 240, 200),
            (5, 640, 480),
        ] {
            RgbImage::from_pixel(width, height, image::Rgb([255, 0, 0]))
                .save(photos.join(format!("{index}.png")))
                .unwrap();
        }
        fs::write(photos.join("4.png"), "corrupt").unwrap();
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut model = batch_embedder();
        let result = catalog
            .index_with_batch_size(&photos, &mut model, 8, || false, |_| Ok(()))
            .unwrap();
        assert_eq!((result.indexed, result.failed), (5, 1));
        assert_eq!(model.calls, [vec![2520; 4], vec![2520]]);
        for size in [0, 9] {
            assert!(catalog
                .index_with_batch_size(&photos, &mut model, size, || false, |_| Ok(()))
                .is_err());
        }
    }
    #[test]
    fn cancellation_during_batch_inference_commits_one_image_and_never_prunes() {
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        for index in 0..5 {
            save(&photos.join(format!("{index}.png")), [255, 0, 0]);
        }
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        catalog
            .index(
                &photos,
                &mut FakeEmbedder { images: 0 },
                || false,
                |_| Ok(()),
            )
            .unwrap();
        let assets = catalog.browse(0, 10).unwrap().items;
        fs::remove_file(photos.join("4.png")).unwrap();
        for index in 0..4 {
            save(&photos.join(format!("{index}.png")), [0, 0, 255]);
        }
        let cancel = std::cell::Cell::new(false);
        let mut model = BatchEmbedder {
            cancel: Some(&cancel),
            ..batch_embedder()
        };
        let result = catalog
            .index_with_batch_size(&photos, &mut model, 4, || cancel.get(), |_| Ok(()))
            .unwrap();
        assert!(result.cancelled);
        assert_eq!((result.indexed, result.removed), (1, 0));
        assert_eq!(model.calls.len(), 1);
        assert_eq!(catalog.status().unwrap().asset_count, 5);
        assert_eq!(
            catalog
                .store
                .asset(&assets[0].asset_id, true)
                .unwrap()
                .embedding
                .unwrap()[1],
            1.
        );
        assert_eq!(
            catalog
                .store
                .asset(&assets[1].asset_id, true)
                .unwrap()
                .embedding
                .unwrap()[0],
            1.
        );
    }
    #[test]
    fn invalid_batch_outputs_are_fatal_without_partial_commits_or_retries() {
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        for index in 0..4 {
            save(&photos.join(format!("{index}.png")), [255, 0, 0]);
        }
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut model = BatchEmbedder {
            invalid_output: true,
            ..batch_embedder()
        };
        let error = catalog
            .index_with_batch_size(&photos, &mut model, 4, || false, |_| Ok(()))
            .unwrap_err();
        assert!(error.to_string().contains("3 embeddings for 4 images"));
        assert_eq!(model.calls.len(), 1);
        assert_eq!(catalog.status().unwrap().asset_count, 0);
    }
    #[test]
    fn cached_images_preserve_aspect_ratio_and_reject_wrong_thumbnail_dimensions() {
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        for (name, width, height) in [
            ("landscape.png", 2400, 800),
            ("portrait.png", 800, 2400),
            ("small.png", 48, 32),
            ("narrow.png", 1, 2400),
        ] {
            RgbImage::from_pixel(width, height, image::Rgb([255, 0, 0]))
                .save(photos.join(name))
                .unwrap();
        }
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut model = FakeEmbedder { images: 0 };
        catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        let assets = catalog.browse(0, 10).unwrap().items;
        for asset in &assets {
            let (thumb_dimensions, preview_dimensions) = match asset.filename.as_str() {
                "landscape.png" => ((384, 128), (1600, 533)),
                "portrait.png" => ((128, 384), (533, 1600)),
                "small.png" => ((48, 32), (48, 32)),
                "narrow.png" => ((1, 384), (1, 1600)),
                _ => unreachable!(),
            };
            let thumbnail = Url::parse(&asset.thumbnail_url)
                .unwrap()
                .to_file_path()
                .unwrap();
            let preview = Url::parse(&catalog.preview(&asset.asset_id).unwrap())
                .unwrap()
                .to_file_path()
                .unwrap();
            assert_eq!(
                image::image_dimensions(&thumbnail).unwrap(),
                thumb_dimensions
            );
            assert_eq!(
                image::image_dimensions(&preview).unwrap(),
                preview_dimensions
            );
            RgbImage::new(384, 384).save(&thumbnail).unwrap();
            RgbImage::new(1600, 1600).save(&preview).unwrap();
            assert!(catalog.browse(0, 10).is_err());
            catalog
                .index(&photos, &mut model, || false, |_| Ok(()))
                .unwrap();
            catalog.preview(&asset.asset_id).unwrap();
            let refreshed = catalog
                .browse(0, 10)
                .unwrap()
                .items
                .into_iter()
                .find(|current| current.asset_id == asset.asset_id)
                .unwrap();
            let repaired_thumbnail = Url::parse(&refreshed.thumbnail_url)
                .unwrap()
                .to_file_path()
                .unwrap();
            assert_ne!(
                repaired_thumbnail, thumbnail,
                "Repair must publish a new generation-specific cache path"
            );
            assert!(
                !thumbnail.exists(),
                "Old thumbnail is removed only after commit"
            );
            assert_eq!(
                image::image_dimensions(&repaired_thumbnail).unwrap(),
                thumb_dimensions
            );
            assert_eq!(
                image::image_dimensions(&preview).unwrap(),
                preview_dimensions
            );
        }
        assert_eq!(model.images, 4);
        assert_eq!(
            assets.iter().map(|a| &a.asset_id).collect::<Vec<_>>(),
            catalog
                .browse(0, 10)
                .unwrap()
                .items
                .iter()
                .map(|a| &a.asset_id)
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn source_filters_rank_before_top_k_and_removal_preserves_other_sources_and_originals() {
        let root = tempfile::tempdir().unwrap();
        let red = root.path().join("red");
        let blue = root.path().join("blue");
        fs::create_dir(&red).unwrap();
        fs::create_dir(&blue).unwrap();
        save(&red.join("red.png"), [255, 0, 0]);
        for name in ["blue-a.png", "blue-b.png"] {
            save(&blue.join(name), [0, 0, 255]);
        }
        let data = root.path().join("data");
        let mut catalog = Catalog::open(&data).unwrap();
        let mut model = FakeEmbedder { images: 0 };
        for source in [&red, &blue] {
            catalog
                .index(source, &mut model, || false, |_| Ok(()))
                .unwrap();
        }
        let blue_id = catalog
            .status()
            .unwrap()
            .sources
            .iter()
            .find(|source| source.directory == blue.to_str().unwrap())
            .unwrap()
            .id;
        let page = catalog.browse_source(1, 1, Some(blue_id)).unwrap();
        assert_eq!(page.total, 2);
        assert_eq!(page.items[0].filename, "blue-b.png");
        let mut query = vec![0.; EMBEDDING_DIM];
        query[0] = 1.;
        assert_eq!(catalog.search(&query, 1).unwrap()[0].filename, "red.png");
        let hits = catalog.search_source(&query, 1, Some(blue_id)).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].filename.starts_with("blue-"));
        let thumbnail = Url::parse(&hits[0].thumbnail_url)
            .unwrap()
            .to_file_path()
            .unwrap();
        let preview = Url::parse(&catalog.preview(&hits[0].asset_id).unwrap())
            .unwrap()
            .to_file_path()
            .unwrap();
        assert_eq!(catalog.remove_source(blue_id).unwrap(), 2);
        assert!(!thumbnail.exists() && !preview.exists());
        assert!(blue.join("blue-a.png").exists() && blue.join("blue-b.png").exists());
        assert_eq!(catalog.status().unwrap().asset_count, 1);
        assert_eq!(catalog.search(&query, 10).unwrap().len(), 1);
        assert!(catalog.browse_source(0, 1, Some(blue_id)).is_err());
        assert!(catalog.search_source(&query, 1, Some(0)).is_err());
        assert!(catalog.remove_source(blue_id).is_err());
        drop(catalog);
        let reopened = Catalog::open(&data).unwrap();
        assert_eq!(reopened.status().unwrap().sources.len(), 1);
        assert_eq!(reopened.browse(0, 10).unwrap().items[0].filename, "red.png");
    }
    #[test]
    fn incremental_index_preserves_ids_repairs_thumbnails_and_prunes_only_complete_scans() {
        crate::initialize_image_decoders();
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        save(&photos.join("red # ü.png"), [255, 0, 0]);
        save(&photos.join("blue.png"), [0, 0, 255]);
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut model = FakeEmbedder { images: 0 };
        let first = catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!(first.indexed, 2);
        let before = catalog.browse(0, 100).unwrap();
        assert_eq!(before.total, 2);
        let mut query = vec![0.; EMBEDDING_DIM];
        query[0] = 1.;
        let hits = catalog.search(&query, 10).unwrap();
        assert_eq!(hits[0].filename, "red # ü.png");
        assert!(hits[0].similarity.unwrap() > 0.99);
        assert!(hits[0].thumbnail_url.starts_with("file://"));
        let next = catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!(next.unchanged, 2);
        assert_eq!(model.images, 2);
        fs::remove_file(
            Url::parse(&hits[0].thumbnail_url)
                .unwrap()
                .to_file_path()
                .unwrap(),
        )
        .unwrap();
        assert!(catalog.browse(0, 100).is_err());
        let next = catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!(next.indexed, 1);
        assert_eq!(model.images, 2);
        assert_eq!(
            before.items[1].asset_id,
            catalog.browse(0, 100).unwrap().items[1].asset_id
        );
        assert!(catalog
            .preview(&hits[0].asset_id)
            .unwrap()
            .starts_with("file://"));
        fs::remove_file(photos.join("blue.png")).unwrap();
        assert!(
            catalog
                .index(&photos, &mut model, || true, |_| Ok(()))
                .unwrap()
                .cancelled
        );
        assert_eq!(catalog.status().unwrap().asset_count, 2);
        assert_eq!(
            catalog
                .index(&photos, &mut model, || false, |_| Ok(()))
                .unwrap()
                .removed,
            1
        );
        save(&photos.join("red # ü.png"), [0, 255, 0]);
        catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!(model.images, 3);
        assert_eq!(
            before.items[1].asset_id,
            catalog.browse(0, 100).unwrap().items[0].asset_id
        );
        catalog.checkpoint().unwrap();
        drop(catalog);
        assert_eq!(
            Catalog::open(&root.path().join("data"))
                .unwrap()
                .status()
                .unwrap()
                .asset_count,
            1
        );
    }
    #[test]
    fn catalog_rejects_missing_storage_overlaps_and_unbounded_queries() {
        let root = tempfile::tempdir().unwrap();
        let data = root.path().join("data");
        let photos = root.path().join("photos");
        fs::create_dir_all(photos.join("child")).unwrap();
        save(&photos.join("valid.png"), [255, 0, 0]);
        let mut catalog = Catalog::open(&data).unwrap();
        let mut model = FakeEmbedder { images: 0 };
        assert_eq!(
            catalog
                .index(&photos, &mut model, || false, |_| Ok(()))
                .unwrap()
                .indexed,
            1
        );
        assert!(catalog
            .index(&photos.join("child"), &mut model, || false, |_| Ok(()))
            .is_err());
        assert!(catalog.browse(0, 0).is_err());
        assert!(catalog.search(&[1.], 10).is_err());
        assert!(catalog.preview("bad-id").is_err());
        for directory in [
            data.clone(),
            data.join("previews"),
            root.path().to_path_buf(),
        ] {
            assert!(catalog
                .index(&directory, &mut model, || false, |_| Ok(()))
                .is_err());
        }
        drop(catalog);
        fs::rename(
            data.join("catalog.lancedb/assets.lance"),
            data.join("catalog.lancedb/missing.lance"),
        )
        .unwrap();
        assert!(Catalog::open(&data)
            .unwrap_err()
            .to_string()
            .contains("missing"));
    }

    #[test]
    fn corrupt_images_report_errors_continue_the_scan_and_retry_after_repair() {
        crate::initialize_image_decoders();
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        save(&photos.join("a-valid.png"), [255, 0, 0]);
        let jpeg = photos.join("b-corrupt.jpg");
        let jpeg_bytes: Vec<u8> = include_str!("../tests/fixtures/jpeg-extraneous-byte.jpg.hex")
            .split_ascii_whitespace()
            .map(|byte| u8::from_str_radix(byte, 16).expect("invalid synthetic JPEG fixture hex"))
            .collect();
        fs::write(&jpeg, jpeg_bytes).unwrap();
        fs::write(photos.join("c-invalid.webp"), b"not an image").unwrap();
        save(&photos.join("d-valid.png"), [0, 0, 255]);
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut model = FakeEmbedder { images: 0 };
        let mut updates = Vec::new();
        let result = catalog
            .index(
                &photos,
                &mut model,
                || false,
                |state| {
                    updates.push((state.current, state.failed, state.failure.clone()));
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(
            (result.indexed, result.failed, result.cancelled),
            (2, 2, false)
        );
        assert_eq!(model.images, 2);
        assert_eq!(result.failures_omitted, 0);
        assert_eq!(result.failures[0].path, jpeg.to_str().unwrap());
        assert!(result.failures[0]
            .error
            .contains("1 extraneous bytes before marker 0xd9"));
        assert!(result.failures[1].path.ends_with("c-invalid.webp"));
        assert_eq!(updates.iter().filter(|state| state.2.is_some()).count(), 2);
        assert_eq!(
            (updates.last().unwrap().0, updates.last().unwrap().1),
            (4, 2)
        );
        assert!(updates.last().unwrap().2.is_none());
        let before = catalog.browse(0, 100).unwrap();
        assert_eq!(before.total, 2);
        assert_eq!(before.items[1].filename, "d-valid.png");
        let result = catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!((result.unchanged, result.failed, model.images), (2, 2, 2));
        save(&jpeg, [255, 0, 0]);
        save(&photos.join("c-invalid.webp"), [255, 0, 0]);
        let result = catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!((result.indexed, result.unchanged, result.failed), (2, 2, 0));
        assert!(result.failures.is_empty());
        let after = catalog.browse(0, 100).unwrap();
        assert_eq!(after.total, 4);
        assert_eq!(before.items[0].asset_id, after.items[0].asset_id);
        assert_eq!(before.items[1].asset_id, after.items[3].asset_id);
    }

    #[test]
    fn failed_rescans_preserve_existing_entries_and_cancelled_scans_do_not_prune() {
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        save(&photos.join("b-existing.png"), [255, 0, 0]);
        save(&photos.join("c-removed.png"), [0, 0, 255]);
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut model = FakeEmbedder { images: 0 };
        catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        let existing = catalog.browse(0, 100).unwrap().items[0].clone();
        fs::write(photos.join("b-existing.png"), b"corruption").unwrap();
        fs::remove_file(photos.join("c-removed.png")).unwrap();
        save(&photos.join("a-new.png"), [255, 0, 0]);
        save(&photos.join("d-new.png"), [0, 0, 255]);
        let result = catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!((result.indexed, result.failed, result.removed), (2, 1, 1));
        let retained = &catalog.browse(0, 100).unwrap().items[1];
        assert_eq!(retained.asset_id, existing.asset_id);
        assert_eq!(retained.thumbnail_url, existing.thumbnail_url);
        assert!(catalog
            .preview(&existing.asset_id)
            .unwrap_err()
            .to_string()
            .contains("changed"));
        let mut query = vec![0.; EMBEDDING_DIM];
        query[0] = 1.;
        assert!(catalog
            .search(&query, 10)
            .unwrap()
            .iter()
            .any(|asset| asset.asset_id == existing.asset_id));

        // Cancel immediately after a reported failure, including when it was
        // the last visited file. Deleted entries must not be pruned on cancel.
        fs::remove_file(photos.join("d-new.png")).unwrap();
        let cancelled = std::cell::Cell::new(false);
        let result = catalog
            .index(
                &photos,
                &mut model,
                || cancelled.get(),
                |state| {
                    if state.failure.is_some() {
                        cancelled.set(true);
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert!(result.cancelled);
        assert_eq!((result.unchanged, result.failed, result.removed), (1, 1, 0));
        assert_eq!(result.failures.len(), 1);
        assert_eq!(catalog.status().unwrap().asset_count, 3);

        save(&photos.join("b-existing.png"), [0, 0, 255]);
        let result = catalog
            .index(&photos, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!((result.indexed, result.failed, result.removed), (1, 0, 1));
        assert_eq!(
            catalog.browse(0, 100).unwrap().items[1].asset_id,
            existing.asset_id
        );
    }

    #[test]
    fn disappearing_and_changing_source_files_are_reported_without_storing_partial_entries() {
        struct ChangingEmbedder {
            model: FakeEmbedder,
            path: PathBuf,
        }
        impl Embedder for ChangingEmbedder {
            fn embed_images(&mut self, images: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
                images.iter().map(|image| self.embed_image(image)).collect()
            }
            fn embed_image(&mut self, image: &RgbImage) -> Result<Vec<f32>> {
                if self.model.images == 0 {
                    fs::write(&self.path, b"changed while embedding").unwrap();
                }
                self.model.embed_image(image)
            }
            fn embed_text(&mut self, _: &str) -> Result<Vec<f32>> {
                bail!("test embedder has no text model")
            }
        }
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        for filename in ["a-vanished.png", "b-changing.png", "c-valid.png"] {
            save(&photos.join(filename), [255, 0, 0]);
        }
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut model = ChangingEmbedder {
            model: FakeEmbedder { images: 0 },
            path: photos.join("b-changing.png"),
        };
        let result = catalog
            .index(
                &photos,
                &mut model,
                || false,
                |state| {
                    if state.stage == "indexing" && state.current == 0 {
                        fs::remove_file(photos.join("a-vanished.png")).unwrap();
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!((result.indexed, result.failed, result.removed), (1, 2, 0));
        assert!(result.failures[0].path.ends_with("a-vanished.png"));
        assert!(result.failures[0].error.contains("Inspect media"));
        assert!(result.failures[1].error.contains("changed during indexing"));
        assert_eq!(
            catalog.browse(0, 100).unwrap().items[0].filename,
            "c-valid.png"
        );
        assert_eq!(model.model.images, 2);
    }

    #[test]
    fn failure_reports_are_bounded_but_every_bad_file_emits_an_error() {
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        for i in 0..MAX_REPORTED_FAILURES + 2 {
            fs::write(photos.join(format!("{i:03}.webp")), b"corruption").unwrap();
        }
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut model = FakeEmbedder { images: 0 };
        let mut reported = 0;
        let result = catalog
            .index(
                &photos,
                &mut model,
                || false,
                |state| {
                    if state.failure.is_some() {
                        reported += 1;
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(result.failed, MAX_REPORTED_FAILURES + 2);
        assert_eq!(reported, result.failed);
        assert_eq!(result.failures.len(), MAX_REPORTED_FAILURES);
        assert_eq!(result.failures_omitted, 2);
        assert_eq!(model.images, 0);
        assert_eq!(catalog.status().unwrap().asset_count, 0);
    }

    #[test]
    fn model_and_cache_failures_still_abort_instead_of_skipping_images() {
        struct FailingEmbedder;
        impl Embedder for FailingEmbedder {
            fn embed_images(&mut self, _: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
                bail!("model inference failed")
            }
            fn embed_image(&mut self, _: &RgbImage) -> Result<Vec<f32>> {
                bail!("model inference failed")
            }
            fn embed_text(&mut self, _: &str) -> Result<Vec<f32>> {
                bail!("test embedder has no text model")
            }
        }
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        save(&photos.join("valid.png"), [255, 0, 0]);
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut skipped = 0;
        let error = catalog
            .index(
                &photos,
                &mut FailingEmbedder,
                || false,
                |state| {
                    skipped += usize::from(state.failure.is_some());
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("model inference failed"));
        assert_eq!(skipped, 0);
        // A non-directory cache path gives a deterministic storage failure
        // without relying on permission bits (which root could bypass).
        fs::remove_dir(&catalog.thumbnails).unwrap();
        fs::write(&catalog.thumbnails, b"not a directory").unwrap();
        let mut model = FakeEmbedder { images: 0 };
        assert!(catalog
            .index(
                &photos,
                &mut model,
                || false,
                |state| {
                    skipped += usize::from(state.failure.is_some());
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(skipped, 0);
        assert_eq!(catalog.status().unwrap().asset_count, 0);
    }
    struct DocumentEmbedder<'a> {
        calls: usize,
        fail_at: Option<usize>,
        invalid_at: Option<usize>,
        cancel: Option<&'a std::cell::Cell<bool>>,
        mutate: Option<PathBuf>,
    }
    impl Embedder for DocumentEmbedder<'_> {
        fn embed_images(&mut self, _: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
            bail!("Unexpected image model")
        }
        fn embed_image(&mut self, _: &RgbImage) -> Result<Vec<f32>> {
            bail!("Unexpected image model")
        }
        fn embed_text(&mut self, _: &str) -> Result<Vec<f32>> {
            bail!("Documents must use the document task prefix")
        }
        fn document_token_count(&mut self, title: &str, text: &str) -> Result<usize> {
            Ok(title.len() + text.len() + 2)
        }
        fn embed_document(&mut self, title: &str, text: &str) -> Result<Vec<f32>> {
            assert!(!title.is_empty() && !text.is_empty());
            self.calls += 1;
            if self.fail_at == Some(self.calls) {
                bail!("Synthetic document model failure");
            }
            if let Some(cancel) = self.cancel {
                cancel.set(true);
            }
            if let Some(path) = self.mutate.take() {
                fs::write(path, b"Changed during inference")?;
            }
            if self.invalid_at == Some(self.calls) {
                return Ok(vec![f32::NAN; EMBEDDING_DIM]);
            }
            let mut vector = vec![0.; EMBEDDING_DIM];
            vector[2] = 1.;
            Ok(vector)
        }
    }
    fn document_embedder() -> DocumentEmbedder<'static> {
        DocumentEmbedder {
            calls: 0,
            fail_at: None,
            invalid_at: None,
            cancel: None,
            mutate: None,
        }
    }
    fn text_fixture() -> (tempfile::TempDir, PathBuf, PathBuf, Catalog) {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("documents");
        fs::create_dir(&source).unwrap();
        let path = source.join("chapter.txt");
        fs::write(
            &path,
            "A searchable document passage with bounded UTF-8 text. ".repeat(500),
        )
        .unwrap();
        let catalog = Catalog::open(&root.path().join("data")).unwrap();
        (root, source, path, catalog)
    }
    #[test]
    fn text_scope_finds_full_passages_and_titles_without_exposing_private_index_content() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("private-document-directory");
        fs::create_dir(&source).unwrap();
        let passage = format!("{} privatefullpassagebeacon", "ordinary text ".repeat(190));
        fs::write(source.join("filenamebeacon.md"), &passage).unwrap();
        fs::write(source.join("other.txt"), "Unrelated passage").unwrap();
        let data = root.path().join("data");
        let mut catalog = Catalog::open(&data).unwrap();
        catalog
            .index(&source, &mut document_embedder(), || false, |_| Ok(()))
            .unwrap();
        let mut vector = vec![0.0; EMBEDDING_DIM];
        vector[2] = 1.0;
        let original = catalog
            .browse_filtered(0, 100, None, Some(MediaKind::Text))
            .unwrap();
        assert_eq!(original.total, 2);
        assert!(original
            .items
            .iter()
            .all(|asset| asset.text_excerpt.as_ref().unwrap().len() <= MAX_TEXT_EXCERPT_BYTES));
        assert!(!serde_json::to_string(&original)
            .unwrap()
            .contains("privatefullpassagebeacon"));
        let before_semantic = catalog.search(&vector, 100).unwrap();
        assert!(before_semantic
            .iter()
            .all(|asset| asset.hybrid_score.is_none()));
        let agent = crate::agent::AgentCatalog::open(&data).unwrap();
        // Search has no dependency on the originals after the atomic row commit.
        fs::remove_dir_all(&source).unwrap();
        for query in ["privatefullpassagebeacon", "filenamebeacon"] {
            let desktop = catalog
                .search_filtered(query, &vector, 100, None, Some(MediaKind::Text))
                .unwrap();
            let remote = agent
                .search_filtered(query, &vector, 100, Some(MediaKind::Text))
                .unwrap();
            assert_eq!(desktop[0].filename, "filenamebeacon.md");
            assert_eq!(desktop[0].asset_id, remote[0].asset_id);
            assert_eq!(
                desktop
                    .iter()
                    .map(|asset| &asset.asset_id)
                    .collect::<Vec<_>>(),
                remote
                    .iter()
                    .map(|asset| &asset.asset_id)
                    .collect::<Vec<_>>()
            );
            assert!(desktop
                .windows(2)
                .all(|pair| pair[0].hybrid_score >= pair[1].hybrid_score));
            assert!(remote
                .iter()
                .all(|asset| asset.similarity.unwrap() > 0.9999));
            let json = serde_json::to_string(&remote).unwrap();
            for private in [
                "indexed_text",
                "indexedText",
                "private-document-directory",
                "\"path\"",
                "file://",
            ] {
                assert!(!json.contains(private), "Metadata leaked {private}");
            }
            assert!(json.contains("hybridScore"));
            assert!(
                !json.contains(&passage),
                "Full private passage must stay private"
            );
            if query == "privatefullpassagebeacon" {
                assert!(remote[0].text_excerpt.as_ref().unwrap().contains(query));
            }
            assert!(remote
                .iter()
                .all(|asset| asset.text_excerpt.as_ref().unwrap().len() <= MAX_TEXT_EXCERPT_BYTES));
        }
        assert_eq!(
            agent
                .browse_filtered(0, 10, Some(MediaKind::Text))
                .unwrap()
                .total,
            2
        );
        assert_eq!(
            agent
                .browse_filtered(0, 10, Some(MediaKind::Image))
                .unwrap()
                .total,
            0
        );
        let after_semantic = catalog
            .search_filtered("privatefullpassagebeacon", &vector, 100, None, None)
            .unwrap();
        assert_eq!(
            before_semantic
                .iter()
                .map(|asset| &asset.asset_id)
                .collect::<Vec<_>>(),
            after_semantic
                .iter()
                .map(|asset| &asset.asset_id)
                .collect::<Vec<_>>()
        );
        assert!(after_semantic
            .iter()
            .all(|asset| asset.hybrid_score.is_none()));
    }
    #[test]
    fn text_segments_have_stable_ids_bounded_metadata_and_atomic_replacement() {
        let (_root, source, path, mut catalog) = text_fixture();
        let mut model = document_embedder();
        let indexed = catalog
            .index(&source, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!(indexed.indexed, 1);
        let first = catalog.browse(0, 100).unwrap();
        assert!(first.total > 1 && first.total <= 100);
        assert_eq!(model.calls, first.total);
        for asset in &first.items {
            crate::agent::validate_asset_id(&asset.asset_id).unwrap();
            assert_eq!(asset.media_kind, "text");
            assert_eq!((asset.width, asset.height), (0, 0));
            assert!(asset.thumbnail_url.is_empty());
            assert!(asset.start_ms.is_none() && asset.end_ms.is_none());
            assert!(asset.text_excerpt.as_ref().unwrap().len() <= MAX_TEXT_EXCERPT_BYTES);
            assert!(catalog.preview(&asset.asset_id).is_err());
            assert_eq!(
                Url::parse(&catalog.original_url(&asset.asset_id).unwrap())
                    .unwrap()
                    .to_file_path()
                    .unwrap(),
                path
            );
        }
        let status = catalog.status().unwrap();
        assert_eq!((status.asset_count, status.file_count), (first.total, 1));
        assert_eq!(
            (status.sources[0].asset_count, status.sources[0].file_count),
            (first.total, 1)
        );
        let same = catalog
            .index(&source, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!(same.unchanged, 1);
        assert_eq!(model.calls, first.total);
        let first_id = first
            .items
            .iter()
            .find(|asset| asset.segment_index == 0)
            .unwrap()
            .asset_id
            .clone();
        fs::write(&path, "A shorter replacement document.").unwrap();
        catalog
            .index(&source, &mut model, || false, |_| Ok(()))
            .unwrap();
        let after = catalog.browse(0, 100).unwrap();
        assert_eq!(after.total, 1);
        assert_eq!(after.items[0].asset_id, first_id);
        let mut query = vec![0.; EMBEDDING_DIM];
        query[2] = 1.;
        assert_eq!(catalog.search(&query, 10).unwrap()[0].asset_id, first_id);
        fs::remove_file(path).unwrap();
        let removed = catalog
            .index(&source, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!(removed.removed, 1);
        assert_eq!(catalog.status().unwrap().file_count, 0);
    }
    #[test]
    fn text_model_failure_invalid_vector_cancellation_and_change_preserve_all_old_segments() {
        let (_root, source, path, mut catalog) = text_fixture();
        catalog
            .index(&source, &mut document_embedder(), || false, |_| Ok(()))
            .unwrap();
        let original = serde_json::to_string(&catalog.browse(0, 100).unwrap()).unwrap();
        for invalid in [false, true] {
            fs::write(&path, "A new longer replacement passage. ".repeat(1000)).unwrap();
            let mut model = document_embedder();
            if invalid {
                model.invalid_at = Some(2);
            } else {
                model.fail_at = Some(2);
            }
            assert!(catalog
                .index(&source, &mut model, || false, |_| Ok(()))
                .is_err());
            assert_eq!(model.calls, 2);
            assert_eq!(
                serde_json::to_string(&catalog.browse(0, 100).unwrap()).unwrap(),
                original
            );
        }
        let cancel = std::cell::Cell::new(false);
        let mut model = DocumentEmbedder {
            cancel: Some(&cancel),
            ..document_embedder()
        };
        let stopped = catalog
            .index(&source, &mut model, || cancel.get(), |_| Ok(()))
            .unwrap();
        assert!(stopped.cancelled);
        assert_eq!((stopped.indexed, stopped.removed), (0, 0));
        assert_eq!(
            serde_json::to_string(&catalog.browse(0, 100).unwrap()).unwrap(),
            original
        );
        let mut model = DocumentEmbedder {
            mutate: Some(path.clone()),
            ..document_embedder()
        };
        let changed = catalog
            .index(&source, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!((changed.failed, changed.indexed), (1, 0));
        assert_eq!(
            serde_json::to_string(&catalog.browse(0, 100).unwrap()).unwrap(),
            original
        );
        fs::write(&path, [0xff, 0xfe, 0x00]).unwrap();
        let corrupt = catalog
            .index(&source, &mut document_embedder(), || false, |_| Ok(()))
            .unwrap();
        assert_eq!((corrupt.failed, corrupt.indexed), (1, 0));
        assert_eq!(
            serde_json::to_string(&catalog.browse(0, 100).unwrap()).unwrap(),
            original
        );
    }
    #[test]
    fn cancelling_multisegment_file_never_prunes_an_unvisited_missing_file() {
        let (_root, source, path, mut catalog) = text_fixture();
        let missing = source.join("z.txt");
        fs::write(&missing, "Retain this indexed file.").unwrap();
        catalog
            .index(&source, &mut document_embedder(), || false, |_| Ok(()))
            .unwrap();
        let original = serde_json::to_string(&catalog.browse(0, 100).unwrap()).unwrap();
        fs::remove_file(missing).unwrap();
        fs::write(&path, "Changed text. ".repeat(1000)).unwrap();
        let cancel = std::cell::Cell::new(false);
        let mut model = DocumentEmbedder {
            cancel: Some(&cancel),
            ..document_embedder()
        };
        let stopped = catalog
            .index(&source, &mut model, || cancel.get(), |_| Ok(()))
            .unwrap();
        assert!(stopped.cancelled);
        assert_eq!(stopped.removed, 0);
        assert_eq!(
            serde_json::to_string(&catalog.browse(0, 100).unwrap()).unwrap(),
            original
        );
        assert_eq!(catalog.status().unwrap().file_count, 2);
    }
    struct MediaEmbedder {
        audio_calls: usize,
        video_calls: usize,
    }
    impl Embedder for MediaEmbedder {
        fn embed_images(&mut self, _: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
            bail!("Unexpected image embedding")
        }
        fn embed_image(&mut self, _: &RgbImage) -> Result<Vec<f32>> {
            bail!("Unexpected image embedding")
        }
        fn embed_text(&mut self, _: &str) -> Result<Vec<f32>> {
            bail!("Unexpected query embedding")
        }
        fn embed_audio(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
            assert!(samples.len() > 160 && samples.len() <= 30 * 16_000);
            assert!(samples.iter().all(|sample| sample.is_finite()));
            self.audio_calls += 1;
            let mut vector = vec![0.; EMBEDDING_DIM];
            vector[3] = 1.;
            Ok(vector)
        }
        fn embed_video(&mut self, frames: &[RgbImage]) -> Result<Vec<f32>> {
            assert!(!frames.is_empty() && frames.len() <= 30);
            self.video_calls += 1;
            let mut vector = vec![0.; EMBEDDING_DIM];
            vector[4] = 1.;
            Ok(vector)
        }
    }
    #[test]
    #[ignore = "requires installed ffmpeg and ffprobe; uses only temporary generated media and fake embeddings"]
    fn native_audio_video_and_soundtrack_segments_share_one_catalog() {
        use std::process::Command;
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("media");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("index.ts"), "export const count = 1;\n").unwrap();
        fs::write(source.join("types.d.ts"), "export interface Item {}\n").unwrap();
        let video = source.join("movie.mkv");
        let created = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=32x24:r=1:d=31",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000:duration=31",
                "-t",
                "31",
                "-c:v",
                "ffv1",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&video)
            .status()
            .unwrap();
        assert!(created.success());
        let audio = source.join("speech.wav");
        let created = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=220:sample_rate=16000:duration=31",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&audio)
            .status()
            .unwrap();
        assert!(created.success());
        let data = root.path().join("data");
        let mut catalog = Catalog::open(&data).unwrap();
        let mut model = MediaEmbedder {
            audio_calls: 0,
            video_calls: 0,
        };
        let result = catalog
            .index(&source, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!((result.indexed, result.failed), (2, 0));
        assert_eq!((model.audio_calls, model.video_calls), (4, 2));
        let status = catalog.status().unwrap();
        assert_eq!((status.asset_count, status.file_count), (6, 2));
        let page = catalog.browse(0, 100).unwrap();
        for asset in &page.items {
            assert_eq!(
                asset.start_ms,
                Some(u64::from(asset.segment_index) * 30_000)
            );
            assert!(asset.end_ms.unwrap() > asset.start_ms.unwrap());
            assert!(asset.text_excerpt.is_none());
            if asset.media_kind == "video" {
                assert_eq!((asset.width, asset.height), (32, 24));
                assert!(asset.thumbnail_url.starts_with("file://"));
            } else {
                assert_eq!(asset.media_kind, "audio");
                assert_eq!((asset.width, asset.height), (0, 0));
                assert!(asset.thumbnail_url.is_empty());
            }
        }
        let mut query = vec![0.; EMBEDDING_DIM];
        query[3] = 1.;
        assert!(catalog
            .search(&query, 4)
            .unwrap()
            .iter()
            .all(|asset| asset.media_kind == "audio"));
        // Shortening only the soundtrack creates a tiny tail and changes its
        // segment boundary. The (modality, index) UUID must remain stable.
        let replaced = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=32x24:r=1:d=31",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000:duration=30.001",
                "-t",
                "31",
                "-c:v",
                "ffv1",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&video)
            .status()
            .unwrap();
        assert!(replaced.success());
        let updated = catalog
            .index(&source, &mut model, || false, |_| Ok(()))
            .unwrap();
        assert_eq!(
            (updated.indexed, updated.unchanged, updated.failed),
            (1, 1, 0)
        );
        let after = catalog.browse(0, 100).unwrap();
        for old in page
            .items
            .iter()
            .filter(|asset| asset.filename == "movie.mkv")
        {
            let new = after
                .items
                .iter()
                .find(|asset| {
                    asset.filename == old.filename
                        && asset.media_kind == old.media_kind
                        && asset.segment_index == old.segment_index
                })
                .unwrap();
            assert_eq!(new.asset_id, old.asset_id);
            if new.media_kind == "audio" && new.segment_index == 1 {
                assert!(new.start_ms.unwrap() < old.start_ms.unwrap());
                assert_eq!(new.end_ms, Some(30_001));
            }
        }
        fs::remove_dir_all(&source).unwrap();
        let agent = crate::agent::AgentCatalog::open(&data).unwrap();
        for asset in agent.browse(0, 100).unwrap().items {
            if let Some(handle) = asset.thumbnail_handle {
                assert!(!agent.thumbnail(&handle).unwrap().is_empty());
            }
            assert!(asset.start_ms.is_some() && asset.end_ms.is_some());
        }
    }

    // Catalog module test, retained verbatim from the initial visible multimodal source
    // except the later recorded image_count -> asset_count field rename.
    #[test]
    fn invalid_batch_embeddings_are_fatal_without_partial_commits_or_retries() {
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        for index in 0..4 {
            save(&photos.join(format!("{index}.png")), [255, 0, 0]);
        }
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        for vector in [
            vec![],
            vec![1.; EMBEDDING_DIM - 1],
            vec![1.; EMBEDDING_DIM + 1],
            vec![0.; EMBEDDING_DIM],
            vec![1.; EMBEDDING_DIM],
            vec![f32::NAN; EMBEDDING_DIM],
            vec![f32::INFINITY; EMBEDDING_DIM],
            vec![f32::NEG_INFINITY; EMBEDDING_DIM],
        ] {
            let mut model = BatchEmbedder {
                invalid_embedding: Some(vector),
                ..batch_embedder()
            };
            let error = catalog
                .index_with_batch_size(&photos, &mut model, 4, || false, |_| Ok(()))
                .unwrap_err();
            assert!(error.to_string().contains("embedding"));
            assert_eq!(model.calls.len(), 1);
            assert_eq!(catalog.status().unwrap().asset_count, 0);
            assert!(fs::read_dir(&catalog.thumbnails).unwrap().next().is_none());
        }
    }
    struct ReuseEmbedder {
        calls: usize,
    }
    impl Embedder for ReuseEmbedder {
        fn embedding_profile(&self) -> Option<String> {
            Some("synthetic-reuse-test-v1".into())
        }
        fn document_token_count(&mut self, title: &str, text: &str) -> Result<usize> {
            Ok(title.len() + text.len() + 2)
        }
        fn embed_image(&mut self, _: &RgbImage) -> Result<Vec<f32>> {
            bail!("unexpected image")
        }
        fn embed_images(&mut self, _: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
            bail!("unexpected images")
        }
        fn embed_text(&mut self, _: &str) -> Result<Vec<f32>> {
            bail!("unexpected query")
        }
        fn embed_document(&mut self, _: &str, _: &str) -> Result<Vec<f32>> {
            self.calls += 1;
            let mut vector = vec![0.; EMBEDDING_DIM];
            vector[0] = 1.;
            Ok(vector)
        }
    }
    #[test]
    fn content_touch_reuses_and_small_edit_reuses_unchanged_passages() {
        let root = tempfile::tempdir().unwrap();
        let docs = root.path().join("docs");
        fs::create_dir(&docs).unwrap();
        let path = docs.join("guide.md");
        let sections = (0..8)
            .map(|i| {
                format!(
                    "# Section {i}\n\n{}\n\n",
                    format!("passage{i} useful content. ").repeat(110)
                )
            })
            .collect::<String>();
        fs::write(&path, &sections).unwrap();
        let mut catalog = Catalog::open(&root.path().join("data")).unwrap();
        let mut embedder = ReuseEmbedder { calls: 0 };
        catalog
            .index(&docs, &mut embedder, || false, |_| Ok(()))
            .unwrap();
        let first = embedder.calls;
        assert!(first > 3);
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(
                std::fs::FileTimes::new()
                    .set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(1)),
            )
            .unwrap();
        let touched = catalog
            .index(&docs, &mut embedder, || false, |_| Ok(()))
            .unwrap();
        assert_eq!(touched.unchanged, 1);
        assert_eq!(embedder.calls, first);
        fs::write(&path, sections.replacen("passage0", "changed0", 1)).unwrap();
        catalog
            .index(&docs, &mut embedder, || false, |_| Ok(()))
            .unwrap();
        assert!(embedder.calls > first);
        assert!(
            embedder.calls < first * 2,
            "unchanged chunks must reuse vectors"
        );
        let calls = embedder.calls;
        fs::rename(&path, docs.join("renamed.md")).unwrap();
        catalog
            .index(&docs, &mut embedder, || false, |_| Ok(()))
            .unwrap();
        assert!(
            embedder.calls > calls,
            "title is part of the exact embedding input"
        );
        assert_eq!(catalog.status().unwrap().file_count, 1);
    }
    #[test]
    fn verification_scan_detects_restored_old_timestamp_and_keeps_locations() {
        let root = tempfile::tempdir().unwrap();
        let docs = root.path().join("docs");
        fs::create_dir(&docs).unwrap();
        let path = docs.join("guide.txt");
        fs::write(&path, "original source\nsecond line").unwrap();
        let old = UNIX_EPOCH + std::time::Duration::from_secs(1000);
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(old))
            .unwrap();
        let mut c = Catalog::open(&root.path().join("data")).unwrap();
        let mut e = ReuseEmbedder { calls: 0 };
        c.index(&docs, &mut e, || false, |_| Ok(())).unwrap();
        fs::write(&path, "modified source\nsecond line").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(old))
            .unwrap();
        c.index_with_options(
            &docs,
            &mut e,
            1,
            IndexOptions {
                verify_content: true,
                ..IndexOptions::default()
            },
            || false,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(e.calls, 2);
        let page = c
            .browse_filtered(0, 10, None, Some(MediaKind::Text))
            .unwrap();
        assert_eq!(page.items[0].start_byte, Some(0));
        assert_eq!(page.items[0].start_line, Some(1));
    }
    #[test]
    fn same_stat_rebuild_invalidates_old_sidecar_generation() {
        let root = tempfile::tempdir().unwrap();
        let docs = root.path().join("docs");
        fs::create_dir(&docs).unwrap();
        let path = docs.join("guide.txt");
        fs::write(&path, "original source").unwrap();
        let old = UNIX_EPOCH + std::time::Duration::from_secs(1000);
        let set_time = || {
            fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(old))
                .unwrap()
        };
        set_time();
        let mut c = Catalog::open(&root.path().join("data")).unwrap();
        let mut e = ReuseEmbedder { calls: 0 };
        c.index(&docs, &mut e, || false, |_| Ok(())).unwrap();
        let source = c.status().unwrap().sources[0].id;
        let previous = c.store.source_manifest(source).unwrap();
        let old_state = c
            .index_state
            .load(source, text_path(&path).unwrap(), &previous)
            .unwrap()
            .unwrap();
        fs::write(&path, "modified source").unwrap();
        set_time();
        c.index_with_options(
            &docs,
            &mut e,
            1,
            IndexOptions {
                verify_content: true,
                ..IndexOptions::default()
            },
            || false,
            |_| Ok(()),
        )
        .unwrap();
        let current = c.store.source_manifest(source).unwrap();
        assert_eq!(previous[0].asset_id, current[0].asset_id);
        assert_ne!(previous[0].modified, current[0].modified);
        // Simulate loss of the post-commit sidecar publish.
        c.index_state
            .save(source, text_path(&path).unwrap(), &old_state)
            .unwrap();
        assert!(c
            .index_state
            .load(source, text_path(&path).unwrap(), &current)
            .unwrap()
            .is_none());
        assert!(c
            .browse_filtered(0, 10, None, Some(MediaKind::Text))
            .unwrap()
            .items[0]
            .start_byte
            .is_none());
        let calls = e.calls;
        c.index(&docs, &mut e, || false, |_| Ok(())).unwrap();
        assert_eq!(
            e.calls, calls,
            "Recover sidecar through exact-input vector cache"
        );
        assert_eq!(
            c.browse_filtered(0, 10, None, Some(MediaKind::Text))
                .unwrap()
                .items[0]
                .start_byte,
            Some(0)
        );
    }
    #[test]
    fn content_identical_image_touch_keeps_preview_available_without_inference() {
        let root = tempfile::tempdir().unwrap();
        let docs = root.path().join("images");
        fs::create_dir(&docs).unwrap();
        let path = docs.join("image.png");
        save(&path, [12, 34, 56]);
        let mut c = Catalog::open(&root.path().join("data")).unwrap();
        let mut e = FakeEmbedder { images: 0 };
        c.index(&docs, &mut e, || false, |_| Ok(())).unwrap();
        let id = c.browse(0, 10).unwrap().items[0].asset_id.clone();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(
                std::fs::FileTimes::new()
                    .set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(1)),
            )
            .unwrap();
        assert_eq!(
            c.index(&docs, &mut e, || false, |_| Ok(()))
                .unwrap()
                .unchanged,
            1
        );
        assert_eq!(e.images, 1);
        assert!(c.preview(&id).is_ok());
    }
    #[test]
    fn racy_admission_stays_verified_after_slow_same_stat_rewrite() {
        struct SlowRewrite {
            path: PathBuf,
            modified: std::time::SystemTime,
        }
        impl Embedder for SlowRewrite {
            fn embed_image(&mut self, _: &RgbImage) -> Result<Vec<f32>> {
                bail!("unexpected")
            }
            fn embed_images(&mut self, _: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
                bail!("unexpected")
            }
            fn embed_text(&mut self, _: &str) -> Result<Vec<f32>> {
                bail!("unexpected")
            }
            fn document_token_count(&mut self, t: &str, s: &str) -> Result<usize> {
                Ok(t.len() + s.len() + 2)
            }
            fn embed_document(&mut self, _: &str, _: &str) -> Result<Vec<f32>> {
                fs::write(&self.path, "third document")?;
                fs::File::options()
                    .write(true)
                    .open(&self.path)?
                    .set_times(std::fs::FileTimes::new().set_modified(self.modified))?;
                std::thread::sleep(std::time::Duration::from_millis(2100));
                let mut v = vec![0.; EMBEDDING_DIM];
                v[0] = 1.;
                Ok(v)
            }
        }
        let root = tempfile::tempdir().unwrap();
        let docs = root.path().join("docs");
        fs::create_dir(&docs).unwrap();
        let path = docs.join("racy.txt");
        fs::write(&path, "first document").unwrap();
        let mut c = Catalog::open(&root.path().join("data")).unwrap();
        c.index(&docs, &mut ReuseEmbedder { calls: 0 }, || false, |_| Ok(()))
            .unwrap();
        let old = c.browse(0, 10).unwrap().items[0].text_excerpt.clone();
        fs::write(&path, "other document").unwrap();
        let modified = fs::metadata(&path).unwrap().modified().unwrap();
        let result = c
            .index(
                &docs,
                &mut SlowRewrite {
                    path: path.clone(),
                    modified,
                },
                || false,
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(result.failed, 1);
        assert_eq!(result.indexed, 0);
        assert_eq!(c.browse(0, 10).unwrap().items[0].text_excerpt, old);
    }
}
