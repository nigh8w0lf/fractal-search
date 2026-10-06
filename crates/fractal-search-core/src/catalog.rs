//! LanceDB catalog and incremental indexing. Originals are never modified.
use crate::store::{image_filter, quote, Record, Store};
use crate::{
    preprocess::{decode_image, patch_budget},
    Embedder, EMBEDDING_DIM, MAX_IMAGE_BATCH_SIZE, MODEL_PROFILE,
};
use anyhow::{ensure, Context, Result};
use image::{codecs::webp::WebPEncoder, ExtendedColorType, RgbImage};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
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

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    pub id: i64,
    pub directory: String,
    pub image_count: usize,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogStatus {
    pub image_count: usize,
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
    pub similarity: Option<f32>,
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
        eprintln!("[Index] Skipped image {path:?}: {message}");
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
#[derive(Debug, PartialEq, Eq)]
struct Stamp {
    bytes: u64,
    modified: String,
}

struct PendingImage {
    path: PathBuf,
    before: Stamp,
    existing: Option<Record>,
    decode_ms: f64,
    start: Instant,
}

struct IndexBatch {
    source_id: i64,
    images: Vec<RgbImage>,
    pending: Vec<PendingImage>,
    budget: Option<usize>,
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
        let vectors = embedder.embed_images(&self.images).with_context(|| {
            format!(
                "Embed image batch of {} starting at {:?}; reduce Image batch size if device memory is insufficient",
                self.images.len(), self.pending[0].path
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
        self.budget = None;
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
}

fn text_path(path: &Path) -> Result<&str> {
    path.to_str()
        .context("Image search requires UTF-8 filesystem paths")
}
fn file_url(path: &Path) -> Result<String> {
    Url::from_file_path(path)
        .map(|u| u.to_string())
        .map_err(|_| anyhow::anyhow!("Cannot create file URL for {path:?}"))
}
fn stamp(path: &Path) -> Result<Stamp> {
    let metadata = fs::metadata(path).with_context(|| format!("Inspect image {path:?}"))?;
    ensure!(metadata.is_file(), "Image is not a regular file: {path:?}");
    let time = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .context("Image modification time predates Unix epoch")?;
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
        values.len() == EMBEDDING_DIM && values.iter().all(|v| v.is_finite()),
        "Invalid 768-D image-search embedding"
    );
    let norm: f64 = values
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        (norm - 1.0).abs() < 0.0001,
        "Image-search embedding must be normalized (norm={norm})"
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
        })
    }
    pub fn status(&self) -> Result<CatalogStatus> {
        let (image_count, rows) = self.store.status()?;
        let mut sources = rows
            .into_iter()
            .map(|(row, image_count)| Source {
                id: row.source_id,
                directory: row.path,
                image_count,
            })
            .collect::<Vec<_>>();
        sources.sort_by(|a, b| a.directory.cmp(&b.directory));
        Ok(CatalogStatus {
            image_count,
            sources,
            model_profile: MODEL_PROFILE,
        })
    }
    fn asset(row: Record) -> (Asset, PathBuf) {
        (
            Asset {
                asset_id: row.asset_id,
                filename: row.filename,
                path: row.path,
                thumbnail_url: String::new(),
                width: row.width,
                height: row.height,
                similarity: None,
            },
            PathBuf::from(row.thumbnail),
        )
    }
    fn complete_asset(&self, (mut asset, thumbnail): (Asset, PathBuf)) -> Result<Asset> {
        ensure!(
            thumbnail.starts_with(&self.thumbnails) && thumbnail.is_file(),
            "Thumbnail is unavailable for {}; rescan its folder",
            asset.path
        );
        let dimensions = image::image_dimensions(&thumbnail)
            .with_context(|| format!("Read cached thumbnail dimensions {thumbnail:?}"))?;
        if dimensions != bounded_dimensions(asset.width, asset.height, 384) {
            ensure!(
                dimensions == (384, 384),
                "Cached thumbnail dimensions are invalid: {thumbnail:?}; rescan its source folder"
            );
            // Repair the earlier square-cache format lazily, without loading
            // models or regenerating stored embeddings.
            let record = self.store.asset(&asset.asset_id, false)?;
            let (bytes, modified) = (record.bytes, record.modified);
            let original = Path::new(&asset.path);
            let indexed_stamp = Stamp { bytes, modified };
            ensure!(
                stamp(original)? == indexed_stamp,
                "Image changed since indexing; rescan its folder before repairing its thumbnail"
            );
            let image = decode_image(original)?;
            ensure!(
                stamp(original)? == indexed_stamp,
                "Image changed while repairing its thumbnail; rescan its folder"
            );
            write_atomic(&thumbnail, &self::thumbnail(&image)?)?;
            eprintln!(
                "[Cache] Repaired legacy square thumbnail for {:?}",
                asset.asset_id
            );
        }
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
        validate_page(offset, limit)?;
        if let Some(id) = source_id {
            self.source_directory(id)?;
        }
        let (total, rows) = self.store.browse(offset, limit, source_id)?;
        let items = rows
            .into_iter()
            .map(|r| self.complete_asset(Self::asset(r)))
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
        ensure!(
            (1..=MAX_RESULTS).contains(&limit),
            "Search result limit must be between 1 and {MAX_RESULTS}"
        );
        if let Some(id) = source_id {
            self.source_directory(id)?;
        }
        let rows = self.store.search(query_embedding, limit, source_id)?;
        let mut assets = Vec::with_capacity(rows.len());
        for (asset, distance) in rows {
            ensure!(
                distance.is_finite(),
                "Vector search returned a non-finite distance"
            );
            let mut asset = self.complete_asset(Self::asset(asset))?;
            asset.similarity = Some((1.0 - distance).clamp(-1.0, 1.0));
            assets.push(asset);
        }
        assets.sort_by(|a, b| {
            b.similarity
                .expect("search score")
                .total_cmp(&a.similarity.expect("search score"))
                .then(a.asset_id.cmp(&b.asset_id))
        });
        Ok(assets)
    }
    /// Remove the source and its images in one Lance commit, then delete caches.
    pub fn remove_source(&mut self, source_id: i64) -> Result<usize> {
        self.source_directory(source_id)?;
        let rows = self
            .store
            .list(&image_filter(Some(source_id)), MAX_FILES + 1, false)?;
        ensure!(rows.len() <= MAX_FILES, "Source exceeds image limit");
        self.store.delete(&format!("source_id = {source_id}"))?;
        for row in &rows {
            self.remove_cache(Path::new(&row.thumbnail))?;
            self.remove_cache(&self.previews.join(format!("{}.png", row.asset_id)))?;
        }
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
            "Choose an image folder outside the search application's data directory"
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
            let extension = entry
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !matches!(
                extension.as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "gif" | "bmp" | "tif" | "tiff" | "heic" | "heif"
            ) {
                continue;
            }
            paths.push(entry.path().canonicalize()?);
            ensure!(
                paths.len() <= MAX_FILES,
                "Folder scan exceeded {MAX_FILES} images; select a smaller folder"
            );
        }
        paths.sort();
        state.stage = "indexing";
        state.total = paths.len();
        progress(&state)?;
        let mut batch = IndexBatch {
            source_id,
            images: Vec::with_capacity(batch_size),
            pending: Vec::with_capacity(batch_size),
            budget: None,
        };
        for path in &paths {
            if cancelled() {
                return Ok(state.result(0, true, failures));
            }
            let start = Instant::now();
            let path_text = text_path(path)?;
            let before = match stamp(path) {
                Ok(stamp) => stamp,
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
            let existing = self.store.get(
                &format!("kind = 'image' AND path = {}", quote(path_text)),
                false,
            )?;
            let unchanged = existing
                .as_ref()
                .is_some_and(|row| row.bytes == before.bytes && row.modified == before.modified);
            if unchanged
                && existing
                    .as_ref()
                    .is_some_and(|row| Path::new(&row.thumbnail).is_file())
            {
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
                state.failure = None;
                state.filename = path
                    .file_name()
                    .context("Image filename missing")?
                    .to_string_lossy()
                    .into_owned();
                state.unchanged += 1;
                state.current += 1;
                progress(&state)?;
                continue;
            }
            let image = match decode_image(path) {
                Ok(image) => image,
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
            let pending = PendingImage {
                path: path.clone(),
                before,
                existing,
                decode_ms: start.elapsed().as_secs_f64() * 1000.0,
                start,
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
            let budget = patch_budget(image.width(), image.height())?;
            // prepare_images uses a shared patch budget. Only group images
            // with the same individual budget to preserve existing embeddings.
            if batch.budget.is_some_and(|previous| previous != budget)
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
            batch.budget = Some(budget);
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
        // Only a completed traversal may prune missing files. Known files that
        // failed decoding stay in `seen`, preserving their last good entries.
        let seen: HashSet<_> = paths
            .iter()
            .map(|p| text_path(p).map(str::to_owned))
            .collect::<Result<_>>()?;
        let previous = self
            .store
            .list(&image_filter(Some(source_id)), MAX_FILES + 1, false)?;
        ensure!(previous.len() <= MAX_FILES, "Source exceeds image limit");
        let stale = previous
            .into_iter()
            .filter(|row| !seen.contains(&row.path))
            .collect::<Vec<_>>();
        let mut removed = 0;
        for rows in stale.chunks(1000) {
            let ids = rows
                .iter()
                .map(|row| quote(&row.asset_id))
                .collect::<Vec<_>>()
                .join(",");
            self.store
                .delete(&format!("kind = 'image' AND asset_id IN ({ids})"))?;
            for row in rows {
                self.remove_cache(Path::new(&row.thumbnail))?;
                self.remove_cache(&self.previews.join(format!("{}.png", row.asset_id)))?;
                removed += 1;
            }
        }
        state.stage = "complete";
        state.failure = None;
        progress(&state)?;
        Ok(state.result(removed, false, failures))
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
        } = pending;
        let source_check = stamp(&path).and_then(|after| {
            ensure!(
                before == after,
                "Image changed during indexing: {path:?}; rescan after it finishes changing"
            );
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
        let digest = format!(
            "{:x}",
            Sha256::digest(format!("{}:{}", before.bytes, before.modified).as_bytes())
        );
        let cache = self
            .thumbnails
            .join(format!("{asset_id}-{}.webp", &digest[..16]));
        let thumb_start = Instant::now();
        write_atomic(&cache, &thumbnail(image)?)?;
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("Image has no UTF-8 filename")?;
        self.store.upsert(&Record {
            kind: "image".into(),
            asset_id: asset_id.clone(),
            source_id,
            path: text_path(&path)?.to_owned(),
            filename: filename.to_owned(),
            width: image.width(),
            height: image.height(),
            bytes: before.bytes,
            modified: before.modified.clone(),
            thumbnail: text_path(&cache)?.to_owned(),
            embedding: Some(embedding),
        })?;
        if let Some(old) = existing
            .as_ref()
            .filter(|r| r.thumbnail != text_path(&cache).expect("validated cache path"))
        {
            self.remove_cache(Path::new(&old.thumbnail))?;
        }
        self.remove_cache(&self.previews.join(format!("{asset_id}.png")))?;
        state.indexed += 1;
        state.current += 1;
        let reused = existing
            .as_ref()
            .is_some_and(|row| row.bytes == before.bytes && row.modified == before.modified);
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
        let (path, bytes, modified, width, height) = (
            record.path,
            record.bytes,
            record.modified,
            record.width,
            record.height,
        );
        let path = Path::new(&path);
        let indexed_stamp = Stamp { bytes, modified };
        ensure!(
            stamp(path)? == indexed_stamp,
            "Image changed since indexing; rescan its folder before previewing it"
        );
        let preview = self.previews.join(format!("{asset_id}.png"));
        let repair_square = if preview.is_file() {
            let dimensions = image::image_dimensions(&preview)
                .with_context(|| format!("Read cached preview dimensions {preview:?}"))?;
            let outdated = dimensions != bounded_dimensions(width, height, 1600);
            ensure!(
                !outdated || dimensions == (1600, 1600),
                "Cached preview dimensions are invalid: {preview:?}; rescan its source folder"
            );
            outdated
        } else {
            false
        };
        if !preview.is_file() || repair_square {
            let image = decode_image(path)?;
            let pixels = bounded_thumbnail(&image, 1600);
            let mut bytes = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(pixels).write_to(&mut bytes, image::ImageFormat::Png)?;
            ensure!(
                stamp(path)? == indexed_stamp,
                "Image changed while generating its preview; rescan its folder"
            );
            write_atomic(&preview, &bytes.into_inner())?;
            if repair_square {
                eprintln!("[Cache] Repaired legacy square preview for {asset_id:?}");
            }
        }
        file_url(&preview)
    }
    pub fn original_url(&self, asset_id: &str) -> Result<String> {
        crate::agent::validate_asset_id(asset_id)?;
        let path = self.store.asset(asset_id, false)?.path;
        ensure!(
            Path::new(&path).is_file(),
            "Original image is missing: {path}"
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
            assert!(budgets.iter().all(|budget| *budget == budgets[0]));
            self.calls.push(budgets);
            if let Some(cancel) = self.cancel {
                cancel.set(true);
            }
            let mut vectors = self.model.embed_images(images)?;
            if self.invalid_output {
                vectors.pop();
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
        }
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
    fn indexing_batches_keep_individual_patch_budgets_and_report_corrupt_files() {
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
        assert_eq!(model.calls, [vec![128, 128], vec![256, 256], vec![1024]]);
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
        assert_eq!(catalog.status().unwrap().image_count, 5);
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
        assert_eq!(catalog.status().unwrap().image_count, 0);
    }
    #[test]
    fn cached_images_preserve_aspect_ratio_and_repair_legacy_square_files() {
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
            catalog.browse(0, 10).unwrap();
            catalog.preview(&asset.asset_id).unwrap();
            assert_eq!(
                image::image_dimensions(&thumbnail).unwrap(),
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
        assert_eq!(catalog.status().unwrap().image_count, 1);
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
        assert_eq!(catalog.status().unwrap().image_count, 2);
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
                .image_count,
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
        assert_eq!(catalog.status().unwrap().image_count, 3);

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
        assert!(result.failures[0].error.contains("Inspect image"));
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
        assert_eq!(catalog.status().unwrap().image_count, 0);
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
        assert_eq!(catalog.status().unwrap().image_count, 0);
    }
}
