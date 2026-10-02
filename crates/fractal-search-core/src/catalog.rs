//! Search-only SQLite catalog and incremental indexing. Originals are never modified.
use crate::{preprocess::decode_image, Embedder, EMBEDDING_DIM, MODEL_PROFILE};
use anyhow::{ensure, Context, Result};
use image::{codecs::webp::WebPEncoder, ExtendedColorType, RgbImage};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::{Instant, UNIX_EPOCH},
};
use url::Url;
use uuid::Uuid;
use walkdir::WalkDir;

pub(crate) const APPLICATION_ID: i64 = 0x46535243;
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

#[derive(Debug)]
pub struct Catalog {
    conn: Connection,
    thumbnails: PathBuf,
    previews: PathBuf,
}

pub(crate) fn register_vectors() -> Result<()> {
    static REGISTER: OnceLock<Result<(), String>> = OnceLock::new();
    REGISTER
        .get_or_init(|| {
            // sqlite-vec exports the SQLite extension entry point; SQLite retains it
            // for every subsequently opened connection in this process.
            let code = unsafe {
                rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                    sqlite_vec::sqlite3_vec_init as *const (),
                )))
            };
            if code == rusqlite::ffi::SQLITE_OK {
                Ok(())
            } else {
                Err(format!("Register sqlite-vec: SQLite error {code}"))
            }
        })
        .as_ref()
        .map_err(|error| anyhow::anyhow!(error.clone()))?;
    Ok(())
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
pub(crate) fn vector_bytes(values: &[f32]) -> Result<Vec<u8>> {
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
    Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
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
        register_vectors()?;
        fs::create_dir_all(data_dir)
            .with_context(|| format!("Create search data directory {data_dir:?}"))?;
        let data_dir = data_dir.canonicalize()?;
        let mut conn = Connection::open(data_dir.join("search.db"))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let id: i64 = conn.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if version == 0 {
            let tables: usize = conn.query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table'",
                [],
                |r| r.get(0),
            )?;
            ensure!(
                id == 0 && tables == 0,
                "Refusing to initialize an existing unrelated database as an image-search catalog"
            );
            let tx = conn.transaction()?;
            tx.execute_batch("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE sources (id INTEGER PRIMARY KEY, directory TEXT NOT NULL UNIQUE);
                CREATE TABLE images (id INTEGER PRIMARY KEY, asset_id TEXT NOT NULL UNIQUE,
                source_id INTEGER NOT NULL REFERENCES sources(id), path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                bytes INTEGER NOT NULL, modified TEXT NOT NULL, thumbnail TEXT NOT NULL);
                CREATE INDEX images_source ON images(source_id);
                CREATE VIRTUAL TABLE image_vectors USING vec0(embedding float[768] distance_metric=cosine);
                PRAGMA application_id=1179865667; PRAGMA user_version=1;")?;
            tx.execute(
                "INSERT INTO settings VALUES ('model_profile', ?1)",
                [MODEL_PROFILE],
            )?;
            tx.commit()?;
        } else {
            ensure!(
                id == APPLICATION_ID && version == 1,
                "Unsupported or unrelated search catalog: application_id={id}, version={version}"
            );
            let profile: String = conn.query_row(
                "SELECT value FROM settings WHERE key='model_profile'",
                [],
                |r| r.get(0),
            )?;
            ensure!(profile == MODEL_PROFILE, "Catalog embedding profile {profile:?} differs from {MODEL_PROFILE:?}; use a separate catalog and reindex");
        }
        conn.execute_batch(
            "PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;",
        )?;
        let thumbnails = data_dir.join("thumbnails");
        let previews = data_dir.join("previews");
        fs::create_dir_all(&thumbnails)?;
        fs::create_dir_all(&previews)?;
        Ok(Self {
            conn,
            thumbnails,
            previews,
        })
    }
    pub fn status(&self) -> Result<CatalogStatus> {
        let image_count = self
            .conn
            .query_row("SELECT count(*) FROM images", [], |r| r.get(0))?;
        let sources = self.conn.prepare("SELECT s.id,s.directory,count(i.id) FROM sources s LEFT JOIN images i ON i.source_id=s.id GROUP BY s.id ORDER BY s.directory")?
            .query_map([], |r| Ok(Source { id: r.get(0)?, directory: r.get(1)?, image_count: r.get(2)? }))?.collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(CatalogStatus {
            image_count,
            sources,
            model_profile: MODEL_PROFILE,
        })
    }
    fn asset(row: &rusqlite::Row<'_>) -> rusqlite::Result<(Asset, PathBuf)> {
        Ok((
            Asset {
                asset_id: row.get(0)?,
                filename: row.get(1)?,
                path: row.get(2)?,
                thumbnail_url: String::new(),
                width: row.get(4)?,
                height: row.get(5)?,
                similarity: None,
            },
            PathBuf::from(row.get::<_, String>(3)?),
        ))
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
            let (bytes, modified): (u64, String) = self.conn.query_row(
                "SELECT bytes,modified FROM images WHERE asset_id=?1",
                [&asset.asset_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
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
        self.conn
            .query_row(
                "SELECT directory FROM sources WHERE id=?1",
                [source_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(PathBuf::from)
            .context("Indexed folder does not exist")
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
        let total = self.conn.query_row(
            "SELECT count(*) FROM images WHERE (?1 IS NULL OR source_id=?1)",
            [source_id],
            |r| r.get(0),
        )?;
        let rows = self.conn.prepare("SELECT asset_id,filename,path,thumbnail,width,height FROM images WHERE (?3 IS NULL OR source_id=?3) ORDER BY filename COLLATE NOCASE,asset_id LIMIT ?1 OFFSET ?2")?
            .query_map(params![limit as i64, offset as i64, source_id], Self::asset)?.collect::<rusqlite::Result<Vec<_>>>()?;
        let items = rows
            .into_iter()
            .map(|r| self.complete_asset(r))
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
        let vector = vector_bytes(query_embedding)?;
        // Apply the rowid constraint inside vec0's KNN query, before ranking;
        // filtering the global top-k afterward would omit valid folder hits.
        let sql = if source_id.is_some() {
            "SELECT i.asset_id,i.filename,i.path,i.thumbnail,i.width,i.height,v.distance FROM image_vectors v JOIN images i ON i.id=v.rowid WHERE v.embedding MATCH ?1 AND k=?2 AND v.rowid IN (SELECT id FROM images WHERE source_id=?3) ORDER BY v.distance"
        } else {
            "SELECT i.asset_id,i.filename,i.path,i.thumbnail,i.width,i.height,v.distance FROM image_vectors v JOIN images i ON i.id=v.rowid WHERE v.embedding MATCH ?1 AND k=?2 ORDER BY v.distance"
        };
        let mut args = vec![
            rusqlite::types::Value::Blob(vector),
            rusqlite::types::Value::Integer(limit as i64),
        ];
        if let Some(id) = source_id {
            args.push(rusqlite::types::Value::Integer(id));
        }
        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(args), |r| {
                Ok((Self::asset(r)?, r.get::<_, f32>(6)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut assets = Vec::with_capacity(rows.len());
        for (asset, distance) in rows {
            ensure!(
                distance.is_finite(),
                "Vector search returned a non-finite distance"
            );
            let mut asset = self.complete_asset(asset)?;
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
    /// Remove this source's index and generated caches, never its originals.
    /// Batches keep memory bounded and each committed batch keeps vectors and
    /// metadata consistent even if a later cache deletion fails visibly.
    pub fn remove_source(&mut self, source_id: i64) -> Result<usize> {
        self.source_directory(source_id)?;
        let mut removed = 0;
        loop {
            let rows = self
                .conn
                .prepare("SELECT id,thumbnail,asset_id FROM images WHERE source_id=?1 LIMIT 1000")?
                .query_map([source_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            if rows.is_empty() {
                break;
            }
            let tx = self.conn.transaction()?;
            for (id, _, _) in &rows {
                tx.execute("DELETE FROM image_vectors WHERE rowid=?1", [id])?;
                tx.execute("DELETE FROM images WHERE id=?1", [id])?;
            }
            tx.commit()?;
            removed += rows.len();
            for (_, thumbnail, asset_id) in rows {
                self.remove_cache(Path::new(&thumbnail))?;
                self.remove_cache(&self.previews.join(format!("{asset_id}.png")))?;
            }
        }
        self.conn
            .execute("DELETE FROM sources WHERE id=?1", [source_id])?;
        Ok(removed)
    }
    fn source(&mut self, directory: &Path) -> Result<i64> {
        for source in self.status()?.sources {
            let other = Path::new(&source.directory);
            if directory == other {
                return Ok(source.id);
            }
            ensure!(!directory.starts_with(other) && !other.starts_with(directory), "Folder overlaps an existing source: {}. Rescan the existing source or choose a separate folder", source.directory);
        }
        self.conn.execute(
            "INSERT INTO sources(directory) VALUES (?1)",
            [text_path(directory)?],
        )?;
        Ok(self.conn.last_insert_rowid())
    }
    pub fn index<E: Embedder>(
        &mut self,
        directory: &Path,
        embedder: &mut E,
        cancelled: impl Fn() -> bool,
        mut progress: impl FnMut(&IndexProgress) -> Result<()>,
    ) -> Result<IndexResult> {
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
        for path in &paths {
            if cancelled() {
                return Ok(state.result(0, true, failures));
            }
            let start = Instant::now();
            state.failure = None;
            state.filename = path
                .file_name()
                .context("Image filename missing")?
                .to_string_lossy()
                .into_owned();
            let path_text = text_path(path)?;
            let before = match stamp(path) {
                Ok(stamp) => stamp,
                Err(error) => {
                    state.record_failure(path, error, &mut failures);
                    progress(&state)?;
                    continue;
                }
            };
            let existing: Option<(i64, String, u64, String, String)> = self
                .conn
                .query_row(
                    "SELECT id,asset_id,bytes,modified,thumbnail FROM images WHERE path=?1",
                    [path_text],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                )
                .optional()?;
            let unchanged = existing
                .as_ref()
                .is_some_and(|row| row.2 == before.bytes && row.3 == before.modified);
            if unchanged
                && existing
                    .as_ref()
                    .is_some_and(|row| Path::new(&row.4).is_file())
            {
                state.unchanged += 1;
            } else {
                let image = match decode_image(path) {
                    Ok(image) => image,
                    Err(error) => {
                        state.record_failure(path, error, &mut failures);
                        progress(&state)?;
                        continue;
                    }
                };
                let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
                let embedding = if unchanged {
                    None
                } else {
                    Some(vector_bytes(
                        &embedder
                            .embed_image(&image)
                            .with_context(|| format!("Embed image {path:?}"))?,
                    )?)
                };
                let source_check = stamp(path).and_then(|after| {
                    ensure!(before == after, "Image changed during indexing: {path:?}; rescan after it finishes changing");
                    Ok(())
                });
                if let Err(error) = source_check {
                    state.record_failure(path, error, &mut failures);
                    progress(&state)?;
                    continue;
                }
                let asset_id = existing
                    .as_ref()
                    .map(|r| r.1.clone())
                    .unwrap_or_else(|| format!("asset_{}", Uuid::new_v4()));
                let digest = format!(
                    "{:x}",
                    Sha256::digest(format!("{}:{}", before.bytes, before.modified).as_bytes())
                );
                let cache = self
                    .thumbnails
                    .join(format!("{asset_id}-{}.webp", &digest[..16]));
                let thumb_start = Instant::now();
                write_atomic(&cache, &thumbnail(&image)?)?;
                let filename = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .context("Image has no UTF-8 filename")?;
                let tx = self.conn.transaction()?;
                tx.execute("INSERT INTO images(asset_id,source_id,path,filename,width,height,bytes,modified,thumbnail) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9) ON CONFLICT(path) DO UPDATE SET width=excluded.width,height=excluded.height,bytes=excluded.bytes,modified=excluded.modified,thumbnail=excluded.thumbnail",
                    params![asset_id,source_id,path_text,filename,image.width(),image.height(),before.bytes,before.modified,text_path(&cache)?])?;
                let row_id: i64 =
                    tx.query_row("SELECT id FROM images WHERE path=?1", [path_text], |r| {
                        r.get(0)
                    })?;
                if let Some(embedding) = embedding {
                    tx.execute("DELETE FROM image_vectors WHERE rowid=?1", [row_id])?;
                    tx.execute(
                        "INSERT INTO image_vectors(rowid,embedding) VALUES (?1,?2)",
                        params![row_id, embedding],
                    )?;
                }
                tx.commit()?;
                if let Some(old) = existing
                    .as_ref()
                    .filter(|r| r.4 != text_path(&cache).expect("validated cache path"))
                {
                    self.remove_cache(Path::new(&old.4))?;
                }
                let preview = self.previews.join(format!("{asset_id}.png"));
                self.remove_cache(&preview)?;
                state.indexed += 1;
                eprintln!("[Index] path={path:?} dimensions={}x{} decode_ms={decode_ms:.1} thumbnail_and_store_ms={:.1} total_ms={:.1} embedding_reused={unchanged}", image.width(), image.height(), thumb_start.elapsed().as_secs_f64()*1000.0, start.elapsed().as_secs_f64()*1000.0);
            }
            state.current += 1;
            progress(&state)?;
        }
        if cancelled() {
            return Ok(state.result(0, true, failures));
        }
        // Only a completed traversal may prune missing files. Known files that
        // failed decoding stay in `seen`, preserving their last good entries.
        let seen: HashSet<_> = paths
            .iter()
            .map(|p| text_path(p).map(str::to_owned))
            .collect::<Result<_>>()?;
        let previous = self
            .conn
            .prepare("SELECT id,path,thumbnail,asset_id FROM images WHERE source_id=?1")?
            .query_map([source_id], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut removed = 0;
        for (id, path, thumbnail, asset_id) in previous {
            if seen.contains(&path) {
                continue;
            }
            let tx = self.conn.transaction()?;
            tx.execute("DELETE FROM image_vectors WHERE rowid=?1", [id])?;
            tx.execute("DELETE FROM images WHERE id=?1", [id])?;
            tx.commit()?;
            self.remove_cache(Path::new(&thumbnail))?;
            self.remove_cache(&self.previews.join(format!("{asset_id}.png")))?;
            removed += 1;
        }
        state.stage = "complete";
        state.failure = None;
        progress(&state)?;
        Ok(state.result(removed, false, failures))
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
        let (path, bytes, modified, width, height): (String, u64, String, u32, u32) = self
            .conn
            .query_row(
                "SELECT path,bytes,modified,width,height FROM images WHERE asset_id=?1",
                [asset_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .optional()?
            .context("Image is not in the search catalog")?;
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
        let path: String = self
            .conn
            .query_row(
                "SELECT path FROM images WHERE asset_id=?1",
                [asset_id],
                |r| r.get(0),
            )
            .optional()?
            .context("Image is not in the search catalog")?;
        ensure!(
            Path::new(&path).is_file(),
            "Original image is missing: {path}"
        );
        file_url(Path::new(&path))
    }
    pub fn checkpoint(&self) -> Result<()> {
        let (busy, _, _): (i64, i64, i64) =
            self.conn
                .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })?;
        ensure!(busy == 0, "Search catalog checkpoint is busy");
        Ok(())
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
        assert_eq!(
            catalog
                .conn
                .query_row("SELECT count(*) FROM image_vectors", [], |row| row
                    .get::<_, usize>(0))
                .unwrap(),
            1
        );
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
    fn catalog_rejects_bad_profiles_overlaps_and_unbounded_queries() {
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
        catalog
            .conn
            .execute(
                "UPDATE settings SET value='different' WHERE key='model_profile'",
                [],
            )
            .unwrap();
        drop(catalog);
        assert!(Catalog::open(&data)
            .unwrap_err()
            .to_string()
            .contains("profile"));
    }

    #[test]
    fn corrupt_images_report_errors_continue_the_scan_and_retry_after_repair() {
        crate::initialize_image_decoders();
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("photos");
        fs::create_dir(&photos).unwrap();
        save(&photos.join("a-valid.png"), [255, 0, 0]);
        let jpeg = photos.join("b-corrupt.jpg");
        fs::write(
            &jpeg,
            include_bytes!("../tests/fixtures/jpeg-extraneous-byte.jpg"),
        )
        .unwrap();
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
