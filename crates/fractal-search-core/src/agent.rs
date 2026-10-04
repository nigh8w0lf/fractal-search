//! Read-only, bounded views of the Quickshell catalog for local agent clients.
//! Never delegates to Catalog: its display helpers may repair caches from originals.
use crate::{
    store::{Record, Store},
    MODEL_PROFILE,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{
    fs::{File, OpenOptions},
    io::{Cursor, Read},
    path::{Path, PathBuf},
};
use uuid::Uuid;

pub const MAX_THUMBNAIL_BYTES: usize = 1024 * 1024;
pub const MAX_QUERY_BYTES: usize = 4096;
const MAX_FILENAME_BYTES: usize = 1024;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetMetadata {
    pub asset_id: String,
    pub filename: String,
    pub width: u32,
    pub height: u32,
    pub thumbnail_handle: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f32>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetPage {
    pub items: Vec<AssetMetadata>,
    pub offset: usize,
    pub limit: usize,
    pub total: usize,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentStatus {
    pub image_count: usize,
    pub source_count: usize,
    pub model_profile: &'static str,
}

pub struct AgentCatalog {
    store: Store,
    thumbnails: PathBuf,
}

fn database<T>(result: Result<T>) -> Result<T> {
    result.map_err(|_| {
        anyhow::anyhow!("Cannot read search catalog; inspect or repair it in Fractal Search")
    })
}
pub fn validate_asset_id(id: &str) -> Result<()> {
    let uuid = id.strip_prefix("asset_").context("Invalid asset ID")?;
    let parsed = Uuid::parse_str(uuid).context("Invalid asset ID")?;
    ensure!(uuid == parsed.hyphenated().to_string(), "Invalid asset ID");
    Ok(())
}
pub fn validate_page(offset: usize, limit: usize) -> Result<()> {
    ensure!(
        (1..=100).contains(&limit),
        "limit must be between 1 and 100"
    );
    ensure!(offset <= 250_000, "offset must be at most 250000");
    Ok(())
}
pub fn validate_query(query: &str) -> Result<()> {
    ensure!(
        !query.trim().is_empty() && query.len() <= MAX_QUERY_BYTES,
        "query must contain 1..=4096 UTF-8 bytes"
    );
    ensure!(!query.contains('\0'), "query must not contain NUL");
    Ok(())
}
impl AgentCatalog {
    /// No creation, migration, desktop lock, writes, checkpoint, or model loading.
    pub fn open(data_dir: &Path) -> Result<Self> {
        let data_dir = data_dir.canonicalize().map_err(|_| {
            anyhow::anyhow!(
                "Search catalog directory is unavailable; index a folder in Fractal Search first"
            )
        })?;
        let store = database(Store::open(&data_dir, true))?;
        Ok(Self {
            store,
            thumbnails: data_dir.join("thumbnails"),
        })
    }
    pub fn status(&self) -> Result<AgentStatus> {
        let (image_count, sources) = database(self.store.status())?;
        Ok(AgentStatus {
            image_count,
            source_count: sources.len(),
            model_profile: MODEL_PROFILE,
        })
    }
    fn row(row: Record) -> AssetMetadata {
        AssetMetadata {
            thumbnail_handle: format!("thumbnail_{}", row.asset_id),
            asset_id: row.asset_id,
            filename: row.filename,
            width: row.width,
            height: row.height,
            similarity: None,
        }
    }
    fn validate(asset: AssetMetadata) -> Result<AssetMetadata> {
        validate_asset_id(&asset.asset_id)
            .context("Invalid indexed asset identity; repair in Fractal Search")?;
        ensure!(
            !asset.filename.is_empty()
                && asset.filename.len() <= MAX_FILENAME_BYTES
                && !asset
                    .filename
                    .chars()
                    .any(|c| c.is_control() || c == '/' || c == '\\'),
            "Invalid indexed filename; repair in Fractal Search"
        );
        ensure!(
            asset.width > 0
                && asset.height > 0
                && asset.width <= 65536
                && asset.height <= 65536
                && u64::from(asset.width) * u64::from(asset.height) <= 100_000_000,
            "Invalid indexed image dimensions"
        );
        Ok(asset)
    }
    pub fn browse(&self, offset: usize, limit: usize) -> Result<AssetPage> {
        validate_page(offset, limit)?;
        let (total, rows) = database(self.store.browse(offset, limit, None))?;
        let items = rows
            .into_iter()
            .map(|row| Self::validate(Self::row(row)))
            .collect::<Result<Vec<_>>>()?;
        Ok(AssetPage {
            items,
            total,
            offset,
            limit,
        })
    }
    pub fn metadata(&self, asset_id: &str) -> Result<AssetMetadata> {
        validate_asset_id(asset_id)?;
        Self::validate(Self::row(database(self.store.asset(asset_id, false))?))
    }
    pub fn search(&self, embedding: &[f32], limit: usize) -> Result<Vec<AssetMetadata>> {
        validate_page(0, limit)?;
        let hits = database(self.store.search(embedding, limit, None))?;
        let mut items = Vec::with_capacity(limit);
        for (row, distance) in hits {
            let mut asset = Self::validate(Self::row(row))?;
            asset.similarity = Some((1.0 - distance).clamp(-1.0, 1.0));
            items.push(asset);
        }
        items.sort_by(|a, b| {
            b.similarity
                .unwrap()
                .total_cmp(&a.similarity.unwrap())
                .then(a.asset_id.cmp(&b.asset_id))
        });
        Ok(items)
    }
    pub fn thumbnail(&self, handle: &str) -> Result<Vec<u8>> {
        let id = handle
            .strip_prefix("thumbnail_")
            .context("Invalid thumbnail handle")?;
        validate_asset_id(id)?;
        let record = database(self.store.asset(id, false))?;
        let cache = record.thumbnail.clone();
        let asset = Self::validate(Self::row(record))?;
        let cache = Path::new(&cache);
        ensure!(
            cache.parent() == Some(self.thumbnails.as_path()),
            "Thumbnail is outside the managed cache; rescan in Fractal Search"
        );
        let name = cache
            .file_name()
            .and_then(|s| s.to_str())
            .context("Invalid thumbnail cache entry")?;
        let digest = name
            .strip_prefix(&format!("{id}-"))
            .and_then(|s| s.strip_suffix(".webp"))
            .context("Invalid thumbnail cache entry")?;
        ensure!(
            digest.len() == 16 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "Invalid thumbnail cache entry"
        );
        let mut file = open_cache_file(&self.thumbnails, name)?;
        let metadata = file
            .metadata()
            .map_err(|_| anyhow::anyhow!("Cannot inspect thumbnail cache entry"))?;
        ensure!(
            metadata.is_file()
                && metadata.len() > 0
                && metadata.len() <= MAX_THUMBNAIL_BYTES as u64,
            "Thumbnail cache entry is not a bounded regular file"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            ensure!(
                metadata.nlink() == 1 && metadata.uid() == unsafe { libc::geteuid() },
                "Thumbnail must be a same-user cache file without hard links"
            );
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.by_ref()
            .take((MAX_THUMBNAIL_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|_| anyhow::anyhow!("Cannot read thumbnail cache entry"))?;
        ensure!(
            bytes.len() <= MAX_THUMBNAIL_BYTES,
            "Thumbnail exceeds byte limit"
        );
        let dimensions =
            image::ImageReader::with_format(Cursor::new(&bytes), image::ImageFormat::WebP)
                .into_dimensions()
                .map_err(|_| {
                    anyhow::anyhow!("Invalid cached WebP thumbnail; rescan in Fractal Search")
                })?;
        let longest = asset.width.max(asset.height);
        let expected = if longest <= 384 {
            (asset.width, asset.height)
        } else {
            (
                ((u64::from(asset.width) * 384) / u64::from(longest)).max(1) as u32,
                ((u64::from(asset.height) * 384) / u64::from(longest)).max(1) as u32,
            )
        };
        ensure!(
            dimensions == expected,
            "Cached thumbnail dimensions are stale; rescan in Fractal Search"
        );
        let mut reader =
            image::ImageReader::with_format(Cursor::new(&bytes), image::ImageFormat::WebP);
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(384);
        limits.max_image_height = Some(384);
        limits.max_alloc = Some(8 * 1024 * 1024);
        reader.limits(limits);
        let pixels = reader
            .decode()
            .map_err(|_| {
                anyhow::anyhow!("Cannot decode bounded cached thumbnail; rescan in Fractal Search")
            })?
            .into_rgb8();
        // Return newly encoded pixels, never EXIF/XMP or arbitrary RIFF chunks
        // that may have been attached to an otherwise valid cache entry.
        let mut sanitized = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut sanitized)
            .encode(
                pixels.as_raw(),
                pixels.width(),
                pixels.height(),
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|_| anyhow::anyhow!("Cannot encode sanitized thumbnail"))?;
        ensure!(
            sanitized.len() <= MAX_THUMBNAIL_BYTES,
            "Sanitized thumbnail exceeds byte limit"
        );
        Ok(sanitized)
    }
}
#[cfg(unix)]
fn open_cache_file(directory: &Path, name: &str) -> Result<File> {
    use std::os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::OpenOptionsExt,
    };
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(directory)
        .map_err(|_| anyhow::anyhow!("Managed thumbnail directory is unavailable"))?;
    let name = std::ffi::CString::new(name).context("Invalid thumbnail cache entry")?;
    // openat pins the directory and O_NOFOLLOW blocks a file swapped to a link.
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
        )
    };
    ensure!(
        fd >= 0,
        "Thumbnail cache file is unavailable; rescan in Fractal Search"
    );
    Ok(unsafe { File::from_raw_fd(fd) })
}
#[cfg(not(unix))]
fn open_cache_file(_: &Path, _: &str) -> Result<File> {
    anyhow::bail!("Secure managed-cache thumbnails require Unix on this Quickshell build")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{catalog::Catalog, Embedder, EMBEDDING_DIM};
    use std::fs;
    struct Fake;
    impl Embedder for Fake {
        fn embed_image(&mut self, _: &image::RgbImage) -> Result<Vec<f32>> {
            let mut v = vec![0.; EMBEDDING_DIM];
            v[0] = 1.;
            Ok(v)
        }
        fn embed_text(&mut self, _: &str) -> Result<Vec<f32>> {
            anyhow::bail!("metadata must not load inference")
        }
    }
    fn fixture() -> (tempfile::TempDir, Catalog, AgentCatalog) {
        let root = tempfile::tempdir().unwrap();
        let photos = root.path().join("private-originals");
        fs::create_dir(&photos).unwrap();
        image::RgbImage::from_pixel(48, 32, image::Rgb([255, 0, 0]))
            .save(photos.join("red.png"))
            .unwrap();
        let mut desktop = Catalog::open(&root.path().join("data")).unwrap();
        desktop
            .index(&photos, &mut Fake, || false, |_| Ok(()))
            .unwrap();
        let agent = AgentCatalog::open(&root.path().join("data")).unwrap();
        (root, desktop, agent)
    }
    #[test]
    fn agent_coexists_read_only_and_exposes_only_bounded_metadata() {
        let (root, _desktop, agent) = fixture();
        let page = agent.browse(0, 10).unwrap();
        assert_eq!(page.total, 1);
        let json = serde_json::to_string(&page).unwrap();
        assert!(!json.contains(&root.path().display().to_string()));
        let asset = &page.items[0];
        assert_eq!(agent.metadata(&asset.asset_id).unwrap().filename, "red.png");
        assert!(!agent.thumbnail(&asset.thumbnail_handle).unwrap().is_empty());
        let mut embedding = vec![0.; EMBEDDING_DIM];
        embedding[0] = 1.;
        assert_eq!(
            agent.search(&embedding, 10).unwrap()[0].asset_id,
            asset.asset_id
        );
        assert!(agent.store.delete("true").is_err());
        assert!(agent.browse(0, 101).is_err());
        assert!(agent.browse(250001, 1).is_err());
        assert!(agent.metadata("../../private-originals/red.png").is_err());
        assert!(agent
            .thumbnail("file:///private-originals/red.png")
            .is_err());
    }
    #[test]
    fn missing_or_wrong_catalog_is_never_created_or_migrated() {
        let root = tempfile::tempdir().unwrap();
        assert!(AgentCatalog::open(root.path()).is_err());
        assert!(!root.path().join("catalog.lancedb").exists());
        fs::create_dir(root.path().join("catalog.lancedb")).unwrap();
        fs::write(root.path().join("catalog.lancedb/unrelated"), b"preserve").unwrap();
        assert!(AgentCatalog::open(root.path()).is_err());
        assert_eq!(
            fs::read(root.path().join("catalog.lancedb/unrelated")).unwrap(),
            b"preserve"
        );
    }
    #[test]
    fn malformed_catalog_metadata_fails_without_path_disclosure() {
        let (root, _desktop, agent) = fixture();
        let writer = Store::open(&root.path().join("data"), false).unwrap();
        let id = agent.browse(0, 1).unwrap().items[0].asset_id.clone();
        writer.update_for_test(&id, "filename", "/private/secrets/photo.png");
        let error = agent.browse(0, 10).unwrap_err().to_string();
        assert!(!error.contains("/private"));
        writer.update_for_test(&id, "filename", "photo.png");
        writer.update_for_test(&id, "asset_id", "legacy/path");
        assert!(agent.browse(0, 10).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn thumbnails_reject_symlinks_hardlinks_and_outside_cache() {
        use std::os::unix::fs::symlink;
        let (root, _desktop, agent) = fixture();
        let asset = agent.browse(0, 1).unwrap().items.remove(0);
        let cache = agent.store.asset(&asset.asset_id, false).unwrap().thumbnail;
        let outside = root.path().join("private-originals/red.png");
        fs::remove_file(&cache).unwrap();
        symlink(&outside, &cache).unwrap();
        assert!(agent.thumbnail(&asset.thumbnail_handle).is_err());
        fs::remove_file(&cache).unwrap();
        fs::hard_link(&outside, &cache).unwrap();
        assert!(agent.thumbnail(&asset.thumbnail_handle).is_err());
        let writer = Store::open(&root.path().join("data"), false).unwrap();
        writer.update_for_test(&asset.asset_id, "thumbnail", outside.to_str().unwrap());
        assert!(agent.thumbnail(&asset.thumbnail_handle).is_err());
    }
    #[test]
    fn missing_or_legacy_thumbnail_never_reads_original_or_repairs() {
        let (_root, _desktop, agent) = fixture();
        let asset = agent.browse(0, 1).unwrap().items.remove(0);
        let cache = agent.store.asset(&asset.asset_id, false).unwrap().thumbnail;
        image::RgbImage::from_pixel(384, 384, image::Rgb([0, 0, 0]))
            .save(&cache)
            .unwrap();
        let before = fs::read(&cache).unwrap();
        assert!(agent.thumbnail(&asset.thumbnail_handle).is_err());
        assert_eq!(before, fs::read(&cache).unwrap());
        fs::remove_file(&cache).unwrap();
        assert!(agent.thumbnail(&asset.thumbnail_handle).is_err());
        assert!(!Path::new(&cache).exists());
    }

    #[test]
    fn returned_thumbnail_strips_nonpixel_riff_payloads() {
        let (_root, _desktop, agent) = fixture();
        let asset = agent.browse(0, 1).unwrap().items.remove(0);
        let cache = agent.store.asset(&asset.asset_id, false).unwrap().thumbnail;
        let mut bytes = fs::read(&cache).unwrap();
        let secret = b"private-metadata";
        bytes.extend_from_slice(b"JUNK");
        bytes.extend_from_slice(&(secret.len() as u32).to_le_bytes());
        bytes.extend_from_slice(secret);
        let riff_size = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&riff_size.to_le_bytes());
        fs::write(&cache, &bytes).unwrap();
        let returned = agent.thumbnail(&asset.thumbnail_handle).unwrap();
        assert!(!returned
            .windows(secret.len())
            .any(|window| window == secret));
        assert_eq!(fs::read(&cache).unwrap(), bytes);
    }

    #[test]
    fn cached_thumbnail_byte_limit_and_corruption_fail_visibly() {
        let (_root, _desktop, agent) = fixture();
        let asset = agent.browse(0, 1).unwrap().items.remove(0);
        let cache = agent.store.asset(&asset.asset_id, false).unwrap().thumbnail;
        fs::write(&cache, vec![0; MAX_THUMBNAIL_BYTES + 1]).unwrap();
        assert!(agent.thumbnail(&asset.thumbnail_handle).is_err());
        fs::write(&cache, b"not a WebP").unwrap();
        assert!(agent.thumbnail(&asset.thumbnail_handle).is_err());
    }
}
