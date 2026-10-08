//! Optional derived state. The Lance file generation remains authoritative.
//! A crash before this sidecar publishes loses reuse, never searchable rows.
use crate::{catalog::validate_embedding, store::Record, EMBEDDING_PROJECTION, MODEL_PROFILE};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

pub(crate) const EMBEDDING_CACHE_VERSION: &str =
    "formatted-input-assets-tokenizer-preprocess-fp32-mrl512-v2";
const MAX_STATE_BYTES: u64 = 1024 * 1024;
const MAX_VECTOR_BYTES: u64 = 32 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FileStamp {
    pub bytes: u64,
    pub modified: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PassageLocation {
    pub asset_id: String,
    #[serde(default)]
    pub row_generation: String,
    pub start_byte: Option<u64>,
    pub end_byte: Option<u64>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FileState {
    version: u32,
    pub generation: String,
    pub stamp: FileStamp,
    pub content_sha256: String,
    pub extraction_profile: String,
    pub embedding_profile: Option<String>,
    pub locations: Vec<PassageLocation>,
}

impl FileState {
    pub fn new(
        rows: &[Record],
        stamp: FileStamp,
        hash: String,
        extraction: &str,
        embedding: Option<String>,
        mut locations: Vec<PassageLocation>,
    ) -> Self {
        for location in &mut locations {
            if let Some(row) = rows.iter().find(|r| r.asset_id == location.asset_id) {
                location.row_generation = generation(std::slice::from_ref(row));
            }
        }
        Self {
            version: 1,
            generation: generation(rows),
            stamp,
            content_sha256: hash,
            extraction_profile: extraction.into(),
            embedding_profile: embedding,
            locations,
        }
    }
    pub fn matches_profile(&self, embedding: Option<&str>, extraction: Option<&str>) -> bool {
        self.embedding_profile.as_deref() == embedding
            && extraction.is_none_or(|profile| self.extraction_profile == profile)
    }
    pub fn matches(&self, rows: &[Record]) -> bool {
        self.generation == generation(rows)
    }
}

#[derive(Debug)]
pub(crate) struct IndexState {
    root: PathBuf,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn generation(rows: &[Record]) -> String {
    let mut keys: Vec<_> = rows
        .iter()
        .map(|r| {
            (
                r.asset_id.as_str(),
                r.source_id,
                r.path.as_str(),
                r.bytes,
                r.modified.as_str(),
                r.media_kind.as_str(),
                r.segment_index,
            )
        })
        .collect();
    keys.sort();
    digest(&serde_json::to_vec(&keys).expect("row identity serialization"))
}

pub(crate) fn hash_file(path: &Path, expected: &FileStamp) -> Result<String> {
    hash_file_with_cancel(path, expected, &|| false)
}
pub(crate) fn hash_file_with_cancel(
    path: &Path,
    expected: &FileStamp,
    cancelled: &impl Fn() -> bool,
) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("Hash source {path:?}"))?;
    ensure!(
        file.metadata()?.len() == expected.bytes,
        "File size changed before hashing: {path:?}"
    );
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        ensure!(!cancelled(), "Indexing cancelled while hashing source");
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        ensure!(total <= expected.bytes, "File grew while hashing: {path:?}");
        h.update(&buf[..n]);
    }
    ensure!(
        total == expected.bytes,
        "File changed while hashing: {path:?}"
    );
    Ok(format!("{:x}", h.finalize()))
}

/// Recent/future timestamps are never trusted by the cheap stat fast path.
/// Restoring an old timestamp deliberately requires an explicit verification scan.
pub(crate) fn is_racy(stamp: &FileStamp) -> bool {
    let Some((seconds, nanos)) = stamp.modified.split_once('.') else {
        return true;
    };
    let (Ok(seconds), Ok(nanos)) = (seconds.parse::<u64>(), nanos.parse::<u32>()) else {
        return true;
    };
    if nanos >= 1_000_000_000 {
        return true;
    }
    let modified = u128::from(seconds) * 1_000_000_000 + u128::from(nanos);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|x| x.as_nanos())
        .unwrap_or_default();
    now.saturating_sub(modified) < 2_000_000_000 || modified > now
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, max: u64) -> Result<Option<T>> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    ensure!(
        file.metadata()?.len() <= max,
        "Derived state exceeds size limit: {path:?}"
    );
    let mut data = Vec::new();
    file.take(max + 1).read_to_end(&mut data)?;
    ensure!(
        data.len() as u64 <= max,
        "Derived state grew beyond size limit: {path:?}"
    );
    Ok(Some(serde_json::from_slice(&data).with_context(|| {
        format!("Corrupt derived index state {path:?}; move this file aside to rebuild it")
    })?))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("Derived state parent missing")?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".{}.part", Uuid::new_v4()));
    let result = (|| {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        serde_json::to_writer(&mut f, value)?;
        f.flush()?;
        f.sync_all()?;
        fs::rename(&tmp, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

impl IndexState {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            root: data_dir.join("index-state-v1"),
        }
    }
    fn file_path(&self, source: i64, path: &str) -> PathBuf {
        self.root.join("files").join(format!(
            "{}.json",
            digest(format!("{source}\0{path}").as_bytes())
        ))
    }
    pub fn load(&self, source: i64, path: &str, rows: &[Record]) -> Result<Option<FileState>> {
        let state: Option<FileState> = read_json(&self.file_path(source, path), MAX_STATE_BYTES)?;
        if let Some(s) = &state {
            ensure!(s.version == 1, "Unsupported derived index state version");
        }
        Ok(state.filter(|s| s.matches(rows)))
    }
    pub fn save(&self, source: i64, path: &str, state: &FileState) -> Result<()> {
        write_json(&self.file_path(source, path), state)
    }
    pub fn location(&self, row: &Record) -> Result<Option<PassageLocation>> {
        if row.media_kind != "text" {
            return Ok(None);
        }
        let state: Option<FileState> =
            read_json(&self.file_path(row.source_id, &row.path), MAX_STATE_BYTES)?;
        if let Some(state) = &state {
            ensure!(
                state.version == 1,
                "Unsupported derived index state version"
            );
        }
        Ok(state.and_then(|s| {
            s.locations.into_iter().find(|l| {
                l.asset_id == row.asset_id
                    && l.row_generation == generation(std::slice::from_ref(row))
            })
        }))
    }
    pub fn document_key(&self, runtime: &str, extraction: &str, title: &str, text: &str) -> String {
        let title = if title.trim().is_empty() {
            "none"
        } else {
            title
        };
        let formatted = format!("title: {title} | text: {text}");
        // Manifest pins every graph, tokenizer and config SHA256, not just a model name.
        let contract = [
            EMBEDDING_CACHE_VERSION,
            MODEL_PROFILE,
            runtime,
            include_str!("../models/embeddinggemma2.json"),
            "text",
            "fp32",
            EMBEDDING_PROJECTION,
            "tokenizers-0.22.2-bos-eos-literal-specials",
            extraction,
            &formatted,
        ];
        digest(&serde_json::to_vec(&contract).expect("cache key serialization"))
    }
    pub fn vector(&self, key: &str) -> Result<Option<Vec<f32>>> {
        #[derive(Deserialize)]
        struct Entry {
            key: String,
            vector: Vec<f32>,
        }
        let entry: Option<Entry> = read_json(
            &self.root.join("vectors").join(format!("{key}.json")),
            MAX_VECTOR_BYTES,
        )?;
        entry
            .map(|x| {
                ensure!(x.key == key, "Embedding cache key mismatch");
                validate_embedding(&x.vector)?;
                Ok(x.vector)
            })
            .transpose()
    }
    pub fn save_vector(&self, key: &str, vector: &[f32]) -> Result<()> {
        validate_embedding(vector)?;
        write_json(
            &self.root.join("vectors").join(format!("{key}.json")),
            &serde_json::json!({"key":key,"vector":vector}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_input_keys_separate_title_runtime_and_extraction() {
        let dir = tempfile::tempdir().unwrap();
        let s = IndexState::new(dir.path());
        let base = s.document_key("model", "chunks", "one", "body");
        for key in [
            s.document_key("model", "chunks", "two", "body"),
            s.document_key("changed", "chunks", "one", "body"),
            s.document_key("model", "new-chunks", "one", "body"),
            s.document_key("model", "chunks", "one", "other"),
        ] {
            assert_ne!(key, base);
        }
        assert_eq!(base, s.document_key("model", "chunks", "one", "body"));
    }
    #[test]
    fn vector_round_trip_and_corruption_is_explicit() {
        let d = tempfile::tempdir().unwrap();
        let s = IndexState::new(d.path());
        let mut v = vec![0.; crate::EMBEDDING_DIM];
        v[0] = 1.;
        assert!(s.vector("key").unwrap().is_none());
        s.save_vector("key", &v).unwrap();
        assert_eq!(s.vector("key").unwrap().unwrap(), v);
        fs::write(s.root.join("vectors/key.json"), b"invalid").unwrap();
        assert!(s.vector("key").is_err());
    }
    #[test]
    fn projected_cache_is_separate_and_rejects_full_width_vectors_without_overwrite() {
        let d = tempfile::tempdir().unwrap();
        let state = IndexState::new(d.path());
        let old_contract = [
            "formatted-input-assets-tokenizer-preprocess-fp32-v1",
            crate::NATIVE_MODEL_PROFILE,
            crate::NATIVE_MODEL_PROFILE,
            include_str!("../models/embeddinggemma2.json"),
            "text",
            "fp32",
            "768-l2-masked-pool",
            "tokenizers-0.22.2-bos-eos-literal-specials",
            "chunks",
            "title: title | text: body",
        ];
        let old_key = digest(&serde_json::to_vec(&old_contract).unwrap());
        let key = state.document_key(MODEL_PROFILE, "chunks", "title", "body");
        assert_ne!(old_key, key);
        let mut old = vec![0.; crate::NATIVE_EMBEDDING_DIM];
        old[0] = 1.;
        assert!(state.save_vector(&key, &old).is_err());
        let path = state.root.join("vectors").join(format!("{old_key}.json"));
        write_json(&path, &serde_json::json!({"key":old_key,"vector":old})).unwrap();
        let before = fs::read(&path).unwrap();
        assert!(state.vector(&key).unwrap().is_none());
        assert!(state.vector(&old_key).is_err());
        assert_eq!(fs::read(path).unwrap(), before);
    }

    #[test]
    fn all_modality_file_states_bind_reuse_to_the_indexed_profile() {
        let d = tempfile::tempdir().unwrap();
        let index = IndexState::new(d.path());
        for kind in ["text", "image", "audio", "video"] {
            let row = Record {
                asset_id: kind.into(),
                source_id: 1,
                path: format!("/{kind}"),
                media_kind: kind.into(),
                ..Record::default()
            };
            let rows = [row];
            let state = FileState::new(
                &rows,
                FileStamp {
                    bytes: 1,
                    modified: "1.0".into(),
                },
                "hash".into(),
                "extraction",
                Some(MODEL_PROFILE.into()),
                vec![],
            );
            index.save(1, &rows[0].path, &state).unwrap();
            let actual = index.load(1, &rows[0].path, &rows).unwrap().unwrap();
            assert!(actual.matches_profile(Some(MODEL_PROFILE), Some("extraction")));
            assert!(!actual.matches_profile(Some(crate::NATIVE_MODEL_PROFILE), Some("extraction")));
            assert!(!actual.matches_profile(None, None));
            assert!(!actual.matches_profile(Some(MODEL_PROFILE), Some("other-extraction")));
        }
    }

    #[test]
    fn recent_and_future_stamps_are_racy_old_ones_are_not() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        for secs in [now, now + 99] {
            assert!(is_racy(&FileStamp {
                bytes: 1,
                modified: format!("{secs}.0")
            }));
        }
        assert!(!is_racy(&FileStamp {
            bytes: 1,
            modified: "1.0".into()
        }));
    }
}
