//! One Lance table owns source rows, segment metadata and multimodal vectors.
//! Each mutation is one atomic Lance commit; readers use a pinned snapshot.
use crate::{
    catalog::validate_embedding, media::MediaKind, TextSearchMode, EMBEDDING_DIM, MODEL_PROFILE,
    TEXT_SEARCH_PROFILE,
};
use anyhow::{ensure, Context, Result};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::{
    index::{
        scalar::{FtsIndexBuilder, FullTextSearchQuery},
        Index, IndexType,
    },
    query::{ColumnOrdering, ExecutableQuery, QueryBase, Select},
    Connection, DistanceType, Table,
};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_ROWS: usize = 250_000;
const TEXT_INDEX_NAME: &str = "text_passage_bm25_v1";
const MAX_INDEXED_TEXT_BYTES: usize = 1024 + 1 + 4096;
const RRF_K: f64 = 60.0;
const MAX_ADDITIONAL_MATCHES: usize = 5;
const MANIFEST_PAGE_SIZE: usize = 1024;
const TEXT_MAINTENANCE_STATE: &str = "fractal-text-index-state.json";
const SMALL_TEXT_CORPUS: usize = 4096;
const MIN_CHANGED_TEXT_ROWS: u64 = 256;
const MAX_INDEX_AGE_SECONDS: u64 = 24 * 60 * 60;

fn text_index_config() -> FtsIndexBuilder {
    FtsIndexBuilder::default()
        .lance_tokenizer("text".into())
        .base_tokenizer("icu/split".into())
        .lower_case(true)
        .ascii_folding(true)
        .stem(false)
        .remove_stop_words(false)
        .max_token_length(Some(4096))
        .with_position(false)
        .memory_limit_mb(128)
        .num_workers(2)
}

#[derive(Clone, Debug)]
pub(crate) struct SearchHit {
    pub row: Record,
    pub distance: f32,
    pub hybrid_score: Option<f64>,
    pub additional_matches: Vec<SearchHit>,
}

const COLUMNS: &[&str] = &[
    "kind",
    "asset_id",
    "source_id",
    "path",
    "filename",
    "width",
    "height",
    "bytes",
    "modified",
    "thumbnail",
    "media_kind",
    "segment_index",
    "start_ms",
    "end_ms",
    "text_excerpt",
];
#[derive(Clone, Debug, Default)]
pub(crate) struct Record {
    pub kind: String,
    pub asset_id: String,
    pub source_id: i64,
    pub path: String,
    pub filename: String,
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
    pub modified: String,
    pub thumbnail: Option<String>,
    pub media_kind: String,
    pub segment_index: u32,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub text_excerpt: Option<String>,
    // Private write-only projection: never part of ordinary metadata reads.
    pub indexed_text: Option<String>,
    pub embedding: Option<Vec<f32>>,
}
pub(crate) struct Store {
    path: PathBuf,
    read_only: bool,
    runtime: tokio::runtime::Runtime,
    // A connection owns reusable Lance caches; each request opens and pins an
    // independent table handle so checkout cannot move another reader's view.
    connection: OnceLock<Connection>,
    text_maintenance: Mutex<TextMaintenance>,
}
impl std::fmt::Debug for Store {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Store")
            .field("path", &self.path)
            .field("read_only", &self.read_only)
            .finish_non_exhaustive()
    }
}
/// Write-ahead generation intent. Persisted before a text mutation, and marked
/// indexed only after the atomic index-manifest commit. A failed row commit can
/// cause redundant maintenance, never falsely mark committed text as indexed.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TextMaintenance {
    format: u32,
    index_uuid: String,
    data_generation: u64,
    indexed_generation: u64,
    changed_rows: u64,
    last_indexed_at: u64,
    #[serde(skip)]
    recovery_due: bool,
}
impl TextMaintenance {
    fn dirty(&self) -> bool {
        self.data_generation != self.indexed_generation
    }
    fn due(&self, text_rows: usize, now: u64) -> bool {
        self.dirty()
            && (self.recovery_due
                || text_rows <= SMALL_TEXT_CORPUS
                || self.changed_rows >= MIN_CHANGED_TEXT_ROWS.max((text_rows as u64).div_ceil(10))
                || now.saturating_sub(self.last_indexed_at) >= MAX_INDEX_AGE_SECONDS)
    }
}
fn unix_seconds() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("System clock is before the Unix epoch")?
        .as_secs())
}
pub(crate) fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}
fn schema() -> Arc<Schema> {
    let mut fields = vec![];
    for name in COLUMNS {
        let data_type = match *name {
            "source_id" => DataType::Int64,
            "width" | "height" | "segment_index" => DataType::UInt32,
            "bytes" | "start_ms" | "end_ms" => DataType::UInt64,
            _ => DataType::Utf8,
        };
        fields.push(Field::new(
            *name,
            data_type,
            matches!(*name, "thumbnail" | "start_ms" | "end_ms" | "text_excerpt"),
        ));
    }
    fields.push(Field::new("indexed_text", DataType::Utf8, true));
    fields.push(Field::new("sort_filename", DataType::Utf8, false));
    fields.push(Field::new(
        "embedding",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            EMBEDDING_DIM as i32,
        ),
        true,
    ));
    Arc::new(Schema::new_with_metadata(
        fields,
        HashMap::from([
            ("fractal.schema".into(), "4".into()),
            (
                "fractal.text_search_profile".into(),
                TEXT_SEARCH_PROFILE.into(),
            ),
            ("fractal.model_profile".into(), MODEL_PROFILE.into()),
            ("fractal.embedding_dim".into(), EMBEDDING_DIM.to_string()),
            ("fractal.distance_metric".into(), "cosine".into()),
            ("fractal.normalization".into(), "l2".into()),
            ("fractal.vectors_per_segment".into(), "1".into()),
        ]),
    ))
}
fn batch(row: &Record) -> Result<RecordBatch> {
    if let Some(vector) = &row.embedding {
        validate_embedding(vector)?;
    }
    ensure!(
        (row.kind == "source" && row.embedding.is_none())
            || (row.kind == "asset" && row.embedding.is_some()),
        "Invalid catalog record kind or missing asset embedding"
    );
    ensure!(
        row.source_id > 0 && row.source_id <= i64::from(i32::MAX),
        "Invalid catalog source ID"
    );
    validate_record_metadata(row)?;
    validate_indexed_text(row)?;
    let string = |value: &str| -> ArrayRef { Arc::new(StringArray::from(vec![value])) };
    let embedding = FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
        [row.embedding.as_ref().map(|v| v.iter().copied().map(Some))],
        EMBEDDING_DIM as i32,
    );
    Ok(RecordBatch::try_new(
        schema(),
        vec![
            string(&row.kind),
            string(&row.asset_id),
            Arc::new(Int64Array::from(vec![row.source_id])),
            string(&row.path),
            string(&row.filename),
            Arc::new(UInt32Array::from(vec![row.width])),
            Arc::new(UInt32Array::from(vec![row.height])),
            Arc::new(UInt64Array::from(vec![row.bytes])),
            string(&row.modified),
            Arc::new(StringArray::from(vec![row.thumbnail.as_deref()])),
            string(&row.media_kind),
            Arc::new(UInt32Array::from(vec![row.segment_index])),
            Arc::new(UInt64Array::from(vec![row.start_ms])),
            Arc::new(UInt64Array::from(vec![row.end_ms])),
            Arc::new(StringArray::from(vec![row.text_excerpt.as_deref()])),
            Arc::new(StringArray::from(vec![row.indexed_text.as_deref()])),
            string(&row.filename.to_ascii_lowercase()),
            Arc::new(embedding),
        ],
    )?)
}
fn array<'a, T: Array + 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .with_context(|| format!("Missing catalog column {name}"))?
        .as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("Invalid catalog column {name}"))
}
fn record(batch: &RecordBatch, i: usize) -> Result<Record> {
    for name in COLUMNS {
        if matches!(*name, "thumbnail" | "start_ms" | "end_ms" | "text_excerpt") {
            continue;
        }
        ensure!(
            !batch
                .column_by_name(name)
                .context("Missing catalog column")?
                .is_null(i),
            "Null catalog metadata"
        );
    }
    let text =
        |name| -> Result<String> { Ok(array::<StringArray>(batch, name)?.value(i).to_owned()) };
    let optional_text = |name| -> Result<Option<String>> {
        let values = array::<StringArray>(batch, name)?;
        Ok((!values.is_null(i)).then(|| values.value(i).to_owned()))
    };
    let optional_time = |name| -> Result<Option<u64>> {
        let values = array::<UInt64Array>(batch, name)?;
        Ok((!values.is_null(i)).then(|| values.value(i)))
    };
    let embedding = if batch.column_by_name("embedding").is_some() {
        let values = array::<FixedSizeListArray>(batch, "embedding")?;
        if values.is_null(i) {
            None
        } else {
            let values = values.value(i);
            let values = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Invalid catalog embedding")?;
            ensure!(values.null_count() == 0, "Null embedding values");
            let vector = values.values().to_vec();
            validate_embedding(&vector)?;
            Some(vector)
        }
    } else {
        None
    };
    let kind = text("kind")?;
    ensure!(
        kind == "source" || kind == "asset",
        "Invalid catalog record kind"
    );
    if kind == "asset" && batch.column_by_name("embedding").is_some() {
        ensure!(
            embedding.is_some(),
            "Indexed asset is missing its embedding"
        );
    }
    let row = Record {
        kind,
        asset_id: text("asset_id")?,
        source_id: array::<Int64Array>(batch, "source_id")?.value(i),
        path: text("path")?,
        filename: text("filename")?,
        width: array::<UInt32Array>(batch, "width")?.value(i),
        height: array::<UInt32Array>(batch, "height")?.value(i),
        bytes: array::<UInt64Array>(batch, "bytes")?.value(i),
        modified: text("modified")?,
        thumbnail: optional_text("thumbnail")?,
        media_kind: text("media_kind")?,
        segment_index: array::<UInt32Array>(batch, "segment_index")?.value(i),
        start_ms: optional_time("start_ms")?,
        end_ms: optional_time("end_ms")?,
        text_excerpt: optional_text("text_excerpt")?,
        indexed_text: if batch.column_by_name("indexed_text").is_some() {
            optional_text("indexed_text")?
        } else {
            None
        },
        embedding,
    };
    validate_record_metadata(&row)?;
    if batch.column_by_name("indexed_text").is_some() {
        validate_indexed_text(&row)?;
    }
    Ok(row)
}
pub(crate) const MAX_TEXT_EXCERPT_BYTES: usize = 2048;
/// Require the complete filename/passage write contract separately from metadata
/// validation, because read-only metadata queries deliberately omit this column.
fn validate_indexed_text(row: &Record) -> Result<()> {
    if row.kind != "asset" || row.media_kind != "text" {
        ensure!(
            row.indexed_text.is_none(),
            "Only text assets may contain indexed text"
        );
        return Ok(());
    }
    let indexed = row
        .indexed_text
        .as_deref()
        .context("Text asset is missing its complete indexed passage")?;
    ensure!(
        !row.filename.is_empty()
            && row.filename.len() <= 1024
            && !row
                .filename
                .chars()
                .any(|c| c.is_control() || c == '/' || c == '\\'),
        "Invalid indexed text filename"
    );
    let prefix = format!("{}\n", row.filename);
    let passage = indexed
        .strip_prefix(&prefix)
        .context("Indexed text must contain filename and full passage")?;
    ensure!(
        indexed.len() <= MAX_INDEXED_TEXT_BYTES
            && !indexed.contains('\0')
            && !passage.trim().is_empty()
            && passage.len() <= 4096,
        "Invalid complete indexed text passage"
    );
    ensure!(
        row.text_excerpt.as_deref() == Some(text_excerpt(passage)),
        "Text excerpt does not match indexed passage"
    );
    Ok(())
}

pub(crate) fn text_excerpt(text: &str) -> &str {
    let mut end = text.len().min(MAX_TEXT_EXCERPT_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn validate_record_metadata(row: &Record) -> Result<()> {
    if row.kind == "source" {
        ensure!(
            row.media_kind.is_empty()
                && row.thumbnail.is_none()
                && row.text_excerpt.is_none()
                && row.start_ms.is_none()
                && row.end_ms.is_none(),
            "Invalid source media metadata"
        );
        return Ok(());
    }
    ensure!(
        matches!(
            row.media_kind.as_str(),
            "image" | "text" | "audio" | "video"
        ),
        "Invalid media kind"
    );
    let visual = matches!(row.media_kind.as_str(), "image" | "video");
    ensure!(
        if visual {
            row.width > 0
                && row.height > 0
                && row.width <= 65536
                && row.height <= 65536
                && u64::from(row.width) * u64::from(row.height) <= 100_000_000
        } else {
            row.width == 0 && row.height == 0
        },
        "Invalid media dimensions"
    );
    ensure!(row.thumbnail.is_some() == visual, "Invalid media thumbnail");
    ensure!(
        row.thumbnail.as_ref().is_none_or(|value| !value.is_empty()),
        "Empty thumbnail path"
    );
    match row.media_kind.as_str() {
        "image" => ensure!(
            row.segment_index == 0
                && row.start_ms.is_none()
                && row.end_ms.is_none()
                && row.text_excerpt.is_none(),
            "Invalid image segment metadata"
        ),
        "text" => ensure!(
            row.segment_index < crate::media::MAX_TEXT_SEGMENTS as u32
                && row.start_ms.is_none()
                && row.end_ms.is_none()
                && row.text_excerpt.is_some(),
            "Invalid text segment metadata"
        ),
        _ => ensure!(
            row.segment_index < crate::media::MAX_TIMED_SEGMENTS as u32
                && row
                    .start_ms
                    .zip(row.end_ms)
                    .is_some_and(|(start, end)| start < end
                        && end <= crate::media::MAX_DURATION_MS
                        && end - start <= crate::media::SEGMENT_DURATION_MS)
                && row.text_excerpt.is_none(),
            "Invalid timed segment metadata"
        ),
    }
    ensure!(
        row.text_excerpt
            .as_ref()
            .is_none_or(|value| value.len() <= MAX_TEXT_EXCERPT_BYTES && !value.contains('\0')),
        "Invalid indexed text excerpt"
    );
    Ok(())
}
impl Store {
    pub fn open(data: &Path, read_only: bool) -> Result<Self> {
        let store = Self {
            path: data.join("catalog.lancedb"),
            read_only,
            connection: OnceLock::new(),
            text_maintenance: Mutex::new(TextMaintenance::default()),
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .context("Start LanceDB runtime")?,
        };
        store.runtime.block_on(async {
            // An absent or completely empty directory has no catalog to open.
            // Inspect it explicitly so permissions and dangling symlinks cannot
            // be mistaken for first-run state. Never repair partial storage.
            let empty = match std::fs::symlink_metadata(&store.path) {
                Ok(metadata) => {
                    ensure!(
                        metadata.is_dir() && !metadata.file_type().is_symlink(),
                        "LanceDB catalog must contain real directories, not symlinks"
                    );
                    std::fs::read_dir(&store.path)
                        .with_context(|| format!("Read catalog directory {:?}", store.path))?
                        .next()
                        .transpose()
                        .context("Inspect catalog directory entry")?
                        .is_none()
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("Inspect catalog directory {:?}", store.path));
                }
            };
            if empty {
                ensure!(
                    !read_only,
                    "Search catalog is unavailable; index a folder in Fractal Search first"
                );
                ensure!(
                    !data.join("search.db").try_exists().context("Inspect old development catalog")?,
                    "Old development catalog detected; use a fresh data directory and reindex"
                );
                eprintln!(
                    "[Catalog] Creating empty catalog at {:?}; add folders in Fractal Search to index media",
                    store.path
                );
                let db = lancedb::connect(store.uri()?)
                    .execute()
                    .await
                    .context("Connect to empty LanceDB catalog")?;
                store.connection.set(db.clone()).map_err(|_| anyhow::anyhow!("Catalog connection was initialized twice"))?;
                let table = db.create_empty_table("assets", schema())
                    .execute()
                    .await
                    .context("Create empty LanceDB catalog")?;
                table.create_index(&["indexed_text"], Index::FTS(text_index_config()))
                    .name(TEXT_INDEX_NAME.to_owned())
                    .execute().await.context("Create text BM25 index; use a fresh data directory and reindex if creation was interrupted")?;
            }
            let table = store.table().await?;
            if !read_only {
                store.initialize_text_maintenance(&table, empty).await?;
            }
            Ok::<_, anyhow::Error>(())
        })?;
        Ok(store)
    }
    fn uri(&self) -> Result<&str> {
        self.path.to_str().context("LanceDB requires UTF-8 paths")
    }
    async fn table(&self) -> Result<Table> {
        for path in [&self.path, &self.path.join("assets.lance")] {
            let metadata = std::fs::symlink_metadata(path)
                .context("LanceDB catalog storage is missing; stop and reopen Fractal Search; only absent or empty catalog directories can be recreated")?;
            ensure!(
                metadata.is_dir() && !metadata.file_type().is_symlink(),
                "LanceDB catalog must contain real directories, not symlinks"
            );
        }
        let db = if let Some(db) = self.connection.get() {
            db.clone()
        } else {
            let db = lancedb::connect(self.uri()?).execute().await?;
            // Store::open initializes this before sharing the Store.
            self.connection
                .set(db.clone())
                .map_err(|_| anyhow::anyhow!("Catalog connection was initialized twice"))?;
            db
        };
        let table = db
            .open_table("assets")
            .execute()
            .await
            .context("Open LanceDB catalog")?;
        ensure!(
            table.schema().await?.as_ref() == schema().as_ref(),
            "Incompatible catalog schema, model profile or embedding contract; use a fresh data directory and reindex"
        );
        Self::validate_text_index(&table).await?;
        Ok(table)
    }
    async fn text_index_uuid(table: &Table) -> Result<String> {
        table
            .list_indices()
            .await?
            .into_iter()
            .find(|index| index.name == TEXT_INDEX_NAME)
            .and_then(|index| index.index_uuid)
            .context("Text BM25 index is missing its persisted identity")
    }
    async fn initialize_text_maintenance(&self, table: &Table, fresh: bool) -> Result<()> {
        let index_uuid = Self::text_index_uuid(table).await?;
        let path = self.path.join(TEXT_MAINTENANCE_STATE);
        let existing = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => {
                ensure!(
                    metadata.is_file()
                        && !metadata.file_type().is_symlink()
                        && metadata.len() <= 4096,
                    "Invalid text-index maintenance state file"
                );
                Some(
                    serde_json::from_slice::<TextMaintenance>(&std::fs::read(&path)?).context(
                        "Invalid text-index maintenance state; committed files remain searchable",
                    )?,
                )
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).context("Read text-index maintenance state"),
        };
        let state = if let Some(mut state) = existing {
            ensure!(
                state.format == 1 && state.indexed_generation <= state.data_generation,
                "Incompatible text-index maintenance state"
            );
            if state.index_uuid != index_uuid {
                // A previous replacement may have committed before its state
                // checkpoint. Conservatively rebuild rather than guess coverage.
                state.data_generation = state
                    .data_generation
                    .checked_add(1)
                    .context("Text-index generation overflow")?;
                state.index_uuid = index_uuid;
            }
            state.recovery_due = state.dirty();
            state
        } else {
            // Schema 4 predates this optional sidecar. Its tokenizer/schema stay
            // untouched; one conservative checkpoint establishes exact tracking.
            let has_text = !fresh
                && table
                    .count_rows(Some("kind = 'asset' AND media_kind = 'text'".into()))
                    .await?
                    > 0;
            TextMaintenance {
                format: 1,
                index_uuid,
                data_generation: u64::from(has_text),
                indexed_generation: 0,
                changed_rows: 0,
                last_indexed_at: unix_seconds()?,
                recovery_due: has_text,
            }
        };
        self.persist_text_maintenance(&state)?;
        *self
            .text_maintenance
            .lock()
            .map_err(|_| anyhow::anyhow!("Text-index maintenance lock poisoned"))? = state;
        Ok(())
    }
    fn persist_text_maintenance(&self, state: &TextMaintenance) -> Result<()> {
        ensure!(!self.read_only, "Catalog is read-only");
        let path = self.path.join(TEXT_MAINTENANCE_STATE);
        let temporary = self
            .path
            .join(format!(".text-index-{}.part", uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(&serde_json::to_vec(state)?)?;
            file.sync_all()?;
            std::fs::rename(&temporary, path)?;
            // Persist the rename before permitting a row commit. File sync alone
            // does not provide the write-ahead ordering after a machine crash.
            std::fs::File::open(&self.path)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result.context("Persist text-index maintenance generation")
    }
    fn mark_text_dirty(&self, state: &mut TextMaintenance, changed_rows: usize) -> Result<()> {
        let mut next = state.clone();
        next.data_generation = next
            .data_generation
            .checked_add(1)
            .context("Text-index generation overflow")?;
        next.changed_rows = next.changed_rows.saturating_add(changed_rows.max(1) as u64);
        self.persist_text_maintenance(&next)?;
        *state = next;
        Ok(())
    }
    async fn validate_text_index(table: &Table) -> Result<()> {
        let indices = table
            .list_indices()
            .await
            .context("Inspect text BM25 index")?;
        let matching = indices
            .iter()
            .filter(|index| index.name == TEXT_INDEX_NAME)
            .collect::<Vec<_>>();
        ensure!(
            matching.len() == 1,
            "Missing text BM25 index; use a fresh data directory and reindex"
        );
        let index = matching[0];
        ensure!(
            index.index_type == IndexType::FTS && index.columns == ["indexed_text"],
            "Incompatible text BM25 index; use a fresh data directory and reindex"
        );
        ensure!(
            indices
                .iter()
                .filter(|index| index.columns.iter().any(|column| column == "indexed_text"))
                .count()
                == 1,
            "Ambiguous text BM25 indexes; use a fresh data directory and reindex"
        );
        let details: serde_json::Value =
            serde_json::from_str(index.index_details.as_deref().context(
                "Missing BM25 tokenizer contract; use a fresh data directory and reindex",
            )?)
            .context("Invalid BM25 tokenizer contract; use a fresh data directory and reindex")?;
        // LanceDB reads the full persisted params from the FTS index file.
        // Only build-time worker/memory settings are omitted from those params.
        ensure!(
            details == serde_json::to_value(text_index_config())?,
            "Incompatible BM25 tokenizer contract; use a fresh data directory and reindex"
        );
        Ok(())
    }
    async fn snapshot(&self) -> Result<Table> {
        let table = self.table().await?;
        table.checkout(table.version().await?).await?;
        Ok(table)
    }
    async fn rows(
        table: &Table,
        filter: &str,
        offset: usize,
        limit: usize,
        ordered: bool,
        vectors: bool,
    ) -> Result<Vec<Record>> {
        ensure!(
            (1..=MAX_ROWS + 1).contains(&limit) && offset <= MAX_ROWS,
            "Catalog query exceeds row limits"
        );
        let mut columns = COLUMNS.to_vec();
        if vectors {
            columns.push("embedding");
        }
        let mut query = table
            .query()
            .only_if(filter)
            .offset(offset)
            .limit(limit)
            .select(Select::columns(&columns));
        if ordered {
            query = query.order_by(Some(vec![
                ColumnOrdering::asc_nulls_last("sort_filename".into()),
                ColumnOrdering::asc_nulls_last("path".into()),
                ColumnOrdering::asc_nulls_last("segment_index".into()),
                ColumnOrdering::asc_nulls_last("media_kind".into()),
                ColumnOrdering::asc_nulls_last("asset_id".into()),
            ]));
        }
        let mut stream = query.execute().await?;
        let mut rows = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            for i in 0..batch.num_rows() {
                rows.push(record(&batch, i)?);
            }
            ensure!(rows.len() <= limit, "Catalog exceeded row limit");
        }
        Ok(rows)
    }
    pub fn list(&self, filter: &str, limit: usize, vectors: bool) -> Result<Vec<Record>> {
        self.runtime.block_on(async {
            Self::rows(&self.snapshot().await?, filter, 0, limit, false, vectors).await
        })
    }
    /// Read one source once per scan, using bounded keyset pages on a single
    /// pinned snapshot. No vectors or private complete passages are materialized.
    pub fn source_manifest(&self, source_id: i64) -> Result<Vec<Record>> {
        ensure!(source_id > 0, "Invalid manifest source ID");
        self.runtime.block_on(async {
            let table = self.snapshot().await?;
            let mut rows = Vec::new();
            let mut last_id: Option<String> = None;
            loop {
                let mut filter = asset_filter(Some(source_id));
                if let Some(id) = &last_id {
                    filter.push_str(&format!(" AND asset_id > {}", quote(id)));
                }
                let mut stream = table
                    .query()
                    .only_if(filter)
                    .select(Select::columns(COLUMNS))
                    .order_by(Some(vec![ColumnOrdering::asc_nulls_last(
                        "asset_id".into(),
                    )]))
                    .limit(MANIFEST_PAGE_SIZE)
                    .execute()
                    .await?;
                let mut page_count = 0;
                while let Some(batch) = stream.try_next().await? {
                    for index in 0..batch.num_rows() {
                        let row = record(&batch, index)?;
                        ensure!(
                            last_id.as_ref().is_none_or(|id| row.asset_id > *id),
                            "Duplicate or unordered manifest asset identity"
                        );
                        last_id = Some(row.asset_id.clone());
                        rows.push(row);
                        page_count += 1;
                        ensure!(rows.len() <= MAX_ROWS, "Too many catalog assets");
                    }
                }
                ensure!(
                    page_count <= MANIFEST_PAGE_SIZE,
                    "Catalog exceeded manifest page limit"
                );
                if page_count < MANIFEST_PAGE_SIZE {
                    break;
                }
            }
            Ok(rows)
        })
    }
    pub fn get(&self, filter: &str, vectors: bool) -> Result<Option<Record>> {
        let mut rows = self.list(filter, 2, vectors)?;
        ensure!(rows.len() <= 1, "Duplicate catalog identity");
        Ok(rows.pop())
    }
    pub fn asset(&self, id: &str, vectors: bool) -> Result<Record> {
        self.get(
            &format!("kind = 'asset' AND asset_id = {}", quote(id)),
            vectors,
        )?
        .context("Asset is not in the search catalog")
    }
    #[cfg(test)]
    pub fn browse(
        &self,
        offset: usize,
        limit: usize,
        source: Option<i64>,
    ) -> Result<(usize, Vec<Record>)> {
        self.browse_filtered(offset, limit, source, None)
    }
    pub fn browse_filtered(
        &self,
        offset: usize,
        limit: usize,
        source: Option<i64>,
        media_kind: Option<MediaKind>,
    ) -> Result<(usize, Vec<Record>)> {
        let filter = filtered_assets(source, media_kind);
        self.runtime.block_on(async {
            let table = self.snapshot().await?;
            let total = table.count_rows(Some(filter.clone())).await?;
            let rows = Self::rows(&table, &filter, offset, limit, true, false).await?;
            Ok((total, rows))
        })
    }
    pub fn status(&self) -> Result<(usize, usize, Vec<(Record, usize, usize)>)> {
        self.runtime.block_on(async {
            let table = self.snapshot().await?;
            let sources =
                Self::rows(&table, "kind = 'source'", 0, MAX_ROWS + 1, false, false).await?;
            ensure!(sources.len() <= MAX_ROWS, "Too many catalog sources");
            // Status does not materialize excerpts or vectors for the whole
            // library. Only source/file identities are needed for these counts.
            let mut stream = table
                .query()
                .only_if("kind = 'asset'")
                .select(Select::columns(&["source_id", "path"]))
                .limit(MAX_ROWS + 1)
                .execute()
                .await?;
            let mut total = 0;
            let mut files = HashSet::new();
            let mut per_source: HashMap<i64, (usize, HashSet<Arc<str>>)> = HashMap::new();
            while let Some(batch) = stream.try_next().await? {
                let ids = array::<Int64Array>(&batch, "source_id")?;
                let paths = array::<StringArray>(&batch, "path")?;
                for index in 0..batch.num_rows() {
                    ensure!(
                        !ids.is_null(index) && !paths.is_null(index),
                        "Null catalog file identity"
                    );
                    total += 1;
                    ensure!(total <= MAX_ROWS, "Too many catalog assets");
                    let path: Arc<str> = Arc::from(paths.value(index));
                    files.insert(path.clone());
                    let entry = per_source.entry(ids.value(index)).or_default();
                    entry.0 += 1;
                    entry.1.insert(path);
                }
            }
            let counts = sources
                .into_iter()
                .map(|source| {
                    let (assets, paths) = per_source.remove(&source.source_id).unwrap_or_default();
                    (source, assets, paths.len())
                })
                .collect();
            Ok((total, files.len(), counts))
        })
    }
    /// Atomically replace every segment belonging to exactly one source file.
    /// All rows are validated before opening a write transaction. Matching UUIDs
    /// retain identity; segments no longer present disappear in the same commit.
    pub fn replace_file(&self, source_id: i64, path: &str, rows: &[Record]) -> Result<()> {
        ensure!(!self.read_only, "Catalog is read-only");
        ensure!(
            !rows.is_empty() && rows.len() <= MAX_ROWS,
            "Invalid segment count"
        );
        let mut ids = HashSet::new();
        let mut segments = HashSet::new();
        let first = &rows[0];
        for row in rows {
            ensure!(
                row.kind == "asset" && row.source_id == source_id && row.path == path,
                "Replacement contains an unrelated source file"
            );
            ensure!(
                row.bytes == first.bytes && row.modified == first.modified,
                "Replacement contains inconsistent file metadata"
            );
            ensure!(
                ids.insert(row.asset_id.as_str())
                    && segments.insert((row.media_kind.as_str(), row.segment_index)),
                "Replacement contains duplicate asset or segment identities"
            );
        }
        for kind in ["image", "text", "audio", "video"] {
            let indices = segments
                .iter()
                .filter_map(|(media, index)| (*media == kind).then_some(*index))
                .collect::<Vec<_>>();
            ensure!(
                indices.is_empty()
                    || indices.iter().copied().max() == Some((indices.len() - 1) as u32),
                "Replacement segment indices must be contiguous from zero within each modality"
            );
        }
        let batches = rows.iter().map(batch).collect::<Result<Vec<_>>>()?;
        let filter = format!(
            "kind = 'asset' AND source_id = {source_id} AND path = {}",
            quote(path)
        );
        let mut maintenance = self
            .text_maintenance
            .lock()
            .map_err(|_| anyhow::anyhow!("Text-index maintenance lock poisoned"))?;
        self.runtime.block_on(async {
            let table = self.table().await?;
            let total = table.count_rows(Some("kind = 'asset'".into())).await?;
            let replaced = table.count_rows(Some(filter.clone())).await?;
            let text_changed = rows.iter().any(|row| row.media_kind == "text")
                || table
                    .count_rows(Some(format!("({filter}) AND media_kind = 'text'")))
                    .await?
                    > 0;
            ensure!(
                total - replaced + rows.len() <= MAX_ROWS,
                "Catalog segment limit reached; choose smaller source folders"
            );
            if text_changed {
                self.mark_text_dirty(&mut maintenance, replaced.max(rows.len()))?;
            }
            let mut merge = table.merge_insert(&["asset_id"]);
            merge
                .when_matched_update_all(None)
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(Some(filter));
            merge
                .execute(Box::new(RecordBatchIterator::new(
                    batches.into_iter().map(Ok),
                    schema(),
                )))
                .await
                .context("Replace indexed file segments")?;
            Ok(())
        })
    }
    pub fn upsert(&self, row: &Record) -> Result<()> {
        ensure!(!self.read_only, "Catalog is read-only");
        let batch = batch(row)?;
        let mut maintenance = self
            .text_maintenance
            .lock()
            .map_err(|_| anyhow::anyhow!("Text-index maintenance lock poisoned"))?;
        self.runtime.block_on(async {
            let table = self.table().await?;
            let replaces_text = table
                .count_rows(Some(format!(
                    "asset_id = {} AND media_kind = 'text'",
                    quote(&row.asset_id)
                )))
                .await?
                > 0;
            if row.media_kind == "text" || replaces_text {
                self.mark_text_dirty(&mut maintenance, 1)?;
            }
            let mut merge = table.merge_insert(&["asset_id"]);
            merge
                .when_matched_update_all(None)
                .when_not_matched_insert_all();
            merge
                .execute(Box::new(RecordBatchIterator::new(
                    vec![Ok(batch)],
                    schema(),
                )))
                .await?;
            Ok(())
        })
    }
    pub fn delete(&self, filter: &str) -> Result<()> {
        ensure!(!self.read_only, "Catalog is read-only");
        let mut maintenance = self
            .text_maintenance
            .lock()
            .map_err(|_| anyhow::anyhow!("Text-index maintenance lock poisoned"))?;
        self.runtime.block_on(async {
            let table = self.table().await?;
            let text_changed = table
                .count_rows(Some(format!(
                    "({filter}) AND kind = 'asset' AND media_kind = 'text'"
                )))
                .await?;
            if text_changed > 0 {
                self.mark_text_dirty(&mut maintenance, text_changed)?;
            }
            table
                .delete(filter)
                .await
                .context("Delete catalog records")?;
            Ok(())
        })
    }
    #[cfg(test)]
    pub fn search(
        &self,
        vector: &[f32],
        limit: usize,
        source: Option<i64>,
    ) -> Result<Vec<(Record, f32)>> {
        Ok(self
            .search_filtered("", vector, limit, source, None)?
            .into_iter()
            .map(|hit| (hit.row, hit.distance))
            .collect())
    }
    pub fn search_filtered(
        &self,
        query: &str,
        vector: &[f32],
        limit: usize,
        source: Option<i64>,
        media_kind: Option<MediaKind>,
    ) -> Result<Vec<SearchHit>> {
        self.search_filtered_grouped(query, vector, limit, source, media_kind, false)
    }
    pub fn search_filtered_grouped(
        &self,
        query: &str,
        vector: &[f32],
        limit: usize,
        source: Option<i64>,
        media_kind: Option<MediaKind>,
        group_text_by_file: bool,
    ) -> Result<Vec<SearchHit>> {
        validate_embedding(vector)?;
        ensure!((1..=1000).contains(&limit), "Invalid search limit");
        let hybrid = media_kind == Some(MediaKind::Text);
        ensure!(
            !group_text_by_file || hybrid,
            "File grouping requires the Text media filter"
        );
        if hybrid {
            crate::agent::validate_query(query)?;
        }
        self.runtime.block_on(async {
            let table = self.snapshot().await?;
            let filter = filtered_assets(source, media_kind);
            let mut candidates = if hybrid {
                hybrid_candidate_limit(limit)
            } else {
                limit
            };
            let mut dense = Self::dense_candidates(&table, vector, candidates, &filter).await?;
            if !hybrid {
                // Preserve cosine ordering; snippet extraction cannot alter rank.
                dense.sort_by(|a, b| {
                    (1.0 - b.1)
                        .clamp(-1.0, 1.0)
                        .total_cmp(&(1.0 - a.1).clamp(-1.0, 1.0))
                        .then(a.0.asset_id.cmp(&b.0.asset_id))
                });
                let mut hits = dense
                    .into_iter()
                    .map(|(row, distance)| SearchHit {
                        row,
                        distance,
                        hybrid_score: None,
                        additional_matches: Vec::new(),
                    })
                    .collect::<Vec<_>>();
                Self::populate_snippets(&table, query, &mut hits).await?;
                return Ok(hits);
            }
            let mut lexical = Self::lexical_candidates(&table, query, candidates, &filter).await?;
            // A long document can consume the initial passage budget. Widen each
            // leg only while it has more candidates and too few unique files.
            // All attempts use this snapshot and retain the 1000-per-leg cap.
            while group_text_by_file
                && candidates < 1000
                && ((dense.len() == candidates && unique_documents(&dense) < limit)
                    || (lexical.len() == candidates && unique_documents(&lexical) < limit))
            {
                candidates = candidates.saturating_mul(2).min(1000);
                dense = Self::dense_candidates(&table, vector, candidates, &filter).await?;
                lexical = Self::lexical_candidates(&table, query, candidates, &filter).await?;
            }
            let mut hits = if group_text_by_file {
                fuse_document_candidates(dense, lexical, vector, limit)?
            } else {
                fuse_candidates(dense, lexical, vector, limit)?
            };
            Self::populate_snippets(&table, query, &mut hits).await?;
            Ok(hits)
        })
    }
    /// Native retrieval ablations for the labeled evaluation harness. BM25 does
    /// not require a model/vector; its distance is unused by the public adapter.
    pub fn search_text_evaluation(
        &self,
        query: &str,
        vector: &[f32],
        limit: usize,
        source: Option<i64>,
        mode: TextSearchMode,
    ) -> Result<Vec<SearchHit>> {
        crate::agent::validate_query(query)?;
        ensure!(
            (1..=1000).contains(&limit),
            "Invalid evaluation search limit"
        );
        if mode != TextSearchMode::Bm25 {
            validate_embedding(vector)?;
        }
        self.runtime.block_on(async {
            let table = self.snapshot().await?;
            let filter = filtered_assets(source, Some(MediaKind::Text));
            let candidates = hybrid_candidate_limit(limit);
            let mut hits = match mode {
                TextSearchMode::Bm25 => {
                    Self::lexical_candidates(&table, query, candidates, &filter)
                        .await?
                        .into_iter()
                        .take(limit)
                        .map(|(row, _)| SearchHit {
                            row,
                            distance: 0.0,
                            hybrid_score: None,
                            additional_matches: Vec::new(),
                        })
                        .collect()
                }
                TextSearchMode::Dense => {
                    Self::dense_candidates(&table, vector, candidates, &filter)
                        .await?
                        .into_iter()
                        .take(limit)
                        .map(|(row, distance)| SearchHit {
                            row,
                            distance,
                            hybrid_score: None,
                            additional_matches: Vec::new(),
                        })
                        .collect()
                }
                TextSearchMode::Hybrid => fuse_candidates(
                    Self::dense_candidates(&table, vector, candidates, &filter).await?,
                    Self::lexical_candidates(&table, query, candidates, &filter).await?,
                    vector,
                    limit,
                )?,
            };
            Self::populate_snippets(&table, query, &mut hits).await?;
            Ok(hits)
        })
    }
    async fn populate_snippets(table: &Table, query: &str, hits: &mut [SearchHit]) -> Result<()> {
        if query.trim().is_empty() {
            return Ok(());
        }
        let ids = hits
            .iter()
            .flat_map(|hit| std::iter::once(hit).chain(&hit.additional_matches))
            .filter(|hit| hit.row.media_kind == "text")
            .map(|hit| hit.row.asset_id.clone())
            .collect::<Vec<_>>();
        ensure!(ids.len() <= 2000, "Too many snippet candidates");
        if ids.is_empty() {
            return Ok(());
        }
        let query_terms = lancedb::tokenize(query, &text_index_config())
            .context("Tokenize search snippet query")?
            .into_iter()
            .map(|token| token.text)
            .collect::<HashSet<_>>();
        let mut passages = HashMap::new();
        for page in ids.chunks(64) {
            let filter = format!(
                "kind = 'asset' AND media_kind = 'text' AND asset_id IN ({})",
                page.iter()
                    .map(|id| quote(id))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let mut stream = table
                .query()
                .only_if(filter)
                .limit(page.len())
                .select(Select::columns(&["asset_id", "indexed_text"]))
                .execute()
                .await
                .context("Read winning stored text passages")?;
            while let Some(batch) = stream.try_next().await? {
                let ids = array::<StringArray>(&batch, "asset_id")?;
                let text = array::<StringArray>(&batch, "indexed_text")?;
                for index in 0..batch.num_rows() {
                    ensure!(
                        !ids.is_null(index) && !text.is_null(index),
                        "Missing winning text passage"
                    );
                    ensure!(
                        passages
                            .insert(ids.value(index).to_owned(), text.value(index).to_owned())
                            .is_none(),
                        "Duplicate winning text passage"
                    );
                }
            }
        }
        for hit in hits.iter_mut().flat_map(|hit| {
            std::iter::once(&mut hit.row).chain(
                hit.additional_matches
                    .iter_mut()
                    .map(|extra| &mut extra.row),
            )
        }) {
            if hit.media_kind != "text" {
                continue;
            }
            hit.indexed_text = Some(
                passages
                    .remove(&hit.asset_id)
                    .context("Winning passage disappeared from snapshot")?,
            );
            validate_indexed_text(hit)?;
            let indexed = hit.indexed_text.take().expect("validated stored passage");
            let passage = indexed
                .strip_prefix(&format!("{}\n", hit.filename))
                .context("Invalid winning passage filename")?;
            hit.text_excerpt = Some(match_centered_excerpt(passage, &query_terms)?);
            // Search projections never retain the private passage or vector.
            hit.embedding = None;
        }
        Ok(())
    }
    async fn dense_candidates(
        table: &Table,
        vector: &[f32],
        limit: usize,
        filter: &str,
    ) -> Result<Vec<(Record, f32)>> {
        let mut stream = table
            .vector_search(vector)?
            .distance_type(DistanceType::Cosine)
            .bypass_vector_index()
            .only_if(filter)
            .limit(limit)
            .select(Select::columns(COLUMNS))
            .execute()
            .await
            .context("Search semantic candidates")?;
        let mut hits = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let distances = array::<Float32Array>(&batch, "_distance")?;
            for i in 0..batch.num_rows() {
                ensure!(
                    !distances.is_null(i) && distances.value(i).is_finite(),
                    "Invalid vector search distance"
                );
                hits.push((record(&batch, i)?, distances.value(i)));
            }
            ensure!(hits.len() <= limit, "Search exceeded candidate limit");
        }
        hits.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.asset_id.cmp(&b.0.asset_id)));
        Ok(hits)
    }
    async fn lexical_candidates(
        table: &Table,
        query: &str,
        limit: usize,
        filter: &str,
    ) -> Result<Vec<(Record, f32)>> {
        // MatchQuery treats user text as literal OR terms, never as SQL, Boolean
        // syntax or a phrase parser. Tokenless input is the only empty shortcut.
        if lancedb::tokenize(query, &text_index_config())
            .context("Tokenize text search query")?
            .is_empty()
        {
            return Ok(Vec::new());
        }
        let mut columns = COLUMNS.to_vec();
        columns.push("embedding");
        let mut stream = table
            .query()
            .full_text_search(
                FullTextSearchQuery::new(query.to_owned())
                    .with_column("indexed_text".to_owned())?,
            )
            .only_if(filter)
            .limit(limit)
            .select(Select::columns(&columns))
            .execute()
            .await
            .context("Search text BM25 candidates")?;
        let mut hits = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let scores = array::<Float32Array>(&batch, "_score")?;
            for i in 0..batch.num_rows() {
                ensure!(
                    !scores.is_null(i) && scores.value(i).is_finite(),
                    "Invalid BM25 search score"
                );
                hits.push((record(&batch, i)?, scores.value(i)));
            }
            ensure!(hits.len() <= limit, "BM25 search exceeded candidate limit");
        }
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.asset_id.cmp(&b.0.asset_id)));
        Ok(hits)
    }
    pub fn checkpoint(&self) -> Result<()> {
        // Pinned Lance 10 OptimizeOptions supports index_names/append, but cannot
        // pass memory_limit_mb/num_workers through the reloaded FTS params. Do
        // not pretend that an unbounded incremental build meets our budget.
        // Thresholded replacement remains explicit and never compacts or prunes.
        self.runtime.block_on(async {
            if self.read_only {
                self.table().await?;
                return Ok(());
            }
            // Serialize before opening a snapshot: no newer text generation may
            // be acknowledged by an index built from an older table handle.
            let mut state = self.text_maintenance.lock()
                .map_err(|_| anyhow::anyhow!("Text-index maintenance lock poisoned"))?;
            let table = self.table().await?;
            if !state.dirty() {
                return Ok(());
            }
            let text_rows = table.count_rows(Some("kind = 'asset' AND media_kind = 'text'".into())).await?;
            let now = unix_seconds()?;
            if !state.due(text_rows, now) {
                return Ok(());
            }
            table.create_index(&["indexed_text"], Index::FTS(text_index_config()))
                .name(TEXT_INDEX_NAME.to_owned()).replace(true)
                .execute().await.context("Indexed files are committed, but rebuilding the text BM25 index failed; retry the checkpoint")?;
            Self::validate_text_index(&table).await?;
            let mut completed = state.clone();
            completed.index_uuid = Self::text_index_uuid(&table).await?;
            completed.indexed_generation = completed.data_generation;
            completed.changed_rows = 0;
            completed.last_indexed_at = now;
            completed.recovery_due = false;
            self.persist_text_maintenance(&completed)?;
            *state = completed;
            Ok(())
        })
    }
    #[cfg(test)]
    pub fn update_for_test(&self, id: &str, column: &str, value: &str) {
        self.runtime.block_on(async {
            self.table()
                .await
                .unwrap()
                .update()
                .only_if(format!("asset_id = {}", quote(id)))
                .column(column, quote(value))
                .execute()
                .await
                .unwrap();
        });
    }
}
pub(crate) fn asset_filter(source: Option<i64>) -> String {
    match source {
        Some(id) => format!("kind = 'asset' AND source_id = {id}"),
        None => "kind = 'asset'".into(),
    }
}

fn filtered_assets(source: Option<i64>, media_kind: Option<MediaKind>) -> String {
    let mut filter = asset_filter(source);
    if let Some(kind) = media_kind {
        filter.push_str(&format!(" AND media_kind = {}", quote(kind.as_str())));
    }
    filter
}
fn hybrid_candidate_limit(limit: usize) -> usize {
    limit.saturating_mul(4).clamp(100, 1000)
}

fn unique_documents(candidates: &[(Record, f32)]) -> usize {
    candidates
        .iter()
        .map(|(row, _)| (row.source_id, row.path.as_str()))
        .collect::<HashSet<_>>()
        .len()
}

/// Use the same ICU/lowercase/folding analyzer as BM25. The public tokenizer
/// reports token positions, not byte offsets, so locate a matching token with
/// bounded UTF-8 prefix probes, then verify the final window with that analyzer.
/// Filename-only, semantic-only and unlocatable matches use the stored prefix.
fn match_centered_excerpt(passage: &str, query_terms: &HashSet<String>) -> Result<String> {
    if passage.len() <= MAX_TEXT_EXCERPT_BYTES || query_terms.is_empty() {
        return Ok(text_excerpt(passage).to_owned());
    }
    let config = text_index_config();
    let tokens =
        lancedb::tokenize(passage, &config).context("Tokenize stored passage for snippet")?;
    let Some(target) = tokens
        .iter()
        .find(|token| query_terms.contains(&token.text))
    else {
        return Ok(text_excerpt(passage).to_owned());
    };
    let boundaries = passage
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(passage.len()))
        .collect::<Vec<_>>();
    let mut low = 0;
    let mut high = boundaries.len() - 1;
    while low < high {
        let mid = low + (high - low) / 2;
        let prefix = lancedb::tokenize(&passage[..boundaries[mid]], &config)
            .context("Locate stored passage snippet match")?;
        if prefix
            .iter()
            .any(|token| token.position == target.position && token.text == target.text)
        {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let center = boundaries[low].saturating_sub(target.text.len() / 2);
    let mut start = center
        .saturating_sub(MAX_TEXT_EXCERPT_BYTES / 2)
        .min(passage.len().saturating_sub(MAX_TEXT_EXCERPT_BYTES));
    while !passage.is_char_boundary(start) {
        start += 1;
    }
    let snippet = text_excerpt(&passage[start..]);
    let verified = lancedb::tokenize(snippet, &config)
        .context("Verify stored passage snippet match")?
        .iter()
        .any(|token| query_terms.contains(&token.text));
    if verified {
        return Ok(snippet.to_owned());
    }
    // Context-sensitive ICU boundaries or a term longer than the display budget
    // may not yield a verified complete token in a window. Keep a deterministic,
    // bounded passage-only fallback, rather than inventing substring semantics.
    Ok(text_excerpt(passage).to_owned())
}

fn fuse_document_candidates(
    mut dense: Vec<(Record, f32)>,
    mut lexical: Vec<(Record, f32)>,
    query: &[f32],
    limit: usize,
) -> Result<Vec<SearchHit>> {
    dense.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.asset_id.cmp(&b.0.asset_id)));
    lexical.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.asset_id.cmp(&b.0.asset_id)));
    let mut document_scores: HashMap<(i64, String), f64> = HashMap::new();
    for candidates in [&dense, &lexical] {
        let mut seen = HashSet::new();
        for (row, _) in candidates {
            let key = (row.source_id, row.path.clone());
            if seen.insert(key.clone()) {
                // Rank files after collapsing the leg, not by the original
                // passage rank. Additional passages never add to this score.
                *document_scores.entry(key).or_default() += 1.0 / (RRF_K + seen.len() as f64);
            }
        }
    }
    let passage_hits = fuse_candidates(dense, lexical, query, 2000)?;
    let mut groups: HashMap<(i64, String), SearchHit> = HashMap::new();
    for hit in passage_hits {
        let key = (hit.row.source_id, hit.row.path.clone());
        if let Some(group) = groups.get_mut(&key) {
            if group.additional_matches.len() < MAX_ADDITIONAL_MATCHES {
                group.additional_matches.push(hit);
            }
        } else {
            groups.insert(key, hit);
        }
    }
    let mut hits = groups
        .into_iter()
        .map(|(key, mut hit)| {
            hit.hybrid_score = document_scores.get(&key).copied();
            hit
        })
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| {
        b.hybrid_score
            .unwrap()
            .total_cmp(&a.hybrid_score.unwrap())
            .then(a.row.asset_id.cmp(&b.row.asset_id))
    });
    hits.truncate(limit);
    Ok(hits)
}

fn cosine_distance(query: &[f32], candidate: &[f32]) -> Result<f32> {
    validate_embedding(candidate)?;
    let (dot, qnorm, cnorm) = query.iter().zip(candidate).fold(
        (0.0_f64, 0.0_f64, 0.0_f64),
        |(dot, qnorm, cnorm), (&q, &c)| {
            let (q, c) = (f64::from(q), f64::from(c));
            (dot + q * c, qnorm + q * q, cnorm + c * c)
        },
    );
    let distance = 1.0 - dot / (qnorm * cnorm).sqrt();
    ensure!(
        distance.is_finite(),
        "Invalid lexical candidate cosine distance"
    );
    Ok(distance as f32)
}

fn fuse_candidates(
    mut dense: Vec<(Record, f32)>,
    mut lexical: Vec<(Record, f32)>,
    query: &[f32],
    limit: usize,
) -> Result<Vec<SearchHit>> {
    ensure!(
        dense
            .iter()
            .chain(&lexical)
            .all(|(_, score)| score.is_finite()),
        "Non-finite hybrid candidate score"
    );
    dense.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.asset_id.cmp(&b.0.asset_id)));
    lexical.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.asset_id.cmp(&b.0.asset_id)));
    let mut fused: HashMap<String, SearchHit> = HashMap::new();
    for (rank, (row, distance)) in dense.into_iter().enumerate() {
        let id = row.asset_id.clone();
        ensure!(
            fused
                .insert(
                    id,
                    SearchHit {
                        row,
                        distance,
                        hybrid_score: Some(1.0 / (RRF_K + rank as f64 + 1.0)),
                        additional_matches: Vec::new(),
                    }
                )
                .is_none(),
            "Duplicate semantic candidate identity"
        );
    }
    let mut lexical_ids = HashSet::new();
    for (rank, (row, _)) in lexical.into_iter().enumerate() {
        ensure!(
            lexical_ids.insert(row.asset_id.clone()),
            "Duplicate lexical candidate identity"
        );
        let contribution = 1.0 / (RRF_K + rank as f64 + 1.0);
        if let Some(hit) = fused.get_mut(&row.asset_id) {
            *hit.hybrid_score.as_mut().expect("hybrid score") += contribution;
        } else {
            let distance = cosine_distance(
                query,
                row.embedding
                    .as_deref()
                    .context("Lexical candidate is missing its embedding")?,
            )?;
            fused.insert(
                row.asset_id.clone(),
                SearchHit {
                    row,
                    distance,
                    hybrid_score: Some(contribution),
                    additional_matches: Vec::new(),
                },
            );
        }
    }
    let mut hits = fused.into_values().collect::<Vec<_>>();
    hits.sort_by(|a, b| {
        b.hybrid_score
            .unwrap()
            .total_cmp(&a.hybrid_score.unwrap())
            .then(a.row.asset_id.cmp(&b.row.asset_id))
    });
    hits.truncate(limit);
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schema_records_the_single_vector_embedding_contract() {
        let schema = schema();
        assert_eq!(schema.metadata().get("fractal.schema").unwrap(), "4");
        assert_eq!(
            schema
                .metadata()
                .get("fractal.text_search_profile")
                .unwrap(),
            TEXT_SEARCH_PROFILE
        );
        assert_eq!(
            text_index_config().to_training_json().unwrap()["memory_limit"],
            128
        );
        assert_eq!(
            text_index_config().to_training_json().unwrap()["num_workers"],
            2
        );
        assert_eq!(
            schema.metadata().get("fractal.model_profile").unwrap(),
            MODEL_PROFILE
        );
        assert_eq!(
            schema.metadata().get("fractal.embedding_dim").unwrap(),
            &EMBEDDING_DIM.to_string()
        );
        assert_eq!(
            schema.metadata().get("fractal.distance_metric").unwrap(),
            "cosine"
        );
        assert_eq!(
            schema.metadata().get("fractal.normalization").unwrap(),
            "l2"
        );
        assert_eq!(
            schema
                .metadata()
                .get("fractal.vectors_per_segment")
                .unwrap(),
            "1"
        );
        assert_eq!(
            schema.field_with_name("embedding").unwrap().data_type(),
            &DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM as i32
            )
        );
    }

    fn text_record(id: &str, filename: &str, passage: &str, axis: usize) -> Record {
        let mut row = image(id, axis);
        row.media_kind = "text".into();
        row.filename = filename.into();
        row.width = 0;
        row.height = 0;
        row.thumbnail = None;
        row.text_excerpt = Some(text_excerpt(passage).into());
        row.indexed_text = Some(format!("{filename}\n{passage}"));
        row
    }
    fn lexical(store: &Store, query: &str, source: Option<i64>) -> Vec<String> {
        store.runtime.block_on(async {
            let table = store.snapshot().await.unwrap();
            Store::lexical_candidates(
                &table,
                query,
                100,
                &filtered_assets(source, Some(MediaKind::Text)),
            )
            .await
            .unwrap()
            .into_iter()
            .map(|(row, _)| row.asset_id)
            .collect()
        })
    }
    #[test]
    fn hybrid_empty_catalog_full_passage_filename_and_unindexed_appends() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let query = image("q", 0).embedding.unwrap();
        assert!(store
            .search_filtered("needle", &query, 10, None, Some(MediaKind::Text))
            .unwrap()
            .is_empty());
        let long = format!("{} tailneedle", "prefix ".repeat(350));
        let first = text_record("a", "filenamebeacon.md", &long, 1);
        let second = text_record("b", "other.txt", "secondneedle", 0);
        store.upsert(&first).unwrap();
        store.upsert(&second).unwrap();
        assert_eq!(lexical(&store, "tailneedle", None), ["a"]);
        assert_eq!(lexical(&store, "filenamebeacon", None), ["a"]);
        assert_eq!(lexical(&store, "secondneedle", None), ["b"]);
        assert!(!store
            .asset("a", false)
            .unwrap()
            .text_excerpt
            .unwrap()
            .contains("tailneedle"));
        assert!(store.asset("a", true).unwrap().indexed_text.is_none());
        let before = store.runtime.block_on(store.snapshot()).unwrap();
        let version = store.runtime.block_on(before.version()).unwrap();
        store.checkpoint().unwrap();
        assert_eq!(lexical(&store, "tailneedle", None), ["a"]);
        assert_eq!(lexical(&store, "secondneedle", None), ["b"]);
        // Index maintenance preserves historical table versions.
        store.runtime.block_on(async {
            before.checkout(version).await.unwrap();
            assert_eq!(
                Store::lexical_candidates(
                    &before,
                    "tailneedle",
                    10,
                    &filtered_assets(None, Some(MediaKind::Text))
                )
                .await
                .unwrap()
                .len(),
                1
            );
        });
        let reader = Store::open(root.path(), true).unwrap();
        let disk = contents(root.path());
        reader
            .search_filtered("tailneedle", &query, 10, None, Some(MediaKind::Text))
            .unwrap();
        reader
            .browse_filtered(0, 10, None, Some(MediaKind::Text))
            .unwrap();
        reader.checkpoint().unwrap();
        assert_eq!(contents(root.path()), disk);
    }
    #[test]
    fn text_index_rebuilds_only_at_dirty_checkpoints_and_leaves_image_only_writes_alone() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let index_id = |store: &Store| {
            store.runtime.block_on(async {
                store
                    .snapshot()
                    .await
                    .unwrap()
                    .list_indices()
                    .await
                    .unwrap()
                    .into_iter()
                    .find(|index| index.name == TEXT_INDEX_NAME)
                    .unwrap()
                    .index_uuid
                    .unwrap()
            })
        };
        let initial = index_id(&store);
        store.upsert(&image("visual", 0)).unwrap();
        store.checkpoint().unwrap();
        assert_eq!(index_id(&store), initial);
        store
            .upsert(&text_record("document", "note.txt", "passagebeacon", 1))
            .unwrap();
        assert_eq!(index_id(&store), initial);
        store.checkpoint().unwrap();
        let rebuilt = index_id(&store);
        assert_ne!(rebuilt, initial);
        store.checkpoint().unwrap();
        assert_eq!(index_id(&store), rebuilt);
        store.upsert(&image("visual", 1)).unwrap();
        store.delete("asset_id = 'visual'").unwrap();
        store.checkpoint().unwrap();
        assert_eq!(index_id(&store), rebuilt);
        store.delete("asset_id = 'document'").unwrap();
        store.checkpoint().unwrap();
        assert_ne!(index_id(&store), rebuilt);
        assert!(lexical(&store, "passagebeacon", None).is_empty());
    }
    #[test]
    fn committed_unindexed_text_survives_reopen_and_unchanged_checkpoint_refreshes_it() {
        let root = tempfile::tempdir().unwrap();
        let index_id = |store: &Store| {
            store.runtime.block_on(async {
                store
                    .snapshot()
                    .await
                    .unwrap()
                    .list_indices()
                    .await
                    .unwrap()
                    .into_iter()
                    .find(|index| index.name == TEXT_INDEX_NAME)
                    .unwrap()
                    .index_uuid
                    .unwrap()
            })
        };
        let original_index = {
            let writer = Store::open(root.path(), false).unwrap();
            let id = index_id(&writer);
            writer
                .upsert(&text_record(
                    "before-crash",
                    "first.txt",
                    "durableunindexedbeacon",
                    0,
                ))
                .unwrap();
            // Simulate process loss after its completed row transaction, without
            // calling the separate derived-index maintenance checkpoint.
            id
        };
        let reader = Store::open(root.path(), true).unwrap();
        let before_read = contents(root.path());
        assert_eq!(
            lexical(&reader, "durableunindexedbeacon", None),
            ["before-crash"]
        );
        let query = image("q", 0).embedding.unwrap();
        let hits = reader
            .search_filtered(
                "durableunindexedbeacon",
                &query,
                10,
                None,
                Some(MediaKind::Text),
            )
            .unwrap();
        assert_eq!(hits[0].row.asset_id, "before-crash");
        assert_eq!(contents(root.path()), before_read);
        let writer = Store::open(root.path(), false).unwrap();
        writer.checkpoint().unwrap();
        // The write-ahead generation survives process loss, so an unchanged
        // scan's checkpoint repairs the index without waiting for new text.
        assert_ne!(index_id(&writer), original_index);
        let recovered_index = index_id(&writer);
        writer.checkpoint().unwrap();
        assert_eq!(index_id(&writer), recovered_index);
        writer
            .upsert(&text_record("next-text", "second.txt", "nexttextbeacon", 1))
            .unwrap();
        writer.checkpoint().unwrap();
        assert_ne!(index_id(&writer), original_index);
        assert_eq!(
            lexical(&reader, "durableunindexedbeacon", None),
            ["before-crash"]
        );
        assert_eq!(lexical(&reader, "nexttextbeacon", None), ["next-text"]);
    }
    #[test]
    fn hybrid_replacement_deletion_and_snapshot_visibility_are_atomic() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let mut original = vec![
            text_record("stable", "note.txt", "oldbeacon", 0),
            text_record("removed", "note.txt", "removedbeacon", 1),
        ];
        original[1].path = original[0].path.clone();
        original[1].segment_index = 1;
        store
            .replace_file(42, &original[0].path, &original)
            .unwrap();
        store.checkpoint().unwrap();
        let reader = Store::open(root.path(), true).unwrap();
        let pinned = reader.runtime.block_on(reader.snapshot()).unwrap();
        let mut changed = text_record("stable", "note.txt", "newbeacon", 1);
        changed.modified = "2".into();
        store
            .replace_file(42, &changed.path, &[changed.clone()])
            .unwrap();
        assert!(lexical(&reader, "oldbeacon removedbeacon", None).is_empty());
        assert_eq!(lexical(&reader, "newbeacon", None), ["stable"]);
        assert!(reader.asset("removed", false).is_err());
        reader.runtime.block_on(async {
            assert_eq!(
                Store::lexical_candidates(
                    &pinned,
                    "oldbeacon removedbeacon",
                    10,
                    &filtered_assets(None, Some(MediaKind::Text))
                )
                .await
                .unwrap()
                .len(),
                2
            );
        });
        store
            .delete(&format!("path = {}", quote(&changed.path)))
            .unwrap();
        assert!(lexical(&reader, "newbeacon", None).is_empty());
        store.upsert(&changed).unwrap();
        store.delete("source_id = 42").unwrap();
        assert!(lexical(&reader, "newbeacon", None).is_empty());
        store.checkpoint().unwrap();
        reader.runtime.block_on(async {
            assert_eq!(
                Store::lexical_candidates(
                    &pinned,
                    "oldbeacon",
                    10,
                    &filtered_assets(None, Some(MediaKind::Text))
                )
                .await
                .unwrap()
                .len(),
                1
            );
        });
    }
    #[test]
    fn hybrid_filters_both_candidate_legs_before_top_k() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let rows = (0..105)
            .map(|i| {
                let mut row = text_record(&format!("other-{i:03}"), "other.txt", "sharedbeacon", 0);
                row.path = "/private/other.txt".into();
                row.segment_index = i;
                row
            })
            .collect::<Vec<_>>();
        store.replace_file(42, &rows[0].path, &rows).unwrap();
        let mut scoped = text_record(
            "scoped",
            "scoped.txt",
            "sharedbeacon extra words longer document",
            1,
        );
        scoped.source_id = 43;
        store.upsert(&scoped).unwrap();
        store.upsert(&image("visual", 0)).unwrap();
        let query = image("q", 0).embedding.unwrap();
        let global = store
            .search_filtered("sharedbeacon", &query, 1, None, Some(MediaKind::Text))
            .unwrap();
        assert_ne!(global[0].row.asset_id, "scoped");
        let hits = store
            .search_filtered("sharedbeacon", &query, 1, Some(43), Some(MediaKind::Text))
            .unwrap();
        assert_eq!(hits[0].row.asset_id, "scoped");
        assert!((hits[0].hybrid_score.unwrap() - 2.0 / 61.0).abs() < 1e-12);
        assert_eq!(hits[0].distance, 1.0);
        assert_eq!(
            store
                .browse_filtered(0, 10, Some(43), Some(MediaKind::Text))
                .unwrap()
                .0,
            1
        );
        assert_eq!(
            store
                .browse_filtered(0, 10, None, Some(MediaKind::Image))
                .unwrap()
                .0,
            1
        );
        assert!(store
            .search_filtered("sharedbeacon", &query, 100, None, Some(MediaKind::Text))
            .unwrap()
            .iter()
            .all(|hit| hit.row.media_kind == "text"));
        assert!(store
            .search_filtered("sharedbeacon", &query, 100, None, None)
            .unwrap()
            .iter()
            .all(|hit| hit.hybrid_score.is_none()));
        let mut lexical_only =
            text_record("aaa-lexical-only", "unique.txt", "rarefulltextbeacon", 1);
        lexical_only.source_id = 44;
        store.upsert(&lexical_only).unwrap();
        let lexical_only_hits = store
            .search_filtered("rarefulltextbeacon", &query, 1, None, Some(MediaKind::Text))
            .unwrap();
        assert_eq!(lexical_only_hits[0].row.asset_id, "aaa-lexical-only");
        assert_eq!(lexical_only_hits[0].distance, 1.0);
        assert!((lexical_only_hits[0].hybrid_score.unwrap() - 1.0 / 61.0).abs() < 1e-12);
    }
    #[test]
    fn reciprocal_ranks_are_one_based_deduplicated_and_tie_broken_by_identity() {
        let query = image("q", 0).embedding.unwrap();
        let a = text_record("a", "a.txt", "one", 0);
        let b = text_record("b", "b.txt", "two", 1);
        let c = text_record("c", "c.txt", "three", 1);
        let hits = fuse_candidates(
            vec![(a.clone(), 0.0), (b.clone(), 1.0)],
            vec![(b.clone(), 8.0), (c.clone(), 4.0)],
            &query,
            10,
        )
        .unwrap();
        assert_eq!(
            hits.iter()
                .map(|hit| hit.row.asset_id.as_str())
                .collect::<Vec<_>>(),
            ["b", "a", "c"]
        );
        assert!((hits[0].hybrid_score.unwrap() - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
        assert_eq!(hits[0].distance, 1.0); // Ranking must not become cosine sorting.
        assert_eq!(hits[2].distance, 1.0); // Lexical-only hit carries actual cosine.
        let tied =
            fuse_candidates(vec![(c.clone(), 0.0)], vec![(a.clone(), 7.0)], &query, 10).unwrap();
        assert_eq!(tied[0].row.asset_id, "a");
        assert_eq!(tied[0].hybrid_score, tied[1].hybrid_score);
        assert_eq!(hybrid_candidate_limit(1), 100);
        assert_eq!(hybrid_candidate_limit(100), 400);
        assert_eq!(hybrid_candidate_limit(1000), 1000);
        assert!(fuse_candidates(vec![(a.clone(), f32::NAN)], vec![], &query, 10).is_err());
        assert!(fuse_candidates(vec![], vec![(a.clone(), f32::INFINITY)], &query, 10).is_err());
        assert!(
            fuse_candidates(vec![(a.clone(), 0.0), (a.clone(), 1.0)], vec![], &query, 10).is_err()
        );
        assert!(fuse_candidates(vec![], vec![(a.clone(), 1.0), (a, 1.0)], &query, 10).is_err());
    }
    #[test]
    fn disabled_dictionary_tokenizers_error_instead_of_substituting_icu() {
        for name in ["jieba/default", "lindera/ipadic"] {
            let config = text_index_config().base_tokenizer(name.into());
            assert!(
                lancedb::tokenize("北京 東京", &config).is_err(),
                "Disabled tokenizer {name} must fail instead of changing tokenization"
            );
        }
    }

    #[test]
    fn lexical_queries_use_literal_multilingual_terms_and_keep_long_identifiers() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let long = "x".repeat(128);
        let cases = [
            ("accent", "Café résumé", "cafe".to_owned()),
            ("cjk", "北京欢迎你", "北京".to_owned()),
            ("japanese", "東京大学", "東京".to_owned()),
            ("long", long.as_str(), long.clone()),
            ("stop", "the AND OR", "AND".to_owned()),
            ("apostrophe", "O'Reilly", "\"O'Reilly\"".to_owned()),
            ("identifier", "Ticket ZXQ-98127", "ZXQ-98127".to_owned()),
        ];
        for (id, passage, query) in &cases {
            store
                .upsert(&text_record(id, "note.txt", passage, 0))
                .unwrap();
            assert!(
                lexical(&store, query, None).contains(&id.to_string()),
                "{id}: {query}"
            );
        }
        for query in ["!!!", "---", "\"\"", "()"] {
            assert!(lexical(&store, query, None).is_empty(), "{query}");
            let hits = store
                .search_filtered(
                    query,
                    &image("q", 0).embedding.unwrap(),
                    10,
                    None,
                    Some(MediaKind::Text),
                )
                .unwrap();
            assert_eq!(hits.len(), cases.len());
            assert!(hits.iter().all(|hit| hit.hybrid_score.is_some()));
        }
        // These are OR-token queries, including literal Boolean-looking words.
        assert!(lexical(&store, "absent OR AND", None).contains(&"stop".into()));
        assert!(lexical(&store, "'; DROP TABLE assets; --", None).is_empty());
        assert_eq!(store.status().unwrap().0, cases.len());
        for query in ["", "  ", "bad\0query"] {
            assert!(store
                .search_filtered(
                    query,
                    &image("q", 0).embedding.unwrap(),
                    10,
                    None,
                    Some(MediaKind::Text)
                )
                .is_err());
        }
        assert!(store
            .search_filtered(
                &"é".repeat(2049),
                &image("q", 0).embedding.unwrap(),
                10,
                None,
                Some(MediaKind::Text)
            )
            .is_err());
    }
    #[test]
    fn lexical_contract_rejects_missing_wrong_or_incompatible_indexes_without_repair() {
        for (name, column, config) in [
            (None, "indexed_text", text_index_config()),
            (Some("wrong_name"), "indexed_text", text_index_config()),
            (Some(TEXT_INDEX_NAME), "filename", text_index_config()),
            (
                Some(TEXT_INDEX_NAME),
                "indexed_text",
                text_index_config().stem(true),
            ),
        ] {
            let root = tempfile::tempdir().unwrap();
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async {
                let db = lancedb::connect(root.path().join("catalog.lancedb").to_str().unwrap())
                    .execute()
                    .await
                    .unwrap();
                let table = db
                    .create_empty_table("assets", schema())
                    .execute()
                    .await
                    .unwrap();
                if let Some(name) = name {
                    table
                        .create_index(&[column], Index::FTS(config))
                        .name(name.into())
                        .execute()
                        .await
                        .unwrap();
                }
            });
            let before = contents(root.path());
            for read_only in [false, true] {
                assert!(Store::open(root.path(), read_only)
                    .unwrap_err()
                    .to_string()
                    .contains("reindex"));
                assert_eq!(contents(root.path()), before);
            }
        }
    }
    #[test]
    fn complete_passages_are_required_on_write_but_not_metadata_projection() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let valid = text_record("valid", "note.txt", "full passage", 0);
        let mut invalid = valid.clone();
        invalid.indexed_text = None;
        assert!(store.upsert(&invalid).is_err());
        invalid.indexed_text = Some("note.txt\nwrong passage".into());
        assert!(store.upsert(&invalid).is_err());
        invalid.indexed_text = Some(format!("note.txt\n{}", "x".repeat(4097)));
        assert!(store.upsert(&invalid).is_err());
        let mut nontext = image("image", 0);
        nontext.indexed_text = Some("secret".into());
        assert!(store.upsert(&nontext).is_err());
        store.upsert(&valid).unwrap();
        assert!(store.asset("valid", false).unwrap().indexed_text.is_none());
        let unicode = format!("{}boundarybeacon", "猫".repeat(900));
        let unicode_row = text_record("unicode", "unicode.txt", &unicode, 0);
        store.upsert(&unicode_row).unwrap();
        assert_eq!(
            store
                .asset("unicode", false)
                .unwrap()
                .text_excerpt
                .unwrap()
                .len(),
            2046
        );
        assert_eq!(lexical(&store, "boundarybeacon", None), ["unicode"]);
        store.update_for_test(
            "valid",
            "text_excerpt",
            &"x".repeat(MAX_TEXT_EXCERPT_BYTES + 1),
        );
        assert!(store
            .search_filtered(
                "passage",
                &valid.embedding.unwrap(),
                10,
                None,
                Some(MediaKind::Text)
            )
            .is_err());
    }
    #[test]
    fn winning_snippet_uses_the_stored_tail_and_leaves_browse_unchanged() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let passage = format!(
            "{} Café tailneedle 北京欢迎你 {}",
            "background ".repeat(250),
            "after ".repeat(60)
        );
        let row = text_record("snippet", "filenamebeacon.txt", &passage, 0);
        store.upsert(&row).unwrap();
        let before = contents(root.path());
        for query in ["tailneedle", "cafe", "北京"] {
            let hits = store
                .search_filtered(
                    query,
                    row.embedding.as_ref().unwrap(),
                    1,
                    None,
                    Some(MediaKind::Text),
                )
                .unwrap();
            let hit = &hits[0].row;
            let excerpt = hit.text_excerpt.as_ref().unwrap();
            assert!(excerpt.len() <= MAX_TEXT_EXCERPT_BYTES);
            assert!(excerpt.contains("tailneedle"), "{query}: {excerpt}");
            assert!(hit.indexed_text.is_none());
            assert!(hit.embedding.is_none());
            assert!(!excerpt.contains("filenamebeacon"));
            assert!(!excerpt.contains("/test/"));
        }
        for query in ["filenamebeacon", "semantic-only-no-term", "!!!"] {
            let hits = store
                .search_filtered(
                    query,
                    row.embedding.as_ref().unwrap(),
                    1,
                    None,
                    Some(MediaKind::Text),
                )
                .unwrap();
            assert_eq!(
                hits[0].row.text_excerpt.as_deref(),
                Some(text_excerpt(&passage))
            );
        }
        assert_eq!(
            store
                .browse_filtered(0, 1, None, Some(MediaKind::Text))
                .unwrap()
                .1[0]
                .text_excerpt,
            row.text_excerpt
        );
        assert_eq!(contents(root.path()), before);
    }
    #[test]
    fn snippet_analyzer_matches_unicode_case_folding_and_literal_terms() {
        for (term, query) in [
            ("Café", "cafe"),
            ("北京", "北京"),
            ("O'Reilly", "o'reilly"),
            ("ZXQ-98127", "zxq-98127"),
        ] {
            let passage = format!("{} {term} {}", "猫 ".repeat(600), "end ".repeat(100));
            let terms = lancedb::tokenize(query, &text_index_config())
                .unwrap()
                .into_iter()
                .map(|token| token.text)
                .collect();
            let snippet = match_centered_excerpt(&passage, &terms).unwrap();
            assert!(snippet.contains(term), "{term}: {snippet}");
            assert!(snippet.len() <= MAX_TEXT_EXCERPT_BYTES);
            assert!(passage.contains(&snippet));
        }
        let passage = format!("{} after", "é".repeat(1500));
        let terms = HashSet::from(["missing".to_owned()]);
        assert_eq!(
            match_centered_excerpt(&passage, &terms).unwrap(),
            text_excerpt(&passage)
        );
    }
    #[test]
    fn document_fusion_collapses_each_leg_before_assigning_ranks_without_length_bonus() {
        let query = image("q", 0).embedding.unwrap();
        let mut a = text_record("a0", "a.txt", "first", 0);
        a.path = "/a.txt".into();
        let mut a1 = a.clone();
        a1.asset_id = "a1".into();
        a1.segment_index = 1;
        let b = text_record("b", "b.txt", "second", 1);
        let base = fuse_document_candidates(
            vec![(a.clone(), 0.0), (b.clone(), 1.0)],
            vec![(b.clone(), 8.0), (a.clone(), 4.0)],
            &query,
            10,
        )
        .unwrap();
        let crowded = fuse_document_candidates(
            vec![(a.clone(), 0.0), (a1.clone(), 0.5), (b.clone(), 1.0)],
            vec![(b.clone(), 8.0), (a1.clone(), 5.0), (a.clone(), 4.0)],
            &query,
            10,
        )
        .unwrap();
        let scores = |hits: &[SearchHit]| {
            hits.iter()
                .map(|hit| (hit.row.path.clone(), hit.hybrid_score.unwrap()))
                .collect::<HashMap<_, _>>()
        };
        assert_eq!(scores(&base), scores(&crowded));
        assert!(base
            .iter()
            .all(|hit| (hit.hybrid_score.unwrap() - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12));
        let a_group = crowded.iter().find(|hit| hit.row.path == "/a.txt").unwrap();
        assert_eq!(a_group.additional_matches.len(), 1);
        assert!(a_group.additional_matches[0].additional_matches.is_empty());
        let mut other_source = a.clone();
        other_source.source_id = 43;
        other_source.asset_id = "other-source".into();
        assert_eq!(
            fuse_document_candidates(vec![(a, 0.0), (other_source, 0.0)], vec![], &query, 10)
                .unwrap()
                .len(),
            2
        );
    }
    #[test]
    fn grouped_search_widens_bounded_candidates_and_returns_expandable_passages() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let rows = (0..120)
            .map(|index| {
                let mut row = text_record(
                    &format!("crowded-{index:03}"),
                    "long.txt",
                    "sharedbeacon",
                    0,
                );
                row.path = "/long.txt".into();
                row.segment_index = index;
                row
            })
            .collect::<Vec<_>>();
        store.replace_file(42, "/long.txt", &rows).unwrap();
        let other = text_record("other", "other.txt", "sharedbeacon several other words", 1);
        store.upsert(&other).unwrap();
        let query = image("q", 0).embedding.unwrap();
        let hits = store
            .search_filtered_grouped("sharedbeacon", &query, 2, None, Some(MediaKind::Text), true)
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_ne!(hits[0].row.path, hits[1].row.path);
        let long = hits.iter().find(|hit| hit.row.path == "/long.txt").unwrap();
        assert_eq!(long.additional_matches.len(), MAX_ADDITIONAL_MATCHES);
        assert!((long.hybrid_score.unwrap() - 2.0 / 61.0).abs() < 1e-12);
        assert!(long
            .additional_matches
            .iter()
            .all(|hit| hit.row.indexed_text.is_none() && hit.row.embedding.is_none()));
        assert!(store
            .search_filtered_grouped(
                "sharedbeacon",
                &query,
                2,
                None,
                Some(MediaKind::Audio),
                true
            )
            .is_err());
        let passages = store
            .search_filtered("sharedbeacon", &query, 2, None, Some(MediaKind::Text))
            .unwrap();
        assert!(passages.iter().all(|hit| hit.additional_matches.is_empty()));
    }
    #[test]
    fn source_manifest_pages_one_snapshot_without_vectors_or_private_passages() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        for (file, count) in [("first", 1024), ("second", 9)] {
            let rows = (0..count)
                .map(|index| {
                    let mut row = text_record(
                        &format!("{file}-{index:04}"),
                        "note.txt",
                        "private passage",
                        0,
                    );
                    row.path = format!("/{file}.txt");
                    row.segment_index = index;
                    row
                })
                .collect::<Vec<_>>();
            store.replace_file(42, &rows[0].path, &rows).unwrap();
        }
        let mut other = text_record("other-source", "other.txt", "separate", 0);
        other.source_id = 43;
        store.upsert(&other).unwrap();
        let manifest = store.source_manifest(42).unwrap();
        assert_eq!(manifest.len(), 1033);
        assert!(manifest
            .windows(2)
            .all(|pair| pair[0].asset_id < pair[1].asset_id));
        assert!(manifest.iter().all(|row| row.source_id == 42
            && row.embedding.is_none()
            && row.indexed_text.is_none()
            && row.text_excerpt.is_some()));
        assert_eq!(store.source_manifest(43).unwrap().len(), 1);
        assert!(store.source_manifest(44).unwrap().is_empty());
    }
    #[test]
    fn text_evaluation_ablations_use_native_legs_and_bm25_needs_no_vector() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        store
            .upsert(&text_record("lexical", "lexical.txt", "exactneedle", 1))
            .unwrap();
        store
            .upsert(&text_record(
                "semantic",
                "semantic.txt",
                "different words",
                0,
            ))
            .unwrap();
        let query = image("q", 0).embedding.unwrap();
        let lexical = store
            .search_text_evaluation("exactneedle", &[], 10, None, TextSearchMode::Bm25)
            .unwrap();
        assert_eq!(lexical.len(), 1);
        assert_eq!(lexical[0].row.asset_id, "lexical");
        assert!(lexical[0].hybrid_score.is_none());
        let dense = store
            .search_text_evaluation("exactneedle", &query, 10, None, TextSearchMode::Dense)
            .unwrap();
        assert_eq!(dense[0].row.asset_id, "semantic");
        let hybrid = store
            .search_text_evaluation("exactneedle", &query, 10, None, TextSearchMode::Hybrid)
            .unwrap();
        assert_eq!(hybrid[0].row.asset_id, "lexical");
        assert!(hybrid[0].hybrid_score.is_some());
    }
    #[test]
    fn text_maintenance_thresholds_bound_repeated_small_edit_rebuilds() {
        let start = std::time::Instant::now();
        let now = 1_000_000;
        let mut state = TextMaintenance {
            format: 1,
            last_indexed_at: now,
            ..Default::default()
        };
        let mut thresholded = 0;
        for _ in 0..100 {
            state.data_generation += 1;
            state.changed_rows += 50;
            if state.due(10_000, now) {
                thresholded += 1;
                state.indexed_generation = state.data_generation;
                state.changed_rows = 0;
            }
        }
        // The old immediate policy rebuilt 100 times for these 100 scans;
        // this deterministic policy simulation rebuilds five times instead.
        assert_eq!(thresholded, 5);
        assert_eq!(state.data_generation, state.indexed_generation);
        state.data_generation += 1;
        state.changed_rows = 1;
        assert!(!state.due(10_000, now));
        assert!(state.due(10_000, now + MAX_INDEX_AGE_SECONDS));
        assert!(state.due(SMALL_TEXT_CORPUS, now));
        state.recovery_due = true;
        assert!(state.due(10_000, now));
        eprintln!("maintenance policy simulation: 100 small-edit scans; immediate=100 rebuilds, thresholded={thresholded}; policy-only elapsed={:?} (not an index-build benchmark)", start.elapsed());
    }
    /// Real Lance/FTS work on synthetic vectors, with no model downloads. Run
    /// explicitly with --ignored --nocapture; this is not a relevance benchmark.
    #[test]
    #[ignore = "synthetic native maintenance benchmark; run explicitly"]
    fn benchmark_small_edit_maintenance_with_readers_and_restart() {
        let corpus_rows = std::env::var("FRACTAL_STORE_BENCH_ROWS")
            .ok()
            .map(|value| value.parse::<usize>().expect("integer benchmark row count"))
            .unwrap_or(5000);
        let edits = std::env::var("FRACTAL_STORE_BENCH_EDITS")
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .expect("integer benchmark edit count")
            })
            .unwrap_or(20);
        assert!((4097..=50_000).contains(&corpus_rows));
        assert!((1..=100).contains(&edits));
        for immediate in [true, false] {
            let root = tempfile::tempdir().unwrap();
            let writer = Store::open(root.path(), false).unwrap();
            let rows = (0..corpus_rows)
                .map(|index| {
                    text_record(
                        &format!("bench-{index:06}"),
                        "note.txt",
                        if index == 0 {
                            "baselineonly original passage"
                        } else {
                            "background synthetic searchable passage"
                        },
                        index % 2,
                    )
                })
                .collect::<Vec<_>>();
            // A single fixture transaction avoids benchmarking fixture creation
            // rather than the one-file edits under comparison.
            let batches = rows.iter().map(batch).collect::<Result<Vec<_>>>().unwrap();
            writer.runtime.block_on(async {
                let table = writer.table().await.unwrap();
                let mut state = writer.text_maintenance.lock().unwrap();
                writer.mark_text_dirty(&mut state, rows.len()).unwrap();
                let mut merge = table.merge_insert(&["asset_id"]);
                merge.when_not_matched_insert_all();
                merge
                    .execute(Box::new(RecordBatchIterator::new(
                        batches.into_iter().map(Ok),
                        schema(),
                    )))
                    .await
                    .unwrap();
            });
            writer.checkpoint().unwrap();
            let reader = Store::open(root.path(), true).unwrap();
            let pinned = reader.runtime.block_on(reader.snapshot()).unwrap();
            let initial_index = writer.runtime.block_on(async {
                Store::text_index_uuid(&writer.table().await.unwrap())
                    .await
                    .unwrap()
            });
            let mut last_index = initial_index;
            let mut rebuilds = 0;
            let mut update_ms = Vec::new();
            let mut read_ms = Vec::new();
            for edit in 0..edits {
                let mut changed = rows[0].clone();
                let passage = format!("latestbeacon version{edit} changed searchable passage");
                changed.text_excerpt = Some(passage.clone());
                changed.indexed_text = Some(format!("note.txt\n{passage}"));
                changed.modified = edit.to_string();
                let start = std::time::Instant::now();
                writer
                    .replace_file(changed.source_id, &changed.path, &[changed.clone()])
                    .unwrap();
                if immediate {
                    writer.text_maintenance.lock().unwrap().recovery_due = true;
                }
                writer.checkpoint().unwrap();
                update_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                let index = writer.runtime.block_on(async {
                    Store::text_index_uuid(&writer.table().await.unwrap())
                        .await
                        .unwrap()
                });
                rebuilds += usize::from(index != last_index);
                last_index = index;
                let start = std::time::Instant::now();
                assert_eq!(lexical(&reader, "latestbeacon", None), ["bench-000000"]);
                assert!(lexical(&reader, "baselineonly", None).is_empty());
                read_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                // A concurrent historical reader must retain its original rows
                // and index even after replacement/maintenance commits.
                reader.runtime.block_on(async {
                    assert_eq!(
                        Store::lexical_candidates(
                            &pinned,
                            "baselineonly",
                            10,
                            &filtered_assets(None, Some(MediaKind::Text))
                        )
                        .await
                        .unwrap()
                        .len(),
                        1
                    );
                });
            }
            update_ms.sort_by(f64::total_cmp);
            read_ms.sort_by(f64::total_cmp);
            let percentile = |samples: &[f64], percent: usize| {
                samples[(samples.len() * percent).div_ceil(100).saturating_sub(1)]
            };
            let state = writer.text_maintenance.lock().unwrap().clone();
            let (fragments, indexed_rows, unindexed_rows) = writer.runtime.block_on(async {
                let table = writer.table().await.unwrap();
                let stats = table.stats().await.unwrap();
                let index = table.index_stats(TEXT_INDEX_NAME).await.unwrap().unwrap();
                (
                    stats.fragment_stats.num_fragments,
                    index.num_indexed_rows,
                    index.num_unindexed_rows,
                )
            });
            let disk_bytes = walkdir::WalkDir::new(root.path())
                .into_iter()
                .map(|entry| entry.unwrap())
                .filter(|entry| entry.file_type().is_file())
                .map(|entry| entry.metadata().unwrap().len())
                .sum::<u64>();
            // The sidecar is already durable before drop, as with a crash after
            // a row commit. No new text is needed for reopened maintenance.
            drop(writer);
            let start = std::time::Instant::now();
            let reopened = Store::open(root.path(), false).unwrap();
            reopened.checkpoint().unwrap();
            let restart_recovery_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert!(!reopened.text_maintenance.lock().unwrap().dirty());
            assert_eq!(lexical(&reader, "latestbeacon", None), ["bench-000000"]);
            if immediate {
                assert_eq!(rebuilds, edits);
            } else {
                assert_eq!(rebuilds, 0);
            }
            let peak_rss = std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|status| {
                    status
                        .lines()
                        .find(|line| line.starts_with("VmHWM:"))
                        .map(str::to_owned)
                });
            eprintln!(
                "{}",
                serde_json::json!({
                    "benchmark": "native-synthetic-small-file-edits",
                    "policy": if immediate { "immediate-full-replacement" } else { "thresholded-full-replacement" },
                    "corpusRows": corpus_rows, "edits": edits, "rebuildsDuringEdits": rebuilds,
                    "dataGeneration": state.data_generation, "indexedGeneration": state.indexed_generation,
                    "changedRowsPending": state.changed_rows,
                    "updateAndCheckpointP50Ms": percentile(&update_ms, 50),
                    "updateAndCheckpointP95Ms": percentile(&update_ms, 95),
                    "twoCurrentReaderQueriesP50Ms": percentile(&read_ms, 50),
                    "twoCurrentReaderQueriesP95Ms": percentile(&read_ms, 95),
                    "fragments": fragments, "indexedRows": indexed_rows, "unindexedRows": unindexed_rows,
                    "retainedDiskBytesBeforeRecovery": disk_bytes,
                    "restartRecoveryMs": restart_recovery_ms,
                    "processLifetimePeakRss": peak_rss,
                    "notes": "Synthetic vectors and short passages; build budget is not an RSS cap; process peak is cumulative across both arms; historical-reader assertions included; initialization excluded"
                })
            );
        }
    }
    #[test]
    fn maintenance_generation_intent_is_persisted_before_commit_and_never_read_only_written() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        store
            .upsert(&text_record("text", "note.txt", "durable", 0))
            .unwrap();
        let path = store.path.join(TEXT_MAINTENANCE_STATE);
        let dirty: TextMaintenance =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(dirty.dirty());
        assert_eq!(dirty.data_generation, 1);
        assert_eq!(dirty.indexed_generation, 0);
        drop(store);
        let reader = Store::open(root.path(), true).unwrap();
        let before = contents(root.path());
        reader.checkpoint().unwrap();
        assert_eq!(contents(root.path()), before);
        let writer = Store::open(root.path(), false).unwrap();
        assert!(writer.text_maintenance.lock().unwrap().recovery_due);
        writer.checkpoint().unwrap();
        let clean: TextMaintenance =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(!clean.dirty());
        assert_eq!(clean.indexed_generation, 1);
        // Intent without a subsequent row commit is safe and recoverable too.
        {
            let mut state = writer.text_maintenance.lock().unwrap();
            writer.mark_text_dirty(&mut state, 1).unwrap();
        }
        drop(writer);
        let writer = Store::open(root.path(), false).unwrap();
        writer.checkpoint().unwrap();
        assert!(!writer.text_maintenance.lock().unwrap().dirty());
        assert_eq!(writer.status().unwrap().0, 1);
    }
    #[test]
    fn schema_four_without_maintenance_sidecar_remains_readable_and_recovers_once() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        store
            .upsert(&text_record(
                "old-schema-four",
                "note.txt",
                "existingbeacon",
                0,
            ))
            .unwrap();
        let path = store.path.join(TEXT_MAINTENANCE_STATE);
        drop(store);
        std::fs::remove_file(&path).unwrap();
        let before = contents(root.path());
        let reader = Store::open(root.path(), true).unwrap();
        assert_eq!(
            lexical(&reader, "existingbeacon", None),
            ["old-schema-four"]
        );
        assert_eq!(contents(root.path()), before);
        let writer = Store::open(root.path(), false).unwrap();
        assert!(writer.text_maintenance.lock().unwrap().recovery_due);
        writer.checkpoint().unwrap();
        assert!(!writer.text_maintenance.lock().unwrap().dirty());
        assert_eq!(
            writer
                .runtime
                .block_on(async { writer.table().await.unwrap().schema().await.unwrap() })
                .as_ref(),
            schema().as_ref()
        );
    }
    fn image(id: &str, axis: usize) -> Record {
        let mut embedding = vec![0.; EMBEDDING_DIM];
        embedding[axis] = 1.;
        Record {
            kind: "asset".into(),
            media_kind: "image".into(),
            asset_id: id.into(),
            source_id: 42,
            path: format!("/test/{id}"),
            filename: id.into(),
            width: 2,
            height: 1,
            bytes: 3,
            modified: "1".into(),
            thumbnail: Some("test.webp".into()),
            embedding: Some(embedding),
            ..Record::default()
        }
    }
    fn contents(root: &Path) -> Vec<(PathBuf, Option<Vec<u8>>)> {
        let mut result = walkdir::WalkDir::new(root)
            .into_iter()
            .map(|entry| entry.unwrap())
            .map(|entry| {
                (
                    entry.path().strip_prefix(root).unwrap().to_owned(),
                    entry
                        .file_type()
                        .is_file()
                        .then(|| std::fs::read(entry.path()).unwrap()),
                )
            })
            .collect::<Vec<_>>();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
    #[test]
    fn snapshot_keeps_metadata_and_vectors_together_and_readers_never_write() {
        let root = tempfile::tempdir().unwrap();
        let writer = Store::open(root.path(), false).unwrap();
        let id = "asset_0ce0d191-86a6-4036-8119-5fabf872780b";
        let original = image(id, 0);
        writer.upsert(&original).unwrap();
        let reader = Store::open(root.path(), true).unwrap();
        let snapshot = reader.runtime.block_on(reader.snapshot()).unwrap();
        let mut changed = image(id, 1);
        changed.filename = "changed.png".into();
        writer.upsert(&changed).unwrap();
        let pinned = reader
            .runtime
            .block_on(Store::rows(&snapshot, "kind = 'asset'", 0, 10, false, true))
            .unwrap();
        assert_eq!(pinned[0].filename, id);
        assert_eq!(pinned[0].embedding, original.embedding);
        assert_eq!(reader.asset(id, true).unwrap().embedding, changed.embedding);
        let before = contents(root.path());
        assert!(reader.upsert(&original).is_err());
        assert!(reader.delete("true").is_err());
        assert_eq!(reader.browse(0, 10, None).unwrap().0, 1);
        assert_eq!(reader.status().unwrap().0, 1);
        assert!(
            reader
                .search(changed.embedding.as_ref().unwrap(), 10, None)
                .unwrap()[0]
                .1
                < 0.0001
        );
        assert_eq!(contents(root.path()), before);
        changed.embedding = Some(vec![f32::NAN; EMBEDDING_DIM]);
        assert!(writer.upsert(&changed).is_err());
        assert_eq!(contents(root.path()), before);
        writer.delete("source_id = 42").unwrap();
        assert_eq!(reader.browse(0, 10, None).unwrap().0, 0);
        assert_eq!(
            reader
                .runtime
                .block_on(Store::rows(
                    &snapshot,
                    "kind = 'asset'",
                    0,
                    10,
                    false,
                    false
                ))
                .unwrap()
                .len(),
            1
        );
    }
    #[test]
    fn writer_recreates_removed_or_empty_catalog_on_reopen() {
        for keep_directory in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("catalog.lancedb");
            let writer = Store::open(root.path(), false).unwrap();
            writer.upsert(&image("old", 0)).unwrap();
            drop(writer);
            std::fs::remove_dir_all(&path).unwrap();
            if keep_directory {
                std::fs::create_dir(&path).unwrap();
            }
            let before = contents(root.path());
            assert!(Store::open(root.path(), true).is_err());
            assert_eq!(contents(root.path()), before);

            let writer = Store::open(root.path(), false).unwrap();
            let (images, files, sources) = writer.status().unwrap();
            assert_eq!(files, 0);
            assert_eq!(images, 0);
            assert!(sources.is_empty());
            assert!(writer.browse(0, 10, None).unwrap().1.is_empty());
            assert!(writer.get("asset_id = 'old'", false).unwrap().is_none());
            writer.upsert(&image("new", 1)).unwrap();
            writer.checkpoint().unwrap();
            drop(writer);
            assert_eq!(
                Store::open(root.path(), false).unwrap().status().unwrap().0,
                1
            );
        }
    }
    #[test]
    fn partial_catalogs_and_legacy_databases_fail_without_repair() {
        for entry in ["unrelated-file", "assets.lance"] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("catalog.lancedb");
            std::fs::create_dir(&path).unwrap();
            if entry == "assets.lance" {
                std::fs::create_dir(path.join(entry)).unwrap();
            } else {
                std::fs::write(path.join(entry), "keep").unwrap();
            }
            let before = contents(root.path());
            for read_only in [false, true] {
                assert!(Store::open(root.path(), read_only).is_err());
                assert_eq!(contents(root.path()), before);
            }
        }
        for keep_directory in [false, true] {
            let root = tempfile::tempdir().unwrap();
            if keep_directory {
                std::fs::create_dir(root.path().join("catalog.lancedb")).unwrap();
            }
            std::fs::write(root.path().join("search.db"), "old catalog").unwrap();
            let before = contents(root.path());
            assert!(Store::open(root.path(), false)
                .unwrap_err()
                .to_string()
                .contains("Old development catalog"));
            assert_eq!(contents(root.path()), before);
        }
    }
    #[test]
    fn removed_catalog_is_not_recreated_during_an_active_session() {
        let root = tempfile::tempdir().unwrap();
        let writer = Store::open(root.path(), false).unwrap();
        let reader = Store::open(root.path(), true).unwrap();
        std::fs::remove_dir_all(root.path().join("catalog.lancedb")).unwrap();
        for store in [&writer, &reader] {
            assert!(store
                .status()
                .unwrap_err()
                .to_string()
                .contains("stop and reopen"));
            assert!(!root.path().join("catalog.lancedb").exists());
        }
    }
    #[cfg(unix)]
    #[test]
    fn catalog_symlinks_and_files_fail_without_repair() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("target");
        let path = root.path().join("catalog.lancedb");
        for target_exists in [false, true] {
            if target_exists {
                std::fs::create_dir(&target).unwrap();
            }
            std::os::unix::fs::symlink(&target, &path).unwrap();
            for read_only in [false, true] {
                assert!(Store::open(root.path(), read_only)
                    .unwrap_err()
                    .to_string()
                    .contains("not symlinks"));
            }
            assert!(!target_exists || std::fs::read_dir(&target).unwrap().next().is_none());
            assert!(target_exists || !target.exists());
            std::fs::remove_file(&path).unwrap();
        }
        std::fs::write(&path, "keep").unwrap();
        for read_only in [false, true] {
            assert!(Store::open(root.path(), read_only).is_err());
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep");
        }
    }
    #[test]
    fn missing_and_incompatible_catalogs_fail_without_repair() {
        let root = tempfile::tempdir().unwrap();
        assert!(Store::open(root.path(), true).is_err());
        assert!(!root.path().join("catalog.lancedb").exists());
        let runtime = tokio::runtime::Runtime::new().unwrap();
        for (key, value) in [
            ("fractal.schema", Some("1")),
            ("fractal.schema", Some("2")),
            ("fractal.schema", Some("3")),
            ("fractal.text_search_profile", Some("wrong")),
            ("fractal.text_search_profile", None),
            (
                "fractal.model_profile",
                Some("fgclip2-base-33f3b9d4-dynamic-torch291-short64-v1"),
            ),
            ("fractal.embedding_dim", Some("1024")),
            ("fractal.distance_metric", Some("l2")),
            ("fractal.normalization", Some("none")),
            ("fractal.vectors_per_segment", Some("2")),
            ("fractal.model_profile", None),
            ("fractal.embedding_dim", None),
            ("fractal.distance_metric", None),
            ("fractal.normalization", None),
            ("fractal.vectors_per_segment", None),
        ] {
            let root = tempfile::tempdir().unwrap();
            let mut metadata = schema().metadata().clone();
            if let Some(value) = value {
                metadata.insert(key.into(), value.into());
            } else {
                metadata.remove(key);
            }
            let wrong = schema().as_ref().clone().with_metadata(metadata);
            runtime.block_on(async {
                let db = lancedb::connect(root.path().join("catalog.lancedb").to_str().unwrap())
                    .execute()
                    .await
                    .unwrap();
                db.create_empty_table("assets", Arc::new(wrong))
                    .execute()
                    .await
                    .unwrap();
            });
            let before = contents(root.path());
            for read_only in [false, true] {
                assert!(Store::open(root.path(), read_only)
                    .unwrap_err()
                    .to_string()
                    .contains("profile"));
                assert_eq!(contents(root.path()), before);
            }
        }
    }
    #[test]
    fn invalid_stored_and_query_embeddings_fail_without_mutation() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let original = image("valid", 0);
        store.upsert(&original).unwrap();
        let before = contents(root.path());
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
            assert!(store.search(&vector, 10, None).is_err());
            let mut invalid = original.clone();
            invalid.embedding = Some(vector);
            assert!(store.upsert(&invalid).is_err());
            assert_eq!(
                store.asset("valid", true).unwrap().embedding,
                original.embedding
            );
            assert_eq!(contents(root.path()), before);
        }
    }
    #[test]
    fn quoted_paths_and_empty_sources_round_trip() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let row = Record {
            kind: "source".into(),
            asset_id: "source_test".into(),
            source_id: 7,
            path: "/photos/it's ünicode".into(),
            ..Record::default()
        };
        store.upsert(&row).unwrap();
        assert_eq!(
            store
                .get(&format!("path = {}", quote(&row.path)), false)
                .unwrap()
                .unwrap()
                .source_id,
            7
        );
        assert_eq!(store.status().unwrap().2[0].1, 0);
        let query = image("test", 0).embedding.unwrap();
        assert!(store.search(&query, 10, None).unwrap().is_empty());
    }
    fn segment(id: &str, media_kind: &str, index: u32, axis: usize) -> Record {
        let mut row = image(id, axis);
        row.path = "/private/movie.mp4".into();
        row.filename = "movie.mp4".into();
        row.media_kind = media_kind.into();
        row.segment_index = index;
        row.start_ms = Some(u64::from(index) * 30_000);
        row.end_ms = Some((u64::from(index) + 1) * 30_000);
        if media_kind == "audio" {
            row.width = 0;
            row.height = 0;
            row.thumbnail = None;
        }
        row
    }
    #[test]
    fn file_replacement_is_one_commit_and_keeps_other_files_and_pinned_snapshots() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let rows = vec![
            segment("video-0", "video", 0, 0),
            segment("audio-0", "audio", 0, 1),
            segment("video-1", "video", 1, 0),
            segment("audio-1", "audio", 1, 1),
        ];
        store.replace_file(42, &rows[0].path, &rows).unwrap();
        let unrelated = image("unrelated", 2);
        store.upsert(&unrelated).unwrap();
        let pinned = store.runtime.block_on(store.snapshot()).unwrap();
        let before = store.runtime.block_on(pinned.version()).unwrap();
        let mut changed = rows[..2].to_vec();
        for row in &mut changed {
            row.modified = "2".into();
            row.end_ms = Some(15_000);
        }
        store.replace_file(42, &changed[0].path, &changed).unwrap();
        let current = store.runtime.block_on(store.snapshot()).unwrap();
        assert_eq!(
            store.runtime.block_on(current.version()).unwrap(),
            before + 1
        );
        assert_eq!(store.browse(0, 10, None).unwrap().0, 3);
        let status = store.status().unwrap();
        assert_eq!((status.0, status.1), (3, 2));
        assert!(store.asset("video-1", false).is_err());
        assert!(store.asset("audio-1", false).is_err());
        assert_eq!(store.asset("video-0", false).unwrap().end_ms, Some(15_000));
        assert_eq!(store.asset("audio-0", false).unwrap().thumbnail, None);
        assert_eq!(
            store.asset("unrelated", true).unwrap().embedding,
            unrelated.embedding
        );
        let old = store
            .runtime
            .block_on(Store::rows(&pinned, "kind = 'asset'", 0, 10, false, true))
            .unwrap();
        assert_eq!(old.len(), 5);
        assert_eq!(
            old.iter()
                .find(|row| row.asset_id == "video-0")
                .unwrap()
                .end_ms,
            Some(30_000)
        );
        let reader = Store::open(root.path(), true).unwrap();
        assert!(reader.replace_file(42, &changed[0].path, &changed).is_err());
        let hit = store
            .search(rows[1].embedding.as_ref().unwrap(), 1, None)
            .unwrap();
        assert_eq!(
            (hit[0].0.media_kind.as_str(), hit[0].0.segment_index),
            ("audio", 0)
        );
    }
    #[test]
    fn invalid_multisegment_replacements_leave_every_old_row_untouched() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path(), false).unwrap();
        let rows = vec![
            segment("video-0", "video", 0, 0),
            segment("video-1", "video", 1, 0),
        ];
        store.replace_file(42, &rows[0].path, &rows).unwrap();
        let before = contents(root.path());
        for scenario in 0..6 {
            let mut changed = rows.clone();
            match scenario {
                0 => changed[1].embedding = Some(vec![f32::NAN; EMBEDDING_DIM]),
                1 => changed[1].path = "/other/file.mp4".into(),
                2 => changed[1].asset_id = changed[0].asset_id.clone(),
                3 => changed[1].segment_index = 0,
                4 => changed[1].end_ms = changed[1].start_ms,
                _ => changed[1].thumbnail = None,
            }
            assert!(store.replace_file(42, &rows[0].path, &changed).is_err());
            assert_eq!(contents(root.path()), before);
        }
        assert!(store.replace_file(42, &rows[0].path, &[]).is_err());
        assert_eq!(contents(root.path()), before);
    }
    #[test]
    fn nullable_multimodal_schema_and_text_bounds_are_enforced() {
        let schema = schema();
        for name in [
            "thumbnail",
            "start_ms",
            "end_ms",
            "text_excerpt",
            "indexed_text",
        ] {
            assert!(schema.field_with_name(name).unwrap().is_nullable());
        }
        let mut text = image("text", 2);
        text.media_kind = "text".into();
        text.width = 0;
        text.height = 0;
        text.thumbnail = None;
        text.text_excerpt = Some("A bounded excerpt".into());
        text.indexed_text = Some("text\nA bounded excerpt".into());
        let serialized = batch(&text).unwrap();
        let decoded = record(&serialized, 0).unwrap();
        assert_eq!(decoded.text_excerpt, text.text_excerpt);
        assert!(decoded.thumbnail.is_none());
        text.text_excerpt = Some("x".repeat(MAX_TEXT_EXCERPT_BYTES + 1));
        assert!(batch(&text).is_err());
        text.text_excerpt = Some("bad\0text".into());
        assert!(batch(&text).is_err());
        text.text_excerpt = Some("valid".into());
        text.thumbnail = Some("placeholder.webp".into());
        assert!(batch(&text).is_err());
    }
}
