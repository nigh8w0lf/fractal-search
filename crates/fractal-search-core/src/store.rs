//! One Lance table owns source rows, image metadata and image vectors.
//! Each mutation is one atomic Lance commit; readers use a pinned snapshot.
use crate::{catalog::validate_embedding, EMBEDDING_DIM, MODEL_PROFILE};
use anyhow::{ensure, Context, Result};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::{
    query::{ColumnOrdering, ExecutableQuery, QueryBase, Select},
    DistanceType, Table,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

const MAX_ROWS: usize = 250_000;
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
    pub thumbnail: String,
    pub embedding: Option<Vec<f32>>,
}
#[derive(Debug)]
pub(crate) struct Store {
    path: PathBuf,
    read_only: bool,
    runtime: tokio::runtime::Runtime,
}
pub(crate) fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}
fn schema() -> Arc<Schema> {
    let mut fields = vec![];
    for name in COLUMNS {
        let data_type = match *name {
            "source_id" => DataType::Int64,
            "width" | "height" => DataType::UInt32,
            "bytes" => DataType::UInt64,
            _ => DataType::Utf8,
        };
        fields.push(Field::new(*name, data_type, false));
    }
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
            ("fractal.schema".into(), "1".into()),
            ("fractal.model_profile".into(), MODEL_PROFILE.into()),
        ]),
    ))
}
fn batch(row: &Record) -> Result<RecordBatch> {
    if let Some(vector) = &row.embedding {
        validate_embedding(vector)?;
    }
    ensure!(
        (row.kind == "source" && row.embedding.is_none())
            || (row.kind == "image" && row.embedding.is_some()),
        "Invalid catalog record kind or missing image embedding"
    );
    ensure!(
        row.source_id > 0 && row.source_id <= i64::from(i32::MAX),
        "Invalid catalog source ID"
    );
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
            string(&row.thumbnail),
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
        kind == "source" || kind == "image",
        "Invalid catalog record kind"
    );
    if kind == "image" && batch.column_by_name("embedding").is_some() {
        ensure!(
            embedding.is_some(),
            "Indexed image is missing its embedding"
        );
    }
    Ok(Record {
        kind,
        asset_id: text("asset_id")?,
        source_id: array::<Int64Array>(batch, "source_id")?.value(i),
        path: text("path")?,
        filename: text("filename")?,
        width: array::<UInt32Array>(batch, "width")?.value(i),
        height: array::<UInt32Array>(batch, "height")?.value(i),
        bytes: array::<UInt64Array>(batch, "bytes")?.value(i),
        modified: text("modified")?,
        thumbnail: text("thumbnail")?,
        embedding,
    })
}
impl Store {
    pub fn open(data: &Path, read_only: bool) -> Result<Self> {
        let store = Self {
            path: data.join("catalog.lancedb"),
            read_only,
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
                    "[Catalog] Creating empty catalog at {:?}; add folders in Fractal Search to index images",
                    store.path
                );
                let db = lancedb::connect(store.uri()?)
                    .execute()
                    .await
                    .context("Connect to empty LanceDB catalog")?;
                db.create_empty_table("assets", schema())
                    .execute()
                    .await
                    .context("Create empty LanceDB catalog")?;
            }
            store.table().await?;
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
        let db = lancedb::connect(self.uri()?).execute().await?;
        let table = db
            .open_table("assets")
            .execute()
            .await
            .context("Open LanceDB catalog")?;
        ensure!(
            table.schema().await?.as_ref() == schema().as_ref(),
            "Incompatible catalog schema or model profile; use a fresh data directory and reindex"
        );
        Ok(table)
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
    pub fn get(&self, filter: &str, vectors: bool) -> Result<Option<Record>> {
        let mut rows = self.list(filter, 2, vectors)?;
        ensure!(rows.len() <= 1, "Duplicate catalog identity");
        Ok(rows.pop())
    }
    pub fn asset(&self, id: &str, vectors: bool) -> Result<Record> {
        self.get(
            &format!("kind = 'image' AND asset_id = {}", quote(id)),
            vectors,
        )?
        .context("Image is not in the search catalog")
    }
    pub fn browse(
        &self,
        offset: usize,
        limit: usize,
        source: Option<i64>,
    ) -> Result<(usize, Vec<Record>)> {
        let filter = image_filter(source);
        self.runtime.block_on(async {
            let table = self.snapshot().await?;
            let total = table.count_rows(Some(filter.clone())).await?;
            let rows = Self::rows(&table, &filter, offset, limit, true, false).await?;
            Ok((total, rows))
        })
    }
    pub fn status(&self) -> Result<(usize, Vec<(Record, usize)>)> {
        self.runtime.block_on(async {
            let table = self.snapshot().await?;
            let sources =
                Self::rows(&table, "kind = 'source'", 0, MAX_ROWS + 1, false, false).await?;
            ensure!(sources.len() <= MAX_ROWS, "Too many catalog sources");
            let total = table.count_rows(Some("kind = 'image'".into())).await?;
            let mut counts = Vec::with_capacity(sources.len());
            for source in sources {
                let count = table
                    .count_rows(Some(image_filter(Some(source.source_id))))
                    .await?;
                counts.push((source, count));
            }
            Ok((total, counts))
        })
    }
    pub fn upsert(&self, row: &Record) -> Result<()> {
        ensure!(!self.read_only, "Catalog is read-only");
        let batch = batch(row)?;
        self.runtime.block_on(async {
            let table = self.table().await?;
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
        self.runtime.block_on(async {
            self.table()
                .await?
                .delete(filter)
                .await
                .context("Delete catalog records")?;
            Ok(())
        })
    }
    pub fn search(
        &self,
        vector: &[f32],
        limit: usize,
        source: Option<i64>,
    ) -> Result<Vec<(Record, f32)>> {
        validate_embedding(vector)?;
        ensure!((1..=1000).contains(&limit), "Invalid search limit");
        self.runtime.block_on(async {
            let table = self.snapshot().await?;
            let mut stream = table
                .vector_search(vector)?
                .distance_type(DistanceType::Cosine)
                .bypass_vector_index()
                .only_if(image_filter(source))
                .limit(limit)
                .select(Select::columns(COLUMNS))
                .execute()
                .await?;
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
                ensure!(hits.len() <= limit, "Search exceeded result limit");
            }
            Ok(hits)
        })
    }
    pub fn checkpoint(&self) -> Result<()> {
        // Writes await their Lance manifest commit. There is no separate WAL.
        self.runtime
            .block_on(async { self.table().await.map(|_| ()) })
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
pub(crate) fn image_filter(source: Option<i64>) -> String {
    match source {
        Some(id) => format!("kind = 'image' AND source_id = {id}"),
        None => "kind = 'image'".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image(id: &str, axis: usize) -> Record {
        let mut embedding = vec![0.; EMBEDDING_DIM];
        embedding[axis] = 1.;
        Record {
            kind: "image".into(),
            asset_id: id.into(),
            source_id: 42,
            path: format!("/test/{id}"),
            filename: id.into(),
            width: 2,
            height: 1,
            bytes: 3,
            modified: "1".into(),
            thumbnail: "test.webp".into(),
            embedding: Some(embedding),
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
            .block_on(Store::rows(&snapshot, "kind = 'image'", 0, 10, false, true))
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
                    "kind = 'image'",
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
            let (images, sources) = writer.status().unwrap();
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
        runtime.block_on(async {
            let db = lancedb::connect(root.path().join("catalog.lancedb").to_str().unwrap())
                .execute()
                .await
                .unwrap();
            let mut metadata = schema().metadata().clone();
            metadata.insert("fractal.model_profile".into(), "wrong".into());
            let wrong = schema().as_ref().clone().with_metadata(metadata);
            db.create_empty_table("assets", Arc::new(wrong))
                .execute()
                .await
                .unwrap();
        });
        let before = contents(root.path());
        assert!(Store::open(root.path(), false)
            .unwrap_err()
            .to_string()
            .contains("profile"));
        assert!(Store::open(root.path(), true).is_err());
        assert_eq!(contents(root.path()), before);
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
        assert_eq!(store.status().unwrap().1[0].1, 0);
        let query = image("test", 0).embedding.unwrap();
        assert!(store.search(&query, 10, None).unwrap().is_empty());
    }
}
