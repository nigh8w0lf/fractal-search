//! Real native retrieval adapter for scripts/evaluate-qmd.py, not a parity test.
//! Uses only a temporary synthetic catalog and an existing verified model cache.
#[cfg(all(feature = "catalog", feature = "runtime"))]
mod evaluation {
    use anyhow::{bail, ensure, Context, Result};
    use fractal_search_core::{
        catalog::{Asset, Catalog},
        media,
        runtime::{ExecutionProvider, NativeRuntime},
        Embedder, TextSearchMode, MODEL_PROFILE, TEXT_SEARCH_PROFILE,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::{
        collections::{BTreeSet, HashMap, HashSet},
        fs,
        path::{Component, Path, PathBuf},
        time::Instant,
    };

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Document {
        id: String,
        folder: String,
        filename: String,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Query {
        id: String,
        text: String,
        #[serde(default)]
        folder: Option<String>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        operation: String,
        corpus_root: PathBuf,
        data_dir: PathBuf,
        model_dir: PathBuf,
        documents: Vec<Document>,
        queries: Vec<Query>,
        mode: String,
        provider: String,
        device_id: Option<i32>,
        threads: usize,
        rank_limit: usize,
        repeats: usize,
        warmups: usize,
    }

    #[derive(Serialize)]
    struct Hit {
        document_id: String,
        passage_id: String,
        segment_index: u32,
        // Opaque identity is diagnostic only, never used as an evaluation qrel.
        asset_id: String,
        similarity: Option<f32>,
        hybrid_score: Option<f64>,
    }

    fn component(value: &str) -> Result<()> {
        ensure!(
            !value.is_empty()
                && Path::new(value).components().count() == 1
                && Path::new(value)
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "Fixture folder and filename must be single relative path components"
        );
        Ok(())
    }

    fn document_map(request: &Request) -> Result<HashMap<String, &Document>> {
        let root = request.corpus_root.canonicalize()?;
        let mut paths = HashMap::new();
        let mut ids = HashSet::new();
        ensure!(
            !request.documents.is_empty() && request.documents.len() <= 10_000,
            "Expected 1..=10000 synthetic documents"
        );
        for doc in &request.documents {
            component(&doc.folder)?;
            component(&doc.filename)?;
            ensure!(
                !doc.id.is_empty() && !doc.id.contains('#') && ids.insert(&doc.id),
                "Empty, duplicate, or ambiguous canonical document ID"
            );
            let path = root.join(&doc.folder).join(&doc.filename).canonicalize()?;
            ensure!(path.starts_with(&root), "Fixture escapes the corpus root");
            ensure!(
                media::kind_for_path(&path) == Some(media::MediaKind::Text),
                "Only supported synthetic text fixtures may be evaluated"
            );
            ensure!(
                paths
                    .insert(path.to_str().context("Non-UTF-8 fixture path")?.into(), doc)
                    .is_none(),
                "Duplicate fixture path"
            );
        }
        Ok(paths)
    }

    fn hits(
        assets: Vec<Asset>,
        documents: &HashMap<String, &Document>,
        folder: Option<&str>,
    ) -> Result<Vec<Hit>> {
        let mut seen = HashSet::new();
        assets
            .into_iter()
            .map(|asset| {
                let doc = documents
                    .get(&asset.path)
                    .context("Search returned an unregistered fixture document")?;
                ensure!(
                    folder.is_none_or(|folder| doc.folder == folder),
                    "Folder prefilter leaked a result from a different source"
                );
                ensure!(asset.media_kind == "text", "Non-text evaluation result");
                let passage_id = format!("{}#{}", doc.id, asset.segment_index);
                ensure!(seen.insert(passage_id.clone()), "Duplicate passage result");
                Ok(Hit {
                    document_id: doc.id.clone(),
                    passage_id,
                    segment_index: asset.segment_index,
                    asset_id: asset.asset_id,
                    similarity: asset.similarity,
                    hybrid_score: asset.hybrid_score,
                })
            })
            .collect()
    }

    fn retrieve(
        catalog: &Catalog,
        runtime: &mut NativeRuntime,
        query: &Query,
        source: Option<i64>,
        mode: TextSearchMode,
        limit: usize,
    ) -> Result<(Vec<Asset>, f64)> {
        let start = Instant::now();
        // BM25 never loads or calls the query model. Corpus vectors were indexed
        // using the real model once, not manufactured for the lexical baseline.
        let vector = if matches!(mode, TextSearchMode::Bm25) {
            Vec::new()
        } else {
            runtime.embed_text(&query.text)?
        };
        let results = catalog.search_text_evaluation(&query.text, &vector, limit, source, mode)?;
        Ok((results, start.elapsed().as_secs_f64() * 1000.0))
    }

    pub fn run() -> Result<()> {
        let started = Instant::now();
        let mut args = std::env::args_os().skip(1);
        let path = PathBuf::from(args.next().context("Usage: evaluate_qmd REQUEST.json")?);
        ensure!(
            args.next().is_none(),
            "Expected exactly one request JSON file"
        );
        ensure!(
            fs::metadata(&path)?.len() <= 4 * 1024 * 1024,
            "Request exceeds 4 MiB"
        );
        let request: Request = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            (1..=1000).contains(&request.rank_limit),
            "Invalid rank limit"
        );
        ensure!((1..=100).contains(&request.repeats), "Invalid repeat count");
        ensure!(request.warmups <= 10, "Too many warmup repetitions");
        let documents = document_map(&request)?;
        let provider = ExecutionProvider::from_selection(&request.provider, request.device_id)?;
        let mut runtime =
            NativeRuntime::with_provider(request.model_dir.clone(), request.threads, provider)?;
        let output = if request.operation == "index" {
            ensure!(
                !request.data_dir.exists(),
                "Evaluation index destination must not exist; never use a live catalog"
            );
            let mut catalog = Catalog::open(&request.data_dir)?;
            let folders: BTreeSet<_> = request.documents.iter().map(|d| &d.folder).collect();
            for folder in folders {
                let result = catalog.index(
                    &request.corpus_root.join(folder),
                    &mut runtime,
                    || false,
                    |_| Ok(()),
                )?;
                ensure!(
                    result.failed == 0 && !result.cancelled,
                    "Synthetic indexing failed: {:?}",
                    result.failures
                );
            }
            catalog.checkpoint()?;
            let index_ms = started.elapsed().as_secs_f64() * 1000.0;
            let status = catalog.status()?;
            ensure!(
                status.file_count == documents.len(),
                "Indexed document count mismatch"
            );
            let mut passages = Vec::new();
            for doc in &request.documents {
                let path = request.corpus_root.join(&doc.folder).join(&doc.filename);
                for segment in media::segments_with_token_counter(
                    &path,
                    &media::inspect(&path)?,
                    |title, text| runtime.document_token_count(title, text),
                )? {
                    passages.push(json!({
                        "document_id": doc.id,
                        "passage_id": format!("{}#{}", doc.id, segment.index),
                        "segment_index": segment.index,
                        "start_byte": segment.start_byte,
                        "end_byte": segment.end_byte,
                        "start_line": segment.start_line,
                        "end_line": segment.end_line,
                        "text": segment.text.context("Missing text fixture passage")?,
                    }));
                }
            }
            ensure!(
                passages.len() == status.asset_count,
                "Indexed passage count mismatch"
            );
            json!({"operation":"index", "documents": status.file_count,
                "passage_count": status.asset_count, "passages": passages,
                "index_ms": index_ms})
        } else if request.operation == "search" || request.operation == "cold" {
            ensure!(
                request.data_dir.join("catalog.lancedb").is_dir(),
                "Evaluation catalog missing; run index first"
            );
            ensure!(
                !request.queries.is_empty() && request.queries.len() <= 1000,
                "Expected 1..=1000 queries"
            );
            let mode = match request.mode.as_str() {
                "bm25" => TextSearchMode::Bm25,
                "dense" => TextSearchMode::Dense,
                "hybrid" => TextSearchMode::Hybrid,
                _ => bail!("Expected bm25, dense, or hybrid evaluation mode"),
            };
            let open_started = Instant::now();
            let catalog = Catalog::open(&request.data_dir)?;
            let status = catalog.status()?;
            let catalog_open_ms = open_started.elapsed().as_secs_f64() * 1000.0;
            let mut source_ids = HashMap::new();
            for source in status.sources {
                source_ids.insert(PathBuf::from(source.directory).canonicalize()?, source.id);
            }
            let sources = request
                .queries
                .iter()
                .map(|query| {
                    query
                        .folder
                        .as_ref()
                        .map(|folder| {
                            component(folder)?;
                            let path = request.corpus_root.join(folder).canonicalize()?;
                            source_ids
                                .get(&path)
                                .copied()
                                .context("Unknown query folder")
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            let mut samples = Vec::new();
            let mut rankings = Vec::new();
            if request.operation == "cold" {
                ensure!(
                    request.queries.len() == 1,
                    "Cold runs require one query per fresh process"
                );
                let query = &request.queries[0];
                let (assets, elapsed_ms) =
                    retrieve(&catalog, &mut runtime, query, sources[0], mode, 6)?;
                // Measure before JSON serialization; the Python harness also
                // records process wall time, which includes start/exit and I/O.
                let first_result_ms = started.elapsed().as_secs_f64() * 1000.0;
                samples.push(
                    json!({"query_id":query.id, "repeat":0, "elapsed_ms":elapsed_ms,
                    "first_result_ms":first_result_ms,
                    "hits":hits(assets, &documents, query.folder.as_deref())?}),
                );
            } else {
                for _ in 0..request.warmups {
                    for (query, source) in request.queries.iter().zip(&sources) {
                        retrieve(&catalog, &mut runtime, query, *source, mode, 6)?;
                    }
                }
                for (query, source) in request.queries.iter().zip(&sources) {
                    let (assets, elapsed_ms) = retrieve(
                        &catalog,
                        &mut runtime,
                        query,
                        *source,
                        mode,
                        request.rank_limit,
                    )?;
                    rankings.push(json!({"query_id":query.id, "elapsed_ms":elapsed_ms,
                        "hits":hits(assets, &documents, query.folder.as_deref())?}));
                }
                // Rotate deterministic query order to avoid always timing the
                // same category immediately after the quality-ranking pass.
                for repeat in 0..request.repeats {
                    for offset in 0..request.queries.len() {
                        let i = (offset + repeat) % request.queries.len();
                        let query = &request.queries[i];
                        let (assets, elapsed_ms) =
                            retrieve(&catalog, &mut runtime, query, sources[i], mode, 6)?;
                        samples.push(
                            json!({"query_id":query.id,"repeat":repeat,"elapsed_ms":elapsed_ms,
                            "hits":hits(assets, &documents, query.folder.as_deref())?}),
                        );
                    }
                }
            }
            json!({"operation":request.operation,"mode":request.mode,
                "catalog_open_ms":catalog_open_ms,"rank_limit":request.rank_limit,
                "quality_rankings":rankings,"samples":samples})
        } else {
            bail!("Expected index, search, or cold operation")
        };
        let mut output = output
            .as_object()
            .context("Expected object report")?
            .clone();
        output.insert("model_profile".into(), Value::from(MODEL_PROFILE));
        output.insert(
            "embedding_dimension".into(),
            Value::from(fractal_search_core::EMBEDDING_DIM),
        );
        output.insert(
            "text_search_profile".into(),
            Value::from(TEXT_SEARCH_PROFILE),
        );
        output.insert(
            "extraction_profile".into(),
            Value::from(media::TEXT_EXTRACTION_PROFILE),
        );
        output.insert("provider".into(), Value::from(request.provider));
        output.insert("threads".into(), Value::from(request.threads));
        println!("{}", serde_json::to_string(&output)?);
        Ok(())
    }
}

#[cfg(all(feature = "catalog", feature = "runtime"))]
fn main() -> anyhow::Result<()> {
    evaluation::run()
}

#[cfg(not(all(feature = "catalog", feature = "runtime")))]
fn main() {
    eprintln!("evaluate_qmd requires the catalog and runtime features");
    std::process::exit(2);
}
