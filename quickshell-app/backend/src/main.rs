//! Trusted local desktop transport, separate from the read-only fractal-search-mcp API.
mod daemon;
mod diagnostics;
mod models;
mod watch;

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use fractal_search_core::{
    catalog::{Asset, Catalog, Page},
    runtime::{ExecutionProvider, NativeRuntime},
    Embedder,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufRead, Write},
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::Duration,
};

const MAX_FRAME: usize = 65536;
static STOP: AtomicBool = AtomicBool::new(false);
type Output = Arc<Mutex<std::io::Stdout>>;

#[derive(Parser)]
#[command(
    version,
    about = "Native FG-CLIP2 image search for Quickshell; no face models or Python"
)]
struct Options {
    #[arg(long, env = "FRACTAL_SEARCH_DATA_DIR")]
    data_dir: Option<PathBuf>,
    #[arg(long, env = "FRACTAL_SEARCH_MODEL_DIR")]
    model_dir: Option<PathBuf>,
    #[arg(long, env = "FRACTAL_SEARCH_THEME_FILE")]
    theme_file: Option<PathBuf>,
    #[arg(long, env = "FRACTAL_SEARCH_THREADS")]
    threads: Option<usize>,
    /// Inference backend. CUDA requires a CUDA-enabled bundle and its runtime libraries.
    #[arg(long, env = "FRACTAL_SEARCH_PROVIDER", default_value = "cpu", value_parser = ["cpu", "cuda"])]
    provider: String,
    #[arg(long, env = "FRACTAL_SEARCH_DEVICE_ID")]
    device_id: Option<i32>,
    #[arg(long)]
    socket: Option<PathBuf>,
    /// Prepare text search on startup without running a query or downloading assets.
    #[arg(long)]
    warm_search: bool,
    #[arg(default_value="serve", value_parser=["serve", "status", "daemon", "providers", "diagnostics"])]
    mode: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: u64,
    command: Command,
}
#[derive(Debug, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Command {
    Status {},
    Browse {
        #[serde(default)]
        offset: usize,
        #[serde(default = "page_limit")]
        limit: usize,
        source_id: Option<i64>,
    },
    Search {
        query: String,
        #[serde(default)]
        offset: usize,
        #[serde(default = "page_limit")]
        limit: usize,
        source_id: Option<i64>,
        #[serde(default = "search_limit")]
        max_results: usize,
    },
    Index {
        directory: String,
    },
    Preview {
        asset_id: String,
    },
    OriginalUrl {
        asset_id: String,
    },
    Reveal {
        asset_id: String,
    },
    RemoveSource {
        source_id: i64,
    },
    UpdateSettings {
        settings: UiSettings,
    },
    SetModelDirectory {
        directory: String,
    },
    InstallModels {},
    Theme {},
    Cancel {},
    Shutdown {},
}
fn page_limit() -> usize {
    50
}
fn search_limit() -> usize {
    1000
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ResultLayout {
    ListPreview,
    List,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct UiSettings {
    top_matches: usize,
    result_layout: ResultLayout,
    watch_changes: bool,
    close_after_opening: bool,
}
impl Default for UiSettings {
    fn default() -> Self {
        Self {
            top_matches: 12,
            result_layout: ResultLayout::ListPreview,
            watch_changes: false,
            close_after_opening: true,
        }
    }
}
impl UiSettings {
    fn validate(&self) -> Result<()> {
        ensure!(
            (1..=100).contains(&self.top_matches),
            "Top matches must be between 1 and 100"
        );
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Settings {
    model_directory: PathBuf,
    // Old prototype settings stored only the model path. This is the explicit
    // migration for those files; malformed or partial UI settings still fail.
    #[serde(default)]
    ui: UiSettings,
}

fn emit(output: &Output, value: &Value) -> Result<()> {
    let mut stream = output
        .lock()
        .map_err(|_| anyhow::anyhow!("JSON output mutex poisoned"))?;
    serde_json::to_writer(&mut *stream, value)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}
fn reply(output: &Output, id: Option<u64>, result: Result<Value>) -> Result<()> {
    match result {
        Ok(result) => emit(output, &json!({"id":id,"ok":true,"result":result})),
        Err(error) => {
            eprintln!("[Error] {error:#}");
            emit(
                output,
                &json!({"id":id,"ok":false,"error":format!("{error:#}")}),
            )
        }
    }
}

/// Stop allocating before accepting an oversized line from any stdin client.
fn read_frame(input: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut result = Vec::new();
    loop {
        let available = input.fill_buf()?;
        if available.is_empty() {
            return Ok(if result.is_empty() {
                None
            } else {
                Some(result)
            });
        }
        let count = available
            .iter()
            .position(|&b| b == b'\n')
            .map(|n| n + 1)
            .unwrap_or(available.len());
        ensure!(
            result.len() + count <= MAX_FRAME,
            "JSON request exceeds {MAX_FRAME} bytes"
        );
        let done = available[count - 1] == b'\n';
        result.extend_from_slice(&available[..count]);
        input.consume(count);
        if done {
            return Ok(Some(result));
        }
    }
}
fn local_path(value: &str) -> Result<PathBuf> {
    ensure!(
        !value.trim().is_empty() && value.len() <= 16384,
        "Directory path is empty or too long"
    );
    let path = if value.starts_with("file:") {
        url::Url::parse(value)?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Directory URL must be a local file URL"))?
    } else {
        PathBuf::from(value)
    };
    ensure!(
        path.is_absolute(),
        "Directory must be an absolute local path"
    );
    Ok(path)
}
fn save_settings(path: &Path, settings: &Settings) -> Result<()> {
    let temporary = path.with_extension("json.part");
    let mut file = fs::File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut file, settings)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}
fn theme(path: &Path) -> Result<Value> {
    let size = fs::metadata(path)
        .with_context(|| format!("Read Omarchy theme {path:?}"))?
        .len();
    ensure!(size <= 65536, "Omarchy color theme exceeds 64 KiB");
    let value: toml::Value = toml::from_str(&fs::read_to_string(path)?)?;
    let mut colors = serde_json::Map::new();
    for (name, role) in [
        ("background", "background"),
        ("foreground", "foreground"),
        ("accent", "accent"),
        ("selection", "surface"),
        ("muted", "muted"),
        ("red", "error"),
        ("green", "success"),
    ] {
        let color = value
            .get(name)
            .and_then(toml::Value::as_str)
            .with_context(|| format!("Omarchy theme missing color {name}"))?;
        ensure!(
            color.len() == 7
                && color.starts_with('#')
                && color[1..].bytes().all(|b| b.is_ascii_hexdigit()),
            "Invalid Omarchy color {name}: {color:?}"
        );
        colors.insert(role.into(), json!(color));
    }
    Ok(Value::Object(colors))
}
fn status(
    catalog: &Catalog,
    runtime: &NativeRuntime,
    theme_file: &Path,
    threads: usize,
    settings: &UiSettings,
) -> Result<Value> {
    Ok(
        json!({"protocolVersion":1,"catalog":catalog.status()?,"modelDirectory":runtime.model_dir(),"modelsPresent":runtime.assets_present(),"modelLoaded":runtime.loaded(),"textModelLoaded":runtime.text_loaded(),"textModelWarming":false,"provider":runtime.provider().name(),"deviceId":runtime.provider().device_id(),"threads":threads,"themeFile":theme_file,"settings":settings}),
    )
}

fn reveal(url: &str) -> Result<()> {
    // ShowItems selects the original in the file manager. Running a process
    // directly keeps filenames out of shell parsing and reports missing D-Bus
    // support instead of silently opening an unrelated location.
    let items = serde_json::to_string(&[url])?;
    let result = std::process::Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.freedesktop.FileManager1",
            "--object-path",
            "/org/freedesktop/FileManager1",
            "--method",
            "org.freedesktop.FileManager1.ShowItems",
            "--timeout",
            "5",
            &items,
            "",
        ])
        .output()
        .context(
            "Reveal image requires gdbus and a file manager implementing FileManager1.ShowItems",
        )?;
    ensure!(
        result.status.success(),
        "File manager could not reveal the image: {}",
        String::from_utf8_lossy(&result.stderr).trim()
    );
    Ok(())
}

fn configure_watcher(
    current: &mut Option<watch::DirectoryWatcher>,
    enabled: bool,
    catalog: &Catalog,
    force: bool,
) -> Result<()> {
    if !enabled {
        *current = None;
        return Ok(());
    }
    let roots = catalog
        .status()?
        .sources
        .into_iter()
        .map(|source| (source.id, PathBuf::from(source.directory)))
        .collect::<Vec<_>>();
    if !force
        && current
            .as_ref()
            .is_some_and(|watcher| watcher.roots() == roots)
    {
        return Ok(());
    }
    let replacement = watch::DirectoryWatcher::new(roots)?;
    if let Some(old) = current {
        replacement.inherit_pending(old)?;
    }
    *current = Some(replacement);
    Ok(())
}

unsafe extern "C" fn stop_signal(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}
fn install_signals() -> Result<()> {
    // AtomicBool is lock-free here; the handler allocates nothing and only asks
    // the worker to finish its active image. Model/SQLite cleanup stays on main.
    for signal in [libc::SIGTERM, libc::SIGINT] {
        ensure!(
            unsafe { libc::signal(signal, stop_signal as *const () as libc::sighandler_t) }
                != libc::SIG_ERR,
            "Cannot install shutdown signal handler"
        );
    }
    Ok(())
}
fn catalog_lock(data_dir: &Path) -> Result<fs::File> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(data_dir.join("service.lock"))?;
    ensure!(
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "This search catalog is already open by another service: {data_dir:?}"
    );
    Ok(file)
}

fn serve(options: Options) -> Result<()> {
    let provider = ExecutionProvider::from_selection(&options.provider, options.device_id)?;
    install_signals()?;
    let data_dir = options.data_dir.map(Ok).unwrap_or_else(|| {
        dirs::data_dir()
            .context("Cannot determine search data directory")
            .map(|p| p.join("fractal-search"))
    })?;
    fs::create_dir_all(&data_dir)?;
    let data_dir = data_dir.canonicalize()?;
    let _lock = catalog_lock(&data_dir)?;
    let settings_path = data_dir.join("settings.json");
    let saved: Option<Settings> = match fs::read(&settings_path) {
        Ok(bytes) => Some(serde_json::from_slice(&bytes).context("Invalid search settings.json")?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).context("Read search settings"),
    };
    let mut ui_settings = saved.as_ref().map(|s| s.ui.clone()).unwrap_or_default();
    ui_settings
        .validate()
        .context("Invalid saved UI settings")?;
    let model_dir = match options
        .model_dir
        .or_else(|| saved.as_ref().map(|s| s.model_directory.clone()))
    {
        Some(path) => path,
        None => dirs::cache_dir()
            .context("Cannot determine model cache directory")?
            .join("fractal-search/models/fgclip2-33f3b9d4"),
    };
    let model_dir = if model_dir.is_absolute() {
        model_dir
    } else {
        std::env::current_dir()?.join(model_dir)
    };
    let theme_file = options.theme_file.map(Ok).unwrap_or_else(|| {
        dirs::state_dir()
            .context("Cannot determine Omarchy state directory")
            .map(|p| p.join("omarchy/current/theme/colors.toml"))
    })?;
    let threads = options.threads.unwrap_or(
        std::thread::available_parallelism()?
            .get()
            .div_ceil(2)
            .clamp(1, 6),
    );
    let mut runtime = NativeRuntime::with_provider(model_dir, threads, provider)?;
    let mut catalog = Catalog::open(&data_dir)?;
    fractal_search_core::initialize_image_decoders();
    let output = Arc::new(Mutex::new(std::io::stdout()));
    if options.mode == "status" {
        return emit(
            &output,
            &status(&catalog, &runtime, &theme_file, threads, &ui_settings)?,
        );
    }
    let mut watcher = None;
    configure_watcher(&mut watcher, ui_settings.watch_changes, &catalog, false)?;
    // Expose catalog/setup immediately while the worker prepares search.
    // Missing assets stay in explicit first-run setup; startup never downloads.
    let warm_search = options.warm_search && runtime.assets_present();
    let mut initial_status = status(&catalog, &runtime, &theme_file, threads, &ui_settings)?;
    initial_status["textModelWarming"] = json!(warm_search);
    emit(&output, &json!({"event":"ready","payload":initial_status}))?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let index_admitted = Arc::new(AtomicBool::new(false));
    let shutdown_id = Arc::new(Mutex::new(None));
    let (sender, receiver) = mpsc::sync_channel::<Request>(16);
    let input_output = output.clone();
    let input_cancel = cancelled.clone();
    let input_index = index_admitted.clone();
    let input_shutdown = shutdown_id.clone();
    std::thread::spawn(move || {
        let read = || -> Result<()> {
            let mut input = std::io::stdin().lock();
            while let Some(frame) = read_frame(&mut input)? {
                let request = match serde_json::from_slice::<Request>(&frame) {
                    Ok(request) => request,
                    Err(error) => {
                        reply(&input_output, None, Err(error.into()))?;
                        continue;
                    }
                };
                ensure!(
                    request.id <= 9_007_199_254_740_991,
                    "Request ID exceeds JavaScript's safe integer range"
                );
                match request.command {
                    Command::Shutdown {} => {
                        *input_shutdown
                            .lock()
                            .map_err(|_| anyhow::anyhow!("Shutdown mutex poisoned"))? =
                            Some(request.id);
                        STOP.store(true, Ordering::Relaxed);
                        break;
                    }
                    Command::Cancel {} => {
                        if !input_index.load(Ordering::Relaxed) {
                            reply(
                                &input_output,
                                Some(request.id),
                                Err(anyhow::anyhow!("No indexing job is active")),
                            )?;
                            continue;
                        }
                        input_cancel.store(true, Ordering::Relaxed);
                        reply(
                            &input_output,
                            Some(request.id),
                            Ok(json!({"cancelRequested":true})),
                        )?;
                    }
                    _ => {
                        let is_index = matches!(request.command, Command::Index { .. });
                        if is_index && input_index.swap(true, Ordering::Relaxed) {
                            reply(
                                &input_output,
                                Some(request.id),
                                Err(anyhow::anyhow!("An indexing job is already active")),
                            )?;
                            continue;
                        }
                        match sender.try_send(request) {
                            Ok(()) => (),
                            Err(mpsc::TrySendError::Full(request)) => {
                                if is_index {
                                    input_index.store(false, Ordering::Relaxed);
                                }
                                reply(
                                    &input_output,
                                    Some(request.id),
                                    Err(anyhow::anyhow!("Search request queue is full")),
                                )?;
                            }
                            Err(mpsc::TrySendError::Disconnected(_)) => break,
                        }
                    }
                }
            }
            Ok(())
        };
        if let Err(error) = read() {
            if let Err(output_error) = reply(&input_output, None, Err(error)) {
                eprintln!("[Error] Cannot report stdin failure: {output_error:#}");
            }
            STOP.store(true, Ordering::Relaxed);
        }
    });
    if warm_search && !STOP.load(Ordering::Relaxed) {
        let started = std::time::Instant::now();
        let result = runtime.prepare_text().context("Warm up text search");
        let error = result.err().map(|error| format!("{error:#}"));
        if let Some(error) = &error {
            eprintln!("[Model] Background search warmup failed: {error}");
        } else {
            eprintln!(
                "[Model] Background text-search warmup complete in {:.1} ms",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        emit(
            &output,
            &json!({"event":"warmupFinished","payload":{"ok":error.is_none(),"error":error}}),
        )?;
    }
    let mut search_cache: Option<((String, Option<i64>, usize), Vec<Asset>)> = None;
    while !STOP.load(Ordering::Relaxed) {
        let request = match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(request) => request,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(active_watcher) = &watcher {
                    for error in active_watcher.take_errors()? {
                        emit(
                            &output,
                            &json!({"event":"indexError","payload":{"directory":"","error":error}}),
                        )?;
                    }
                    // The same admission latch protects both manual and watch
                    // indexing. Stdin can still cancel/close during inference.
                    if let Some((source_id, path)) = active_watcher.take_due()? {
                        if index_admitted
                            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok()
                        {
                            search_cache = None;
                            emit(
                                &output,
                                &json!({"event":"indexStarted","payload":{"directory":path}}),
                            )?;
                            let result = catalog.index(&path, &mut runtime,
                                || cancelled.load(Ordering::Relaxed) || STOP.load(Ordering::Relaxed),
                                |progress| emit(&output, &json!({"id":null,"event":"indexProgress","payload":progress})));
                            cancelled.store(false, Ordering::Relaxed);
                            index_admitted.store(false, Ordering::Relaxed);
                            if !STOP.load(Ordering::Relaxed) {
                                // Reattach recursive watches if the selected
                                // root was renamed/deleted and recreated.
                                if let Err(error) = configure_watcher(
                                    &mut watcher,
                                    ui_settings.watch_changes,
                                    &catalog,
                                    true,
                                ) {
                                    eprintln!("[Watch] {error:#}");
                                    emit(
                                        &output,
                                        &json!({"event":"indexError","payload":{"directory":path,"error":format!("{error:#}")}}),
                                    )?;
                                }
                            }
                            match result {
                                Ok(result) => emit(
                                    &output,
                                    &json!({"event":"indexFinished","payload":result}),
                                )?,
                                Err(error) => {
                                    eprintln!("[Watch] {error:#}");
                                    emit(
                                        &output,
                                        &json!({"event":"indexError","payload":{"directory":path,"error":format!("{error:#}")}}),
                                    )?;
                                }
                            }
                        } else {
                            active_watcher.reschedule(source_id)?;
                        }
                    }
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let id = request.id;
        let is_index = matches!(request.command, Command::Index { .. });
        let refresh_watch = is_index || matches!(request.command, Command::RemoveSource { .. });
        let result: Result<Value> = (|| match request.command {
            Command::Status {} => status(&catalog, &runtime, &theme_file, threads, &ui_settings),
            Command::Browse {
                offset,
                limit,
                source_id,
            } => Ok(serde_json::to_value(
                catalog.browse_source(offset, limit, source_id)?,
            )?),
            Command::Search {
                query,
                offset,
                limit,
                source_id,
                max_results,
            } => {
                ensure!(
                    !query.trim().is_empty() && query.len() <= 4096,
                    "Enter a search description between 1 and 4096 UTF-8 bytes"
                );
                ensure!(
                    (1..=100).contains(&limit) && offset <= 1000,
                    "Search page limit must be 1..=100 and offset <=1000"
                );
                ensure!(
                    (1..=1000).contains(&max_results),
                    "Search maxResults must be between 1 and 1000"
                );
                if let Some(id) = source_id {
                    catalog.source_directory(id)?;
                }
                let key = (query.clone(), source_id, max_results);
                if !search_cache.as_ref().is_some_and(|r| r.0 == key) {
                    if !runtime.text_loaded() {
                        emit(
                            &output,
                            &json!({"id":id,"event":"searchProgress","payload":{"stage":"starting"}}),
                        )?;
                        runtime.prepare_text()?;
                    }
                    emit(
                        &output,
                        &json!({"id":id,"event":"searchProgress","payload":{"stage":"searching"}}),
                    )?;
                    let embedding = runtime.embed_text(&query)?;
                    search_cache = Some((
                        key,
                        catalog.search_source(&embedding, max_results, source_id)?,
                    ));
                }
                let hits = &search_cache
                    .as_ref()
                    .context("Search cache not initialized")?
                    .1;
                Ok(serde_json::to_value(Page {
                    items: hits.iter().skip(offset).take(limit).cloned().collect(),
                    total: hits.len(),
                    offset,
                    limit,
                })?)
            }
            Command::Index { directory } => {
                search_cache = None;
                let path = local_path(&directory)?;
                let result = catalog.index(
                    &path,
                    &mut runtime,
                    || cancelled.load(Ordering::Relaxed) || STOP.load(Ordering::Relaxed),
                    |progress| {
                        emit(
                            &output,
                            &json!({"id":id,"event":"indexProgress","payload":progress}),
                        )
                    },
                )?;
                Ok(serde_json::to_value(result)?)
            }
            Command::Preview { asset_id } => Ok(json!({"url":catalog.preview(&asset_id)?})),
            Command::OriginalUrl { asset_id } => {
                Ok(json!({"url":catalog.original_url(&asset_id)?}))
            }
            Command::Reveal { asset_id } => {
                reveal(&catalog.original_url(&asset_id)?)?;
                Ok(json!({"revealed":true}))
            }
            Command::RemoveSource { source_id } => {
                search_cache = None;
                Ok(json!({"removed":catalog.remove_source(source_id)?}))
            }
            Command::UpdateSettings { settings } => {
                settings.validate()?;
                let mut replacement = None;
                configure_watcher(&mut replacement, settings.watch_changes, &catalog, false)?;
                if let (Some(new), Some(old)) = (&replacement, &watcher) {
                    new.inherit_pending(old)?;
                }
                save_settings(
                    &settings_path,
                    &Settings {
                        model_directory: runtime.model_dir().to_path_buf(),
                        ui: settings.clone(),
                    },
                )?;
                ui_settings = settings;
                watcher = replacement;
                status(&catalog, &runtime, &theme_file, threads, &ui_settings)
            }
            Command::SetModelDirectory { directory } => {
                let path = local_path(&directory)?.canonicalize()?;
                let mut replacement =
                    NativeRuntime::with_provider(path.clone(), threads, runtime.provider())?;
                replacement.verify()?;
                save_settings(
                    &settings_path,
                    &Settings {
                        model_directory: path,
                        ui: ui_settings.clone(),
                    },
                )?;
                runtime = replacement;
                search_cache = None;
                status(&catalog, &runtime, &theme_file, threads, &ui_settings)
            }
            Command::InstallModels {} => {
                models::install(
                    runtime.model_dir(),
                    || STOP.load(Ordering::Relaxed),
                    |progress| {
                        emit(
                            &output,
                            &json!({"id":id,"event":"modelProgress","payload":progress}),
                        )
                    },
                )?;
                runtime.verify()?;
                status(&catalog, &runtime, &theme_file, threads, &ui_settings)
            }
            Command::Theme {} => theme(&theme_file),
            Command::Cancel {} | Command::Shutdown {} => {
                bail!("Control command was not handled by stdin admission")
            }
        })();
        // A failed index may leave completed images in the catalog. Clear its
        // cancellation latch as well, so retrying never silently cancels itself.
        if is_index {
            cancelled.store(false, Ordering::Relaxed);
            index_admitted.store(false, Ordering::Relaxed);
        }
        // Indexing can add a source even when traversal later fails; removal
        // can commit partial batches. Keep watching aligned in either case.
        let result = if refresh_watch && !STOP.load(Ordering::Relaxed) {
            match configure_watcher(&mut watcher, ui_settings.watch_changes, &catalog, is_index) {
                Ok(()) => result,
                Err(error) => match result {
                    Ok(_) => Err(error.context("Update indexed-folder watcher")),
                    Err(original) => Err(error.context(format!(
                        "Update indexed-folder watcher after command failed: {original:#}"
                    ))),
                },
            }
        } else {
            result
        };
        reply(&output, Some(id), result)?;
    }
    eprintln!("[Shutdown] Finishing search catalog writes and releasing native models");
    drop(watcher);
    for request in receiver.try_iter() {
        reply(
            &output,
            Some(request.id),
            Err(anyhow::anyhow!("Search service is shutting down")),
        )?;
    }
    catalog.checkpoint()?;
    drop(runtime);
    let id = *shutdown_id
        .lock()
        .map_err(|_| anyhow::anyhow!("Shutdown mutex poisoned"))?;
    if id.is_some() {
        reply(&output, id, Ok(json!({"stopped":true})))?;
    }
    eprintln!("[Shutdown] Search service drained and stopped");
    Ok(())
}

fn run(options: Options) -> Result<()> {
    // Bundle checks must not create/open a catalog or initialize model/GPU state.
    if options.mode == "providers" {
        let providers = if cfg!(feature = "cuda") {
            vec!["cpu", "cuda"]
        } else {
            vec!["cpu"]
        };
        return emit(
            &Arc::new(Mutex::new(std::io::stdout())),
            &json!({"cudaCompiled":cfg!(feature = "cuda"),"providers":providers}),
        );
    }
    if options.mode == "diagnostics" {
        return emit(
            &Arc::new(Mutex::new(std::io::stdout())),
            &serde_json::to_value(diagnostics::collect(&options.provider, options.device_id)?)?,
        );
    }
    ExecutionProvider::from_selection(&options.provider, options.device_id)?;
    if options.mode == "daemon" {
        daemon::run(options)
    } else {
        serve(options)
    }
}

fn main() {
    let result = run(Options::parse());
    if let Err(error) = result {
        eprintln!("[Fatal] {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn provider_cli_accepts_explicit_selection_and_rejects_unknown_providers() {
        let options = Options::try_parse_from([
            "fractal-search",
            "daemon",
            "--provider",
            "cuda",
            "--device-id",
            "2",
        ])
        .unwrap();
        assert_eq!(options.provider, "cuda");
        assert_eq!(options.device_id, Some(2));
        assert!(Options::try_parse_from(["fractal-search", "--provider", "auto"]).is_err());
        assert!(Options::try_parse_from(["fractal-search", "providers"]).is_ok());
    }
    #[test]
    fn transport_rejects_unknown_fields_and_oversized_frames() {
        // Internally tagged unit variants ignore extra fields in Serde. These
        // commands deliberately use empty struct variants to reject them.
        for kind in ["status", "theme", "cancel", "shutdown", "installModels"] {
            let value = json!({"id":1,"command":{"type":kind,"unknown":true}});
            assert!(serde_json::from_value::<Request>(value).is_err());
            let value = json!({"id":1,"command":{"type":kind}});
            assert!(serde_json::from_value::<Request>(value).is_ok());
        }
        assert!(serde_json::from_str::<Request>(
            r#"{"id":1,"command":{"type":"search","query":"red","extra":true}}"#
        )
        .is_err());
        assert!(serde_json::from_str::<Request>(
            r#"{"id":1,"command":{"type":"status"},"extra":true}"#
        )
        .is_err());
        assert!(read_frame(&mut std::io::Cursor::new(vec![b'a'; MAX_FRAME + 1])).is_err());
        let frame = read_frame(&mut std::io::Cursor::new(b"{}\nnext"))
            .unwrap()
            .unwrap();
        assert_eq!(frame, b"{}\n");
        assert_eq!(
            read_frame(&mut std::io::Cursor::new(b"{}"))
                .unwrap()
                .unwrap(),
            b"{}"
        );
    }
    #[test]
    fn local_directory_urls_preserve_special_characters_and_reject_remote_urls() {
        assert_eq!(
            local_path("file:///tmp/red%20%23%20%C3%BC").unwrap(),
            Path::new("/tmp/red # ü")
        );
        assert!(local_path("file://remote/photos").is_err());
        assert!(local_path("relative").is_err());
    }
    #[test]
    fn themes_validate_colors_and_settings_survive_reopen() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("colors.toml");
        fs::write(&path,"background='#111111'\nforeground='#ffffff'\naccent='#abcdef'\nselection='#222222'\nmuted='#333333'\nred='#ff0000'\ngreen='#00ff00'").unwrap();
        assert_eq!(theme(&path).unwrap()["accent"], "#abcdef");
        fs::write(&path, "accent='invalid'").unwrap();
        assert!(theme(&path).is_err());
        let path = root.path().join("settings.json");
        let settings = Settings {
            model_directory: root.path().join("models"),
            ui: UiSettings {
                top_matches: 20,
                result_layout: ResultLayout::List,
                watch_changes: true,
                close_after_opening: false,
            },
        };
        save_settings(&path, &settings).unwrap();
        assert_eq!(
            serde_json::from_slice::<Settings>(&fs::read(&path).unwrap())
                .unwrap()
                .model_directory,
            settings.model_directory
        );
        let reopened: Settings = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(reopened.ui, settings.ui);
        let legacy: Settings = serde_json::from_value(json!({"modelDirectory":"/models"})).unwrap();
        assert_eq!(legacy.ui, UiSettings::default());
        // Only missing whole preferences in the old file migrate. Partial,
        // unknown or invalid preferences must never be silently repaired.
        assert!(serde_json::from_value::<Settings>(
            json!({"modelDirectory":"/models","ui":{"topMatches":12}})
        )
        .is_err());
        assert!(serde_json::from_value::<Request>(json!({"id":1,"command":{"type":"updateSettings","settings":{"topMatches":12,"resultLayout":"grid","watchChanges":false,"closeAfterOpening":true}}})).is_err());
        let mut invalid = UiSettings::default();
        invalid.top_matches = 0;
        assert!(invalid.validate().is_err());
        invalid.top_matches = 101;
        assert!(invalid.validate().is_err());
    }
}
