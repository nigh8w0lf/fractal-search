//! Read-only MCP server for the independent Quickshell search catalog.
#[path = "fractal_search_mcp/transport.rs"]
mod transport;
use anyhow::{ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use clap::Parser;
use fractal_search_core::{
    agent::{validate_asset_id, validate_page, validate_query, AgentCatalog},
    runtime::{ExecutionProvider, NativeRuntime},
    Embedder,
};
use rmcp::{model::*, service::RequestContext, ErrorData, RoleServer, ServerHandler, ServiceExt};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use std::{
    io::Read,
    path::PathBuf,
    sync::{atomic::Ordering, Arc, Mutex},
};

#[derive(Parser)]
#[command(
    version,
    about = "Read-only stdio MCP for the Fractal Search Quickshell plugin"
)]
struct Options {
    /// Print compiled providers without starting MCP, opening a catalog or loading models.
    #[arg(long)]
    providers: bool,
    #[arg(long, env = "FRACTAL_SEARCH_DATA_DIR")]
    data_dir: Option<PathBuf>,
    #[arg(long, env = "FRACTAL_SEARCH_MODEL_DIR")]
    model_dir: Option<PathBuf>,
    #[arg(long, env = "FRACTAL_SEARCH_THREADS")]
    threads: Option<usize>,
    #[arg(long, env="FRACTAL_SEARCH_PROVIDER", default_value="cpu", value_parser=["cpu","cuda"])]
    provider: String,
    #[arg(long, env = "FRACTAL_SEARCH_DEVICE_ID")]
    device_id: Option<i32>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Browse {
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Search {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Asset {
    asset_id: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Thumbnail {
    thumbnail_handle: String,
}
fn default_limit() -> usize {
    20
}
fn arguments<T: DeserializeOwned>(value: Option<JsonObject>) -> Result<T, ErrorData> {
    serde_json::from_value(Value::Object(value.unwrap_or_default())).map_err(|_| {
        ErrorData::invalid_params(
            "Invalid tool arguments or unknown fields; follow the tool input schema",
            None,
        )
    })
}
enum Action {
    Status,
    Browse(Browse),
    Search(Search),
    Metadata(Asset),
    Thumbnail(Thumbnail),
}
fn action(request: CallToolRequestParams) -> Result<Action, ErrorData> {
    let action = match request.name.as_ref() {
        "status" => {
            let _: Empty = arguments(request.arguments)?;
            Action::Status
        }
        "browse_assets" => Action::Browse(arguments(request.arguments)?),
        "search_assets" => Action::Search(arguments(request.arguments)?),
        "get_asset_metadata" => Action::Metadata(arguments(request.arguments)?),
        "get_thumbnail" => Action::Thumbnail(arguments(request.arguments)?),
        _ => return Err(ErrorData::invalid_params("Unknown tool", None)),
    };
    let validation = match &action {
        Action::Browse(args) => validate_page(args.offset, args.limit),
        Action::Search(args) => {
            validate_query(&args.query).and_then(|_| validate_page(0, args.limit))
        }
        Action::Metadata(args) => validate_asset_id(&args.asset_id),
        Action::Thumbnail(args) => args
            .thumbnail_handle
            .strip_prefix("thumbnail_")
            .context("Invalid thumbnail handle")
            .and_then(validate_asset_id),
        Action::Status => Ok(()),
    };
    validation.map_err(|_| {
        ErrorData::invalid_params(
            "Invalid argument value; check query, ID, offset and limit bounds",
            None,
        )
    })?;
    Ok(action)
}
struct State {
    data_dir: PathBuf,
    model_dir: Option<PathBuf>,
    threads: usize,
    provider: ExecutionProvider,
    runtime: Option<NativeRuntime>,
}
impl State {
    fn model_directory(&self) -> Result<PathBuf> {
        if let Some(path) = &self.model_dir {
            return Ok(path.clone());
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Settings {
            model_directory: PathBuf,
            #[serde(default, rename = "ui")]
            _ui: Option<Value>,
        }
        let saved = match std::fs::File::open(self.data_dir.join("settings.json")) {
            Ok(file) => {
                let mut bytes = Vec::new();
                file.take(65537)
                    .read_to_end(&mut bytes)
                    .context("Cannot read model settings")?;
                ensure!(bytes.len() <= 65536, "Model settings exceed 64 KiB");
                let settings: Settings =
                    serde_json::from_slice(&bytes).context("Invalid model settings")?;
                Some(settings.model_directory)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(_) => anyhow::bail!("Cannot read model settings"),
        };
        saved.map(Ok).unwrap_or_else(|| {
            dirs::cache_dir()
                .context("Cannot locate model cache")
                .map(|path| path.join("fractal-search/models/fgclip2-33f3b9d4"))
        })
    }
    fn execute(&mut self, action: Action) -> Result<CallToolResult> {
        let catalog = AgentCatalog::open(&self.data_dir)?;
        match action {
            Action::Status => {
                let mut status = serde_json::to_value(catalog.status()?)?;
                status["textModelLoaded"] =
                    json!(self.runtime.as_ref().is_some_and(|r| r.text_loaded()));
                status["provider"] = json!(self.provider.name());
                status["deviceId"] = json!(self.provider.device_id());
                status["readOnly"] = json!(true);
                Ok(CallToolResult::structured(status))
            }
            Action::Browse(args) => Ok(CallToolResult::structured(serde_json::to_value(
                catalog.browse(args.offset, args.limit)?,
            )?)),
            Action::Metadata(args) => Ok(CallToolResult::structured(serde_json::to_value(
                catalog.metadata(&args.asset_id)?,
            )?)),
            Action::Thumbnail(args) => Ok(CallToolResult::success(vec![ContentBlock::image(
                STANDARD.encode(catalog.thumbnail(&args.thumbnail_handle)?),
                "image/webp",
            )])),
            Action::Search(args) => {
                if self.runtime.is_none() {
                    self.runtime = Some(NativeRuntime::with_provider(
                        self.model_directory()?,
                        self.threads,
                        self.provider,
                    )?);
                }
                let embedding = self.runtime.as_mut().expect("runtime initialized").embed_text(&args.query).map_err(|error| {
                    eprintln!("[MCP] Text inference failed: {error:#}");
                    anyhow::anyhow!("Text inference failed; check the model setup and selected provider in server diagnostics")
                })?;
                Ok(CallToolResult::structured(
                    json!({"items":catalog.search(&embedding,args.limit)?,"limit":args.limit}),
                ))
            }
        }
    }
}
#[derive(Clone)]
struct Server {
    state: Arc<Mutex<State>>,
}
fn tools() -> Vec<Tool> {
    let limit = json!({"type":"integer","minimum":1,"maximum":100,"default":20});
    let mut entries = Vec::new();
    for (name,description,properties,required) in [
        ("status","Read indexed image and folder counts, profile and inference state; no model loading.",json!({}),json!([])),
        ("browse_assets","Browse indexed image metadata in stable filename order. Returns opaque asset and thumbnail handles, never original paths.",json!({"offset":{"type":"integer","minimum":0,"maximum":250000,"default":0},"limit":limit}),json!([])),
        ("search_assets","Search indexed images using FG-CLIP2 short text. Query is limited to 4096 UTF-8 bytes; only this tool loads text inference. Returns at most 100 ranked images.",json!({"query":{"type":"string","minLength":1,"maxLength":4096},"limit":limit}),json!(["query"])),
        ("get_asset_metadata","Read metadata for an opaque asset ID returned by browse/search.",json!({"assetId":{"type":"string","pattern":"^asset_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"}}),json!(["assetId"])),
        ("get_thumbnail","Read an existing managed WebP thumbnail, at most 384 pixels per side and 1 MiB. Returns MCP image content; does not read originals or repair caches.",json!({"thumbnailHandle":{"type":"string","pattern":"^thumbnail_asset_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"}}),json!(["thumbnailHandle"])),
    ] {
        let schema = json!({"type":"object","properties":properties,"required":required,"additionalProperties":false});
        entries.push(Tool::new_with_raw(name,Some(description.into()),schema.as_object().expect("schema object").clone()).with_annotations(ToolAnnotations::new().read_only(true).destructive(false).idempotent(true).open_world(false)));
    }
    entries
}
impl ServerHandler for Server {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("fractal-search-mcp",env!("CARGO_PKG_VERSION")))
            .with_instructions("Read-only access to the Quickshell Fractal Search catalog. Index folders and set up models in the desktop plugin first. Original paths and files are never returned. Thumbnails disclose image content to the calling client only when requested.")
    }
    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        if request.is_some_and(|r| r.cursor.is_some()) {
            return Err(ErrorData::invalid_params("Tool list has no cursor", None));
        }
        Ok(ListToolsResult::with_all_items(tools()))
    }
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let action = action(request)?;
        let state = self.state.clone();
        let result = tokio::task::spawn_blocking(move || {
            state
                .lock()
                .map_err(|_| anyhow::anyhow!("MCP worker state is unavailable"))?
                .execute(action)
        })
        .await
        .map_err(|_| ErrorData::internal_error("MCP worker failed", None))?;
        let result = match result {
            Ok(result) => result,
            Err(error) => CallToolResult::error(vec![ContentBlock::text(error.to_string())]),
        };
        Ok(result.into())
    }
}
#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let options = Options::parse();
    if options.providers {
        let providers = if cfg!(feature = "cuda") {
            vec!["cpu", "cuda"]
        } else {
            vec!["cpu"]
        };
        println!(
            "{}",
            json!({"cudaCompiled":cfg!(feature = "cuda"),"providers":providers})
        );
        return Ok(());
    }
    let data_dir = options.data_dir.map(Ok).unwrap_or_else(|| {
        dirs::data_dir()
            .context("Cannot locate data directory")
            .map(|p| p.join("fractal-search"))
    })?;
    let threads = options.threads.unwrap_or(
        std::thread::available_parallelism()?
            .get()
            .div_ceil(2)
            .clamp(1, 6),
    );
    ensure!(
        (1..=64).contains(&threads),
        "--threads must be between 1 and 64"
    );
    let provider = ExecutionProvider::from_selection(&options.provider, options.device_id)?;
    let server = Server {
        state: Arc::new(Mutex::new(State {
            data_dir,
            model_dir: options.model_dir,
            threads,
            provider,
            runtime: None,
        })),
    };
    let transport = transport::BoundedStdio::new();
    let failed = transport.failed.clone();
    let service = server
        .serve(transport)
        .await
        .context("MCP initialization failed")?;
    service.waiting().await.context("MCP service failed")?;
    ensure!(
        !failed.load(Ordering::Acquire),
        "MCP transport failed; see diagnostics"
    );
    Ok(())
}
