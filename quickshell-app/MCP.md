# Fractal Search MCP

`fractal-search-mcp` gives an MCP client read-only access to the **Quickshell plugin's existing search catalog**. It has a separate process and permission boundary from the trusted desktop worker/socket. It does not create a catalog, index folders, edit settings, open originals or repair thumbnails. Index supported media and set up the pinned EmbeddingGemma 2 models in Fractal Search first.

The server uses stdio JSON-RPC and the official Rust MCP SDK, pinned to rmcp 3.1.4. It negotiates the SDK's supported initialize-based protocol revisions (including 2025-11-25). The client launches one process; no TCP listener or daemon registration is involved. JSON-RPC goes to stdout and diagnostics to stderr. See the [MCP stdio specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports#stdio).

## Connect a client

An installed bundle exposes a stable launcher at:

```text
~/.config/omarchy/plugins/fractal.search/fractal-search-mcp
```

Use an absolute path in the client's configuration; MCP clients need not expand `~`. If using a custom XDG configuration directory, use its matching `omarchy/plugins/fractal.search/fractal-search-mcp` path. A typical client entry is:

```json
{
  "mcpServers": {
    "fractal-search": {
      "command": "/home/YOUR_USER/.config/omarchy/plugins/fractal.search/fractal-search-mcp",
      "args": []
    }
  }
}
```

The installed wrapper selects the installation's catalog directory. To run the developer binary directly:

```bash
cargo build -p fractal-search-mcp --locked
target/debug/fractal-search-mcp --data-dir /absolute/path/to/fractal-search
```

The default catalog is `$XDG_DATA_HOME/fractal-search/catalog.lancedb/` (normally `~/.local/share/fractal-search/catalog.lancedb/`). The process can run while the desktop service indexes the catalog. Each read pins a LanceDB table snapshot containing both metadata and vectors. It validates the schema/model profile and does not acquire the desktop service lock, create a catalog, write records or repair caches. Missing or incompatible catalogs return visible tool errors. This schema-4 `embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2` format uses normalized 512-dimensional catalog/query vectors. The unchanged FP32 graph emits 768 dimensions, then the runtime keeps the leading 512 and L2-normalizes. The previous 768-dimensional profile is incompatible even at schema 4: use a separate fresh data directory and explicitly reindex with the desktop service. Old catalogs are never deleted or converted automatically; no SQLite migration or dimension selector is provided.

`--model-dir` overrides model selection. Otherwise the first semantic search reads `modelDirectory` from the plugin's `settings.json`, or uses `$XDG_CACHE_HOME/fractal-search/models/embeddinggemma2-daa72c51` when no settings exist. Missing or malformed settings/assets fail visibly. No model files are downloaded by the MCP process. `--threads` accepts 1–64; the default is half the logical CPU count, rounded up and capped at six.

`--provider cpu` is the default. A CUDA-enabled bundle can explicitly select `--provider cuda --device-id 0`. CUDA errors do not retry on CPU. The MCP process has its own resident text session; it does not share loaded models or GPU memory with the desktop worker. Catalog metadata and thumbnails do not initialize a model or GPU. The vision and audio encoders are never loaded by this server. No decoder processes are required for catalog reads.

`fractal-search-mcp --providers` prints `{ "cudaCompiled": false, "providers": ["cpu"] }` (or the corresponding CUDA-enabled list) and exits without starting MCP, opening a catalog, reading model assets or initializing a GPU. Packaging uses this diagnostic to verify that both executables support the advertised providers.

The same settings can be provided through `FRACTAL_SEARCH_DATA_DIR`, `FRACTAL_SEARCH_MODEL_DIR`, `FRACTAL_SEARCH_THREADS`, `FRACTAL_SEARCH_PROVIDER` and `FRACTAL_SEARCH_DEVICE_ID`. CLI options take precedence; a device ID requires the CUDA provider.

## Tools and limits

All five tools advertise `readOnlyHint: true`, `destructiveHint: false`, `idempotentHint: true` and `openWorldHint: false`. Tool argument schemas reject additional properties, and runtime deserialization enforces the same rule.

| Tool | Arguments | Result |
| --- | --- | --- |
| `status` | `{}` | Segment/file/source counts, model profile, text-session state and selected provider/device. |
| `browse_assets` | Optional `offset` (0–250,000), `limit` (1–100, default 20), `mediaKind` (`image`, `text`, `audio`, `video` or null). | Page of metadata ordered by filename and asset ID, with total count. |
| `search_assets` | Required `query` (1–4,096 UTF-8 bytes), optional `limit` (1–100, default 20) and `mediaKind` (same enum as browse). | Ranked segment metadata, actual cosine similarities, and `searchMode`; Text adds `hybridScore` from BM25 + semantic RRF; lazily loads only native text inference. |
| `get_asset_metadata` | Required `assetId`. | One metadata object for an existing opaque `asset_<UUID>` identity. |
| `get_thumbnail` | Required `thumbnailHandle`. | One MCP `image` content block containing base64 WebP pixels. |

`status.modelProfile` reports the indexed 512-dimensional profile, not the native graph contract. `status` reports `assetCount` (searchable segments), `fileCount` (distinct indexed source files) and `sourceCount` (folders).

Metadata contains `assetId`, `filename`, `mediaKind` (`image`, `text`, `audio` or `video`), `segmentIndex`, nullable `startMs`/`endMs`, nullable `textExcerpt`, `width`, `height`, nullable `thumbnailHandle` and, for search, `similarity` plus optional `hybridScore` (present only for text-hybrid search). Text excerpts are cached, bounded to 2,048 UTF-8 bytes and disclose source text to the client. Nonvisual media has zero dimensions and no thumbnail handle. A video may produce distinct audio and visual segments with the same filename/time range. Metadata excludes original directories, filesystem path fields and original file bytes. Handles must come from browse/search results; no tool accepts a filesystem path. File names, cached text excerpts and requested thumbnail pixels still reveal library content to the connected client.

Omit `mediaKind` or pass null for All media. Text-only query search fuses BM25 and exact semantic ranks using equal-weight RRF (k=60); All media and other filters retain cosine order. `searchMode` is `"hybrid"` only for Text and `"semantic"` otherwise. `hybridScore` is a rank-fusion value, never a percent or cosine; `similarity` stays actual cosine. The complete filename + passage lexical field is internal and never serialized. Empty-query browsing uses `browse_assets` and stays model-free. Literal ICU terms, independent bounded candidate lists, lifecycle/failure behavior and maintenance limits are detailed in the [hybrid contract](../docs/HYBRID-TEXT-SEARCH.md).

Image/video thumbnails are limited to one managed cache entry, 1 MiB encoded and 384 pixels on the longest side. Reads pin the managed directory, reject symbolic links, hard links and foreign-owned cache files, and verify indexed dimensions. The bounded decoder then re-encodes fresh WebP pixels, discarding embedded metadata and arbitrary RIFF payloads. Missing, corrupt or legacy thumbnails produce an error directing the user to the desktop plugin; the MCP process never reads originals to repair them.

Input JSON-RPC frames include their newline and are capped at 64 KiB; output frames are capped at 2 MiB. Unknown envelope/tool-parameter fields are rejected. One request is admitted at a time, so concurrent calls apply pipe backpressure instead of building an unbounded inference queue. Accepted work and responses drain before EOF exits the process. Cancellation notifications do not interrupt native inference already in progress. An overlong or unterminated frame reports an error and closes the connection explicitly. Invalid JSON receives a parse error without echoing the submitted data.

## Validation

Run the workspace checks plus the temporary-catalog integration test:

```bash
cargo fmt --all --check
cargo check --workspace --locked
cargo test --workspace --locked
cargo build -p fractal-search -p fractal-folder-picker -p fractal-search-mcp --bins --locked
python3 quickshell-app/tests/smoke_mcp.py \
  target/debug/fractal-search target/debug/fractal-search-mcp \
  /absolute/path/to/pinned/models
```

The smoke check creates two synthetic images and a new temporary catalog, indexes through the desktop service, removes the test originals, and verifies metadata/thumbnails while that service remains running. It exercises initialization/tool discovery, strict arguments and frame limits, read-only behavior, real native text search, missing-model errors, no vision session, and complete responses for queued searches followed by EOF. `smoke_multimodal.py` additionally checks text/audio/video segment metadata, explicit media filters, text-only hybrid scores/order, unchanged all-media cosine order, soundtrack separation, missing thumbnails and no original-file reads after synthetic originals are removed. Core unit tests cover incompatible catalogs, legacy identities, cache confinement, symlinks/hardlinks and no thumbnail repair. It never uses the user's photo library or changes client configuration.
