#!/usr/bin/env bash
set -euo pipefail
SEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SEARCH_PROFILE=release
SEARCH_BUILD=1
SEARCH_CUDA=0
SEARCH_OUTPUT="$SEARCH_ROOT/target/fractal-search-plugin"
while (($#)); do
  case "$1" in
    --dev) SEARCH_PROFILE=debug; shift ;;
    --no-build) SEARCH_BUILD=0; shift ;;
    --cuda) SEARCH_CUDA=1; shift ;;
    --output) [[ $# -ge 2 ]] || { echo 'Missing --output value' >&2; exit 1; }; SEARCH_OUTPUT="$(realpath -m -- "$2")"; shift 2 ;;
    *) echo "Unknown packaging argument: $1" >&2; exit 1 ;;
  esac
done
cd -- "$SEARCH_ROOT"
if ((SEARCH_BUILD)); then
  SEARCH_ARGS=(build --locked -p fractal-search)
  [[ $SEARCH_PROFILE == release ]] && SEARCH_ARGS+=(--release)
  ((SEARCH_CUDA)) && SEARCH_ARGS+=(--features cuda)
  cargo "${SEARCH_ARGS[@]}"
fi
SEARCH_TARGET=$(cargo metadata --no-deps --format-version 1 --locked | jq -er .target_directory)
SEARCH_BINARY="$SEARCH_TARGET/$SEARCH_PROFILE/fractal-search"
[[ -x $SEARCH_BINARY ]] || { echo "Missing built backend: $SEARCH_BINARY" >&2; exit 1; }
SEARCH_CAPABILITIES=$("$SEARCH_BINARY" providers)
[[ $(jq -er '.cudaCompiled | if . then 1 else 0 end' <<< "$SEARCH_CAPABILITIES") == "$SEARCH_CUDA" ]] || { echo 'Built backend does not match the requested CPU/CUDA bundle. Rebuild with the matching --cuda selection.' >&2; exit 1; }
SEARCH_PICKER="$SEARCH_TARGET/$SEARCH_PROFILE/fractal-folder-picker"
[[ -x $SEARCH_PICKER ]] || { echo "Missing built folder chooser: $SEARCH_PICKER" >&2; exit 1; }
SEARCH_MCP="$SEARCH_TARGET/$SEARCH_PROFILE/fractal-search-mcp"
[[ -x $SEARCH_MCP ]] || { echo "Missing built MCP server: $SEARCH_MCP" >&2; exit 1; }
SEARCH_MCP_CAPABILITIES=$("$SEARCH_MCP" --providers)
[[ $(jq -er '.cudaCompiled | if . then 1 else 0 end' <<< "$SEARCH_MCP_CAPABILITIES") == "$SEARCH_CUDA" ]] || { echo 'Built MCP server does not match the requested CPU/CUDA bundle. Rebuild all binaries with the matching --cuda selection.' >&2; exit 1; }
if ((SEARCH_CUDA)); then
  for SEARCH_LIBRARY in libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do
    [[ -f $SEARCH_TARGET/$SEARCH_PROFILE/$SEARCH_LIBRARY ]] || { echo "Missing CUDA provider library $SEARCH_LIBRARY beside the built backend. Rebuild with --cuda and ort copy-dylibs enabled." >&2; exit 1; }
  done
fi
[[ ! -e $SEARCH_OUTPUT ]] || { echo "Output already exists: $SEARCH_OUTPUT (choose a new output directory)" >&2; exit 1; }
mkdir -p -- "$SEARCH_OUTPUT/ui" "$SEARCH_OUTPUT/bin"
cp -- "$SEARCH_BINARY" "$SEARCH_OUTPUT/bin/fractal-search"
cp -- "$SEARCH_PICKER" "$SEARCH_OUTPUT/bin/fractal-folder-picker"
cp -- "$SEARCH_MCP" "$SEARCH_OUTPUT/bin/fractal-search-mcp"
if ((SEARCH_CUDA)); then
  cp -L -- "$SEARCH_TARGET/$SEARCH_PROFILE/libonnxruntime_providers_shared.so" "$SEARCH_TARGET/$SEARCH_PROFILE/libonnxruntime_providers_cuda.so" "$SEARCH_OUTPUT/bin/"
fi
cp -- quickshell-app/manifest.json quickshell-app/packaging/install.sh quickshell-app/packaging/prune-runtimes.sh quickshell-app/packaging/uninstall.sh quickshell-app/packaging/close-window.sh quickshell-app/packaging/close-binding.sh quickshell-app/packaging/README.md quickshell-app/packaging/NOTICE "$SEARCH_OUTPUT/"
for SEARCH_QML in quickshell-app/ui/*.qml quickshell-app/ui/qmldir; do
  [[ ${SEARCH_QML##*/} == shell.qml ]] && continue
  cp -- "$SEARCH_QML" "$SEARCH_OUTPUT/ui/"
done
cp -- crates/fractal-search-core/models/embeddinggemma2.json "$SEARCH_OUTPUT/model-manifest.json"
cp -- quickshell-app/MCP.md "$SEARCH_OUTPUT/MCP.md"
cp -- quickshell-app/CUDA.md "$SEARCH_OUTPUT/CUDA.md"
jq -n --arg architecture "$(uname -m)" --arg profile "$SEARCH_PROFILE" --argjson cuda "$SEARCH_CUDA" '{architecture:$architecture,profile:$profile,protocolVersion:1,cuda:($cuda == 1)}' > "$SEARCH_OUTPUT/bundle.json"
(cd -- "$SEARCH_OUTPUT"; find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
[[ ! -x /usr/bin/omarchy ]] || omarchy plugin validate "$SEARCH_OUTPUT"
echo "Bundle ready: $SEARCH_OUTPUT"
echo "Install: $SEARCH_OUTPUT/install.sh"
