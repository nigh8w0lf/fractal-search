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
if ((SEARCH_CUDA)) && [[ ${ORT_CUDA_VERSION-12} != 12 ]]; then
  echo 'CUDA bundles require ORT_CUDA_VERSION=12 to match the supported runtime libraries.' >&2
  exit 1
fi
if ((SEARCH_BUILD)); then
  SEARCH_ARGS=(build --locked -p fractal-search -p fractal-folder-picker -p fractal-search-mcp --bins)
  [[ $SEARCH_PROFILE == release ]] && SEARCH_ARGS+=(--release)
  ((SEARCH_CUDA)) && SEARCH_ARGS+=(--features fractal-search/cuda,fractal-search-mcp/cuda)
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
# Keep one source manifest at the repository root for marketplace discovery.
# Bundles flatten quickshell-app/, so rebase only the panel entry point.
jq -e '.entryPoints.panel == "quickshell-app/ui/SetupPanel.qml"' manifest.json >/dev/null || { echo 'Unexpected source panel entry point; update bundle layout explicitly.' >&2; exit 1; }
jq '.entryPoints.panel = "ui/SearchPanel.qml"' manifest.json > "$SEARCH_OUTPUT/manifest.json"
cp -- LICENSE quickshell-app/packaging/install.sh quickshell-app/packaging/prune-runtimes.sh quickshell-app/packaging/uninstall.sh quickshell-app/packaging/close-window.sh quickshell-app/packaging/close-binding.sh quickshell-app/packaging/README.md quickshell-app/packaging/NOTICE "$SEARCH_OUTPUT/"
cp -- docs/AGENT-SETUP.md "$SEARCH_OUTPUT/AGENT-SETUP.md"
cp -- quickshell-app/packaging/AGENTS.bundle.md "$SEARCH_OUTPUT/AGENTS.md"
SEARCH_NOTICE_ARGS=("$SEARCH_OUTPUT")
((SEARCH_CUDA)) && SEARCH_NOTICE_ARGS+=(--cuda)
python3 scripts/generate-third-party-notices.py "${SEARCH_NOTICE_ARGS[@]}"
cp -- vendor/lancedb/LICENSE "$SEARCH_OUTPUT/LANCEDB-LICENSE"
for SEARCH_QML in quickshell-app/ui/*.qml quickshell-app/ui/qmldir; do
  [[ ${SEARCH_QML##*/} == shell.qml ]] && continue
  cp -- "$SEARCH_QML" "$SEARCH_OUTPUT/ui/"
done
cp -- crates/fractal-search-core/models/embeddinggemma2.json "$SEARCH_OUTPUT/model-manifest.json"
cp -- quickshell-app/MCP.md "$SEARCH_OUTPUT/MCP.md"
cp -- quickshell-app/CUDA.md "$SEARCH_OUTPUT/CUDA.md"
jq -n --arg architecture "$(uname -m)" --arg profile "$SEARCH_PROFILE" --argjson cuda "$SEARCH_CUDA" --arg sourceCommit "$(git rev-parse HEAD)" --arg rustc "$(rustc --version)" --arg glibc "$(getconf GNU_LIBC_VERSION)" --argjson sourceDirty "$(if [[ -n $(git status --porcelain) ]]; then echo true; else echo false; fi)" '{architecture:$architecture,profile:$profile,protocolVersion:1,cuda:($cuda == 1),sourceCommit:$sourceCommit,sourceDirty:$sourceDirty,rustc:$rustc,buildGlibc:$glibc}' > "$SEARCH_OUTPUT/bundle.json"
(cd -- "$SEARCH_OUTPUT"; find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
if command -v omarchy >/dev/null; then
  omarchy plugin validate "$SEARCH_OUTPUT"
else
  echo 'Omarchy manifest validation was not run: omarchy is not installed on this build host.' >&2
fi
echo "Bundle ready: $SEARCH_OUTPUT"
echo "Install: $SEARCH_OUTPUT/install.sh"
