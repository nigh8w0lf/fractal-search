#!/usr/bin/env bash
set -euo pipefail
SEARCH_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
SEARCH_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
SEARCH_RUNTIME_DIR="${XDG_RUNTIME_DIR:?XDG_RUNTIME_DIR must be set by the desktop session}"
SEARCH_SKIP_SHELL=0
SEARCH_MODEL_DIR=''
SEARCH_THREADS=''
SEARCH_PROVIDER=''
SEARCH_DEVICE_ID=''
SEARCH_CUDA_LIB_DIR=''
SEARCH_CUDA_LIB_DIR_SET=0
SEARCH_BIND_CLOSE=0
while (($#)); do
  case "$1" in
    --config-home|--data-home|--runtime-dir|--model-dir|--threads|--provider|--device-id|--cuda-lib-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      case "$1" in
        --config-home) SEARCH_CONFIG_HOME=$2 ;;
        --data-home) SEARCH_DATA_HOME=$2 ;;
        --runtime-dir) SEARCH_RUNTIME_DIR=$2 ;;
        --model-dir) SEARCH_MODEL_DIR=$2 ;;
        --threads) SEARCH_THREADS=$2 ;;
        --provider) SEARCH_PROVIDER=$2 ;;
        --device-id) SEARCH_DEVICE_ID=$2 ;;
        --cuda-lib-dir) SEARCH_CUDA_LIB_DIR=$2; SEARCH_CUDA_LIB_DIR_SET=1 ;;
      esac
      shift 2 ;;
    --skip-shell) SEARCH_SKIP_SHELL=1; shift ;;
    --bind-close) SEARCH_BIND_CLOSE=1; shift ;;
    *) echo "Unknown install argument: $1" >&2; exit 1 ;;
  esac
done
for SEARCH_DIR in "$SEARCH_CONFIG_HOME" "$SEARCH_DATA_HOME" "$SEARCH_RUNTIME_DIR"; do
  [[ $SEARCH_DIR == /* && $SEARCH_DIR != *$'\n'* && $SEARCH_DIR != *$'\r'* ]] || { echo 'Install paths must be absolute single-line paths' >&2; exit 1; }
done
for SEARCH_TOOL in jq sha256sum systemctl curl flock; do
  command -v "$SEARCH_TOOL" >/dev/null || { echo "Missing $SEARCH_TOOL. Install with: omarchy pkg add $SEARCH_TOOL" >&2; exit 1; }
done
for SEARCH_TOOL in ffmpeg ffprobe; do
  command -v "$SEARCH_TOOL" >/dev/null || { echo "Missing $SEARCH_TOOL for audio/video indexing. Install with: omarchy pkg add ffmpeg" >&2; exit 1; }
done
command -v gio >/dev/null || { echo "Missing gio for opening files in their default desktop applications. Install with: omarchy pkg add glib2" >&2; exit 1; }
[[ $(jq -r .architecture "$SEARCH_BUNDLE/bundle.json") == "$(uname -m)" ]] || { echo 'The bundle architecture does not match this computer' >&2; exit 1; }
[[ $(jq -r .protocolVersion "$SEARCH_BUNDLE/bundle.json") == 1 ]] || { echo 'Unsupported bundle protocol' >&2; exit 1; }
(cd -- "$SEARCH_BUNDLE"; sha256sum --check --quiet SHA256SUMS)
SEARCH_LINKS=$(ldd "$SEARCH_BUNDLE/bin/fractal-search")
[[ $SEARCH_LINKS != *'not found'* ]] || { echo "$SEARCH_LINKS" >&2; echo 'Install native runtime dependencies with: omarchy pkg add libheif libjpeg-turbo gcc-libs' >&2; exit 1; }
[[ -x $SEARCH_BUNDLE/bin/fractal-folder-picker ]] || { echo 'The bundle is missing its native folder chooser' >&2; exit 1; }
SEARCH_PICKER_LINKS=$(ldd "$SEARCH_BUNDLE/bin/fractal-folder-picker")
[[ $SEARCH_PICKER_LINKS != *'not found'* ]] || { echo "$SEARCH_PICKER_LINKS" >&2; echo 'The native folder chooser has missing runtime dependencies' >&2; exit 1; }
[[ -x $SEARCH_BUNDLE/bin/fractal-search-mcp ]] || { echo 'The bundle is missing its read-only MCP server' >&2; exit 1; }
SEARCH_MCP_LINKS=$(ldd "$SEARCH_BUNDLE/bin/fractal-search-mcp")
[[ $SEARCH_MCP_LINKS != *'not found'* ]] || { echo "$SEARCH_MCP_LINKS" >&2; echo 'The MCP server has missing runtime dependencies' >&2; exit 1; }
if ((!SEARCH_SKIP_SHELL)); then
  command -v omarchy >/dev/null && command -v omarchy-shell >/dev/null || { echo 'Omarchy Shell is required' >&2; exit 1; }
  [[ -r /usr/lib/qt6/qml/QtMultimedia/qmldir ]] || { echo "Qt Multimedia is required for audio/video previews. Install with: omarchy pkg add qt6-multimedia qt6-multimedia-ffmpeg" >&2; exit 1; }
  omarchy plugin validate "$SEARCH_BUNDLE"
  omarchy-shell shell ping >/dev/null
fi
SEARCH_PLUGIN="$SEARCH_CONFIG_HOME/omarchy/plugins/fractal.search"
SEARCH_UNIT="$SEARCH_CONFIG_HOME/systemd/user/fractal-search.service"
SEARCH_RUNTIME="$SEARCH_DATA_HOME/fractal-search/runtime"
SEARCH_VERSION="$(jq -r .version "$SEARCH_BUNDLE/manifest.json")-$(sha256sum "$SEARCH_BUNDLE/SHA256SUMS" | cut -c1-12)"
SEARCH_BINDINGS="$SEARCH_CONFIG_HOME/hypr/bindings.lua"
if [[ -f $SEARCH_BINDINGS ]] && awk '$0 == "-- BEGIN Fractal Search close binding" {found=1} END {exit !found}' "$SEARCH_BINDINGS"; then SEARCH_BIND_CLOSE=1; fi
if ((SEARCH_BIND_CLOSE && !SEARCH_SKIP_SHELL)); then
  for SEARCH_TOOL in hyprctl notify-send; do
    command -v "$SEARCH_TOOL" >/dev/null || { echo "The close binding requires $SEARCH_TOOL" >&2; exit 1; }
  done
fi
if ((SEARCH_BIND_CLOSE)); then SEARCH_VERSION+="-close"; fi
[[ ! -e $SEARCH_PLUGIN || -f $SEARCH_PLUGIN/.fractal-search-install ]] || { echo 'Refusing to overwrite an unmanaged fractal.search plugin' >&2; exit 1; }
if [[ -e $SEARCH_UNIT ]] && ! awk '/^# Managed by Fractal Search installer$/ {found=1} END {exit !found}' "$SEARCH_UNIT"; then
  echo 'Refusing to overwrite an unmanaged search service' >&2; exit 1
fi
# Serialize updates and pruning before reading the installed configuration.
mkdir -p -- "$SEARCH_RUNTIME"
exec {SEARCH_INSTALL_LOCK}>"$SEARCH_RUNTIME/.install.lock"
flock -n "$SEARCH_INSTALL_LOCK" || { echo 'Another Fractal Search install is running.' >&2; exit 1; }
SEARCH_PREVIOUS=''
if [[ -f $SEARCH_PLUGIN/.fractal-search-install ]]; then
  SEARCH_PREVIOUS=$(cat -- "$SEARCH_PLUGIN/.fractal-search-install")
fi
SEARCH_RUNTIME_CONFIG="$SEARCH_PLUGIN/.fractal-search-runtime.json"
if [[ -f $SEARCH_RUNTIME_CONFIG ]]; then
  jq -e 'type == "object" and ((keys - ["provider", "deviceId", "cudaLibraryDirectory"]) | length == 0) and (.provider == "cpu" or .provider == "cuda") and ((.deviceId == null) or (.deviceId | type == "number" and floor == . and . >= 0 and . <= 2147483647)) and ((.cudaLibraryDirectory == null) or (.cudaLibraryDirectory | type == "string"))' "$SEARCH_RUNTIME_CONFIG" >/dev/null || { echo 'Invalid installed inference configuration; repair it or reinstall after backing it up.' >&2; exit 1; }
  if [[ -z $SEARCH_PROVIDER ]]; then
    SEARCH_PROVIDER=$(jq -r .provider "$SEARCH_RUNTIME_CONFIG")
    [[ -n $SEARCH_DEVICE_ID ]] || SEARCH_DEVICE_ID=$(jq -r '.deviceId // empty' "$SEARCH_RUNTIME_CONFIG")
  fi
  if ((!SEARCH_CUDA_LIB_DIR_SET)) && [[ $SEARCH_PROVIDER == cuda ]]; then
    SEARCH_CUDA_LIB_DIR=$(jq -r '.cudaLibraryDirectory // empty' "$SEARCH_RUNTIME_CONFIG")
  fi
fi
[[ -n $SEARCH_PROVIDER ]] || SEARCH_PROVIDER=cpu
case "$SEARCH_PROVIDER" in
  cpu) [[ -z $SEARCH_DEVICE_ID && -z $SEARCH_CUDA_LIB_DIR ]] || { echo '--device-id and --cuda-lib-dir require --provider cuda' >&2; exit 1; } ;;
  cuda)
    [[ -n $SEARCH_DEVICE_ID ]] || SEARCH_DEVICE_ID=0
    [[ $SEARCH_DEVICE_ID =~ ^(0|[1-9][0-9]{0,9})$ && $SEARCH_DEVICE_ID -le 2147483647 ]] || { echo 'CUDA device ID must be an integer from 0 to 2147483647' >&2; exit 1; }
    [[ $(jq -r .cuda "$SEARCH_BUNDLE/bundle.json") == true ]] || { echo 'CUDA selection requires a CUDA bundle. Rebuild/package with --cuda, or explicitly select --provider cpu.' >&2; exit 1; }
    if [[ -n $SEARCH_CUDA_LIB_DIR ]]; then
      [[ $SEARCH_CUDA_LIB_DIR == /* && -d $SEARCH_CUDA_LIB_DIR && $SEARCH_CUDA_LIB_DIR != *:* && $SEARCH_CUDA_LIB_DIR != *$'\n'* && $SEARCH_CUDA_LIB_DIR != *$'\r'* ]] || { echo '--cuda-lib-dir must be one existing absolute directory without colons or newlines' >&2; exit 1; }
      SEARCH_CUDA_LIB_DIR=$(realpath -- "$SEARCH_CUDA_LIB_DIR")
    fi
    for SEARCH_LIBRARY in libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do
      [[ -f $SEARCH_BUNDLE/bin/$SEARCH_LIBRARY ]] || { echo "CUDA bundle is missing $SEARCH_LIBRARY" >&2; exit 1; }
      if [[ -n $SEARCH_CUDA_LIB_DIR ]]; then
        SEARCH_CUDA_LINKS=$(env LD_LIBRARY_PATH="$SEARCH_BUNDLE/bin:$SEARCH_CUDA_LIB_DIR" ldd "$SEARCH_BUNDLE/bin/$SEARCH_LIBRARY")
      else
        SEARCH_CUDA_LINKS=$(env LD_LIBRARY_PATH="$SEARCH_BUNDLE/bin" ldd "$SEARCH_BUNDLE/bin/$SEARCH_LIBRARY")
      fi
      [[ $SEARCH_CUDA_LINKS != *'not found'* ]] || { echo "$SEARCH_CUDA_LINKS" >&2; echo 'CUDA dependencies are missing. Install CUDA 12 and cuDNN 9 runtime libraries, or pass --cuda-lib-dir with their directory. CPU fallback is disabled.' >&2; exit 1; }
    done ;;
  *) echo 'Provider must be cpu or cuda' >&2; exit 1 ;;
esac
# Provider changes also need a new QML URL in the long-lived host engine.
SEARCH_VERSION+="-$SEARCH_PROVIDER"
if [[ $SEARCH_PROVIDER == cuda ]]; then
  SEARCH_VERSION+="-$SEARCH_DEVICE_ID-$(printf '%s' "$SEARCH_CUDA_LIB_DIR" | sha256sum | cut -c1-8)"
fi
SEARCH_DEST="$SEARCH_RUNTIME/$SEARCH_VERSION"
SEARCH_LIBRARY_PATH=''
if [[ $(jq -r .cuda "$SEARCH_BUNDLE/bundle.json") == true ]]; then
  SEARCH_LIBRARY_PATH="$SEARCH_DEST/bin"
  [[ -z $SEARCH_CUDA_LIB_DIR ]] || SEARCH_LIBRARY_PATH+=":$SEARCH_CUDA_LIB_DIR"
fi
mkdir -p -- "$SEARCH_RUNTIME" "$(dirname -- "$SEARCH_PLUGIN")" "$(dirname -- "$SEARCH_UNIT")"
SEARCH_STAGE=$(mktemp -d "$SEARCH_RUNTIME/install.XXXXXX")
SEARCH_OLD_PLUGIN="$SEARCH_STAGE/old-plugin"
SEARCH_OLD_UNIT="$SEARCH_STAGE/old-unit"
SEARCH_OLD_BINDINGS="$SEARCH_STAGE/old-bindings"
SEARCH_BINDING_APPLIED=0
SEARCH_APPLIED=0
SEARCH_DONE=0
SEARCH_SERVICE_WAS_ACTIVE=0
SEARCH_SERVICE_STOPPED=0
cleanup() {
  local SEARCH_EXIT=$?
  local SEARCH_ROLLBACK_FAILED=0
  # Attempt every rollback step and retain recovery files if any step fails.
  set +e
  if ((SEARCH_APPLIED && !SEARCH_DONE)); then
    echo 'Install failed; restoring the previous plugin and service.' >&2
    if ((!SEARCH_SKIP_SHELL)); then
      systemctl --user stop fractal-search.service || SEARCH_ROLLBACK_FAILED=1
    fi
    if [[ -e $SEARCH_PLUGIN ]]; then rm -rf -- "$SEARCH_PLUGIN" || SEARCH_ROLLBACK_FAILED=1; fi
    if [[ -e $SEARCH_OLD_PLUGIN ]]; then mv -- "$SEARCH_OLD_PLUGIN" "$SEARCH_PLUGIN" || SEARCH_ROLLBACK_FAILED=1; fi
    if [[ -e $SEARCH_OLD_UNIT ]]; then cp -- "$SEARCH_OLD_UNIT" "$SEARCH_UNIT" || SEARCH_ROLLBACK_FAILED=1; else rm -f -- "$SEARCH_UNIT" || SEARCH_ROLLBACK_FAILED=1; fi
    if ((SEARCH_BINDING_APPLIED)); then
      if [[ -e $SEARCH_OLD_BINDINGS ]]; then cp -p -- "$SEARCH_OLD_BINDINGS" "$SEARCH_BINDINGS" || SEARCH_ROLLBACK_FAILED=1; else rm -f -- "$SEARCH_BINDINGS" || SEARCH_ROLLBACK_FAILED=1; fi
      if ((!SEARCH_SKIP_SHELL)); then hyprctl reload || SEARCH_ROLLBACK_FAILED=1; fi
    fi
    if ((!SEARCH_SKIP_SHELL)); then
      systemctl --user daemon-reload || SEARCH_ROLLBACK_FAILED=1
      omarchy-shell shell rescanPlugins || SEARCH_ROLLBACK_FAILED=1
    fi
  fi
  if ((!SEARCH_DONE && SEARCH_SERVICE_STOPPED && SEARCH_SERVICE_WAS_ACTIVE && !SEARCH_ROLLBACK_FAILED)); then
    systemctl --user start fractal-search.service || SEARCH_ROLLBACK_FAILED=1
  fi
  if ((SEARCH_ROLLBACK_FAILED)); then
    echo "Rollback incomplete; inspect the errors above. Recovery files retained at $SEARCH_STAGE" >&2
    SEARCH_EXIT=1
  else
    rm -rf -- "$SEARCH_STAGE"
  fi
  exit "$SEARCH_EXIT"
}
trap cleanup EXIT
mkdir -- "$SEARCH_STAGE/plugin"
# Unique component URLs avoid retaining an older compiled QML component in
# Omarchy's long-lived engine. The manifest selects one complete UI version.
mkdir -p -- "$SEARCH_STAGE/plugin/ui/$SEARCH_VERSION"
cp -a -- "$SEARCH_BUNDLE/ui/." "$SEARCH_STAGE/plugin/ui/$SEARCH_VERSION/"
cp -- "$SEARCH_BUNDLE/close-window.sh" "$SEARCH_BUNDLE/close-binding.sh" "$SEARCH_STAGE/plugin/"
jq --arg entry "ui/$SEARCH_VERSION/SearchPanel.qml" '.entryPoints.panel = $entry' "$SEARCH_BUNDLE/manifest.json" > "$SEARCH_STAGE/plugin/manifest.json"
printf '%s\n' "$SEARCH_DEST" > "$SEARCH_STAGE/plugin/.fractal-search-install"
jq -n --arg provider "$SEARCH_PROVIDER" --arg device "$SEARCH_DEVICE_ID" --arg directory "$SEARCH_CUDA_LIB_DIR" '{provider:$provider,deviceId:(if $device == "" then null else ($device|tonumber) end),cudaLibraryDirectory:(if $directory == "" then null else $directory end)}' > "$SEARCH_STAGE/plugin/.fractal-search-runtime.json"
# Stable MCP command path survives versioned runtime upgrades. Arguments remain
# literal argv entries; client-supplied options never pass through shell eval.
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  if [[ -n $SEARCH_LIBRARY_PATH ]]; then
    printf 'export LD_LIBRARY_PATH=%q\n' "$SEARCH_LIBRARY_PATH"
  fi
  printf 'exec %q --data-dir %q "$@"\n' "$SEARCH_DEST/bin/fractal-search-mcp" "$SEARCH_DATA_HOME/fractal-search"
} > "$SEARCH_STAGE/plugin/fractal-search-mcp"
chmod +x -- "$SEARCH_STAGE/plugin/fractal-search-mcp"
# jq produces valid QML string literals, including spaces and backslashes.
SEARCH_VERSION_JSON=$(jq -Rn --arg version "$SEARCH_VERSION" '$version')
SEARCH_SOCKET_JSON=$(jq -Rn --arg path "$SEARCH_RUNTIME_DIR/fractal-search/backend.sock" '$path')
SEARCH_PICKER_JSON=$(jq -Rn --arg path "$SEARCH_DEST/bin/fractal-folder-picker" '$path')
SEARCH_BACKEND_JSON=$(jq -Rn --arg path "$SEARCH_DEST/bin/fractal-search" '$path')
SEARCH_PROVIDER_JSON=$(jq -Rn --arg provider "$SEARCH_PROVIDER" '$provider')
SEARCH_CUDA_DIRECTORY_JSON=$(jq -Rn --arg directory "$SEARCH_CUDA_LIB_DIR" '$directory')
SEARCH_LIBRARY_PATH_JSON=$(jq -Rn --arg path "$SEARCH_LIBRARY_PATH" '$path')
cat > "$SEARCH_STAGE/plugin/ui/$SEARCH_VERSION/RuntimeConfig.qml" <<CONFIG
import QtQuick
QtObject {
    readonly property string socketPath: $SEARCH_SOCKET_JSON
    readonly property string bundleVersion: $SEARCH_VERSION_JSON
    readonly property string serviceName: "fractal-search.service"
    readonly property string pickerExecutable: $SEARCH_PICKER_JSON
    readonly property string backendExecutable: $SEARCH_BACKEND_JSON
    readonly property string configuredProvider: $SEARCH_PROVIDER_JSON
    readonly property int configuredDeviceId: ${SEARCH_DEVICE_ID:--1}
    readonly property string cudaLibraryDirectory: $SEARCH_CUDA_DIRECTORY_JSON
    readonly property string runtimeLibraryPath: $SEARCH_LIBRARY_PATH_JSON
    readonly property bool closeShortcutEnabled: $([[ $SEARCH_BIND_CLOSE == 1 ]] && echo true || echo false)
}
CONFIG
systemd_quote() {
  local SEARCH_VALUE=$1
  [[ $SEARCH_VALUE != *$'\n'* && $SEARCH_VALUE != *$'\r'* ]] || { echo 'Invalid newline in service option' >&2; return 1; }
  SEARCH_VALUE=${SEARCH_VALUE//\\/\\\\}; SEARCH_VALUE=${SEARCH_VALUE//\"/\\\"}; SEARCH_VALUE=${SEARCH_VALUE//%/%%}
  # Environment= expands specifiers but does not perform ExecStart's $ expansion.
  if [[ ${2:-exec} == exec ]]; then SEARCH_VALUE=${SEARCH_VALUE//\$/\$\$}; fi
  printf '"%s"' "$SEARCH_VALUE"
}
SEARCH_EXEC="$(systemd_quote "$SEARCH_DEST/bin/fractal-search") daemon --warm-search --data-dir $(systemd_quote "$SEARCH_DATA_HOME/fractal-search") --socket $(systemd_quote "$SEARCH_RUNTIME_DIR/fractal-search/backend.sock")"
SEARCH_EXEC+=" --provider $SEARCH_PROVIDER"
[[ -z $SEARCH_DEVICE_ID ]] || SEARCH_EXEC+=" --device-id $SEARCH_DEVICE_ID"
[[ -z $SEARCH_MODEL_DIR ]] || SEARCH_EXEC+=" --model-dir $(systemd_quote "$SEARCH_MODEL_DIR")"
if [[ -n $SEARCH_THREADS ]]; then
  [[ $SEARCH_THREADS =~ ^[1-9][0-9]*$ && $SEARCH_THREADS -le 64 ]] || { echo 'Threads must be 1..64' >&2; exit 1; }
  SEARCH_EXEC+=" --threads $SEARCH_THREADS"
fi
SEARCH_ENV_LINE=''
if [[ -n $SEARCH_LIBRARY_PATH ]]; then
  SEARCH_ENV_LINE="Environment=$(systemd_quote "LD_LIBRARY_PATH=$SEARCH_LIBRARY_PATH" environment)"
fi
cat > "$SEARCH_STAGE/fractal-search.service" <<UNIT
# Managed by Fractal Search installer
[Unit]
Description=Fractal Search native model and catalog service
PartOf=graphical-session.target
[Service]
Type=simple
ExecStart=$SEARCH_EXEC
$SEARCH_ENV_LINE
RuntimeDirectory=fractal-search
RuntimeDirectoryMode=0700
UMask=0077
KillMode=mixed
TimeoutStopSec=300
[Install]
WantedBy=graphical-session.target
UNIT
if [[ ! -e $SEARCH_DEST ]]; then
  mkdir -- "$SEARCH_DEST"
  cp -a -- "$SEARCH_BUNDLE/bin" "$SEARCH_BUNDLE/uninstall.sh" "$SEARCH_BUNDLE/README.md" "$SEARCH_BUNDLE/MCP.md" "$SEARCH_BUNDLE/CUDA.md" "$SEARCH_BUNDLE/LICENSE" "$SEARCH_BUNDLE/NOTICE" "$SEARCH_BUNDLE/LANCEDB-LICENSE" "$SEARCH_BUNDLE/third-party" "$SEARCH_BUNDLE/AGENTS.md" "$SEARCH_BUNDLE/AGENT-SETUP.md" "$SEARCH_BUNDLE/bundle.json" "$SEARCH_DEST/"
fi
systemd-analyze --user verify "$SEARCH_STAGE/fractal-search.service" 2> "$SEARCH_STAGE/unit-errors" || { cat "$SEARCH_STAGE/unit-errors" >&2; exit 1; }
if ((!SEARCH_SKIP_SHELL)); then
  # Stop finishes active inference and checkpoints before replacing the runtime.
  if [[ -e $SEARCH_UNIT ]]; then
    SEARCH_SERVICE_STATE=$(systemctl --user show fractal-search.service --property=ActiveState --value)
    case "$SEARCH_SERVICE_STATE" in
      active|activating|reloading) SEARCH_SERVICE_WAS_ACTIVE=1 ;;
      inactive|failed|deactivating) ;;
      *) echo "Cannot determine previous service activity: $SEARCH_SERVICE_STATE" >&2; exit 1 ;;
    esac
    SEARCH_SERVICE_STOPPED=1
    systemctl --user stop fractal-search.service
  fi
  omarchy-shell shell hide fractal.search
fi
[[ ! -e $SEARCH_PLUGIN ]] || mv -- "$SEARCH_PLUGIN" "$SEARCH_OLD_PLUGIN"
[[ ! -e $SEARCH_UNIT ]] || cp -- "$SEARCH_UNIT" "$SEARCH_OLD_UNIT"
SEARCH_APPLIED=1
mv -- "$SEARCH_STAGE/plugin" "$SEARCH_PLUGIN"
cp -- "$SEARCH_STAGE/fractal-search.service" "$SEARCH_UNIT"
if ((SEARCH_BIND_CLOSE)); then
  [[ ! -e $SEARCH_BINDINGS ]] || cp -p -- "$SEARCH_BINDINGS" "$SEARCH_OLD_BINDINGS"
  SEARCH_BINDING_APPLIED=1
  "$SEARCH_BUNDLE/close-binding.sh" install "$SEARCH_CONFIG_HOME" "$SEARCH_PLUGIN"
  if ((!SEARCH_SKIP_SHELL)); then
    hyprctl reload
    SEARCH_CONFIG_ERRORS=$(hyprctl configerrors)
    [[ -z $SEARCH_CONFIG_ERRORS ]] || { echo "Hyprland configuration errors: $SEARCH_CONFIG_ERRORS" >&2; exit 1; }
  fi
fi
if ((!SEARCH_SKIP_SHELL)); then
  systemctl --user daemon-reload
  omarchy-shell shell rescanPlugins
  omarchy plugin enable fractal.search
  # Both file discovery and enabling schedule asynchronous host reloads. Wait
  # for this exact installed version to remain loaded through the debounce.
  SEARCH_STABLE=0
  for ((SEARCH_ATTEMPT=0; SEARCH_ATTEMPT<50; SEARCH_ATTEMPT++)); do
    SEARCH_HEALTH=$(omarchy-shell shell call fractal.search ping '')
    if [[ $SEARCH_HEALTH == "$SEARCH_VERSION" ]]; then
      ((SEARCH_STABLE+=1))
      ((SEARCH_STABLE >= 6)) && break
    else
      SEARCH_STABLE=0
    fi
    sleep 0.2
  done
  ((SEARCH_STABLE >= 6)) || { echo "The installed panel did not load. Check quickshell log for Omarchy; rolling back." >&2; exit 1; }
  if ((SEARCH_SERVICE_WAS_ACTIVE)); then
    systemctl --user start fractal-search.service
  fi
fi
SEARCH_DONE=1
# Cleanup errors must not roll back a successfully installed plugin.
"$SEARCH_BUNDLE/prune-runtimes.sh" "$SEARCH_RUNTIME" "$SEARCH_DEST" "$SEARCH_PREVIOUS" "$SEARCH_CUDA_LIB_DIR" || {
  echo 'Installation succeeded, but old-build cleanup failed. Fix the reported error and rerun the installer.' >&2
  exit 1
}
echo "Installed Fractal Search $SEARCH_VERSION"
echo 'Open: omarchy-shell shell toggle fractal.search'
echo 'Models: use Set up model on first open; verified cached files are reused.'
echo "Uninstall (preserves catalog/models): $SEARCH_DEST/uninstall.sh"
echo "Read-only MCP command: $SEARCH_PLUGIN/fractal-search-mcp"
