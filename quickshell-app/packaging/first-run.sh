#!/usr/bin/env bash
# First-run bootstrap for the git-installed panel. No compiler or Python required.
set -euo pipefail
SEARCH_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/fractal-search/setup"
SEARCH_PLUGIN="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins/fractal.search"
SEARCH_ACTION=${1:?Expected launch, status or worker}; shift
umask 077
mkdir -p -- "$SEARCH_STATE"
chmod 700 -- "$SEARCH_STATE"
status() {
  local temp
  temp=$(mktemp "$SEARCH_STATE/status.XXXXXX")
  jq -n --arg stage "$1" --arg message "$2" '{stage:$stage,message:$message}' > "$temp"
  mv -- "$temp" "$SEARCH_STATE/status.json"
}
validate_source() {
  [[ $SEARCH_SOURCE == "$SEARCH_PLUGIN" && -d $SEARCH_SOURCE/.git && ! -L $SEARCH_SOURCE ]] || { echo 'Setup must run from the git-installed Fractal Search plugin.' >&2; return 1; }
  [[ ! -e $SEARCH_SOURCE/.fractal-search-install ]] || { echo 'A managed installation already exists; use its release installer to update.' >&2; return 1; }
  local remote upstream
  remote=$(git -C "$SEARCH_SOURCE" remote get-url origin)
  case "$remote" in
    https://github.com/nigh8w0lf/fractal-search|https://github.com/nigh8w0lf/fractal-search.git) ;;
    /*)
      [[ -d $remote/.git ]] || { echo 'Local origin must be a source checkout.' >&2; return 1; }
      upstream=$(git -C "$remote" remote get-url origin)
      [[ $upstream == https://github.com/nigh8w0lf/fractal-search || $upstream == https://github.com/nigh8w0lf/fractal-search.git ]] || { echo 'Unexpected local source upstream; refusing replacement.' >&2; return 1; }
      [[ $(git -C "$remote" rev-parse HEAD) == $(git -C "$SEARCH_SOURCE" rev-parse HEAD) ]] || { echo 'Local source HEAD changed; reinstall its current committed clone before setup.' >&2; return 1; }
      ;;
    *) echo 'Unexpected plugin repository; refusing replacement.' >&2; return 1 ;;
  esac
  [[ -z $(git -C "$SEARCH_SOURCE" status --porcelain --untracked-files=all) ]] || { echo 'This plugin has local changes. Save them elsewhere before installing the release.' >&2; return 1; }
  [[ $(jq -r .id "$SEARCH_SOURCE/manifest.json") == fractal.search ]] || { echo 'Unexpected plugin identity; refusing replacement.' >&2; return 1; }
}
validate_cuda_directory() {
  [[ -z $SEARCH_CUDA_DIR ]] || {
    [[ $SEARCH_PROVIDER == cuda && $SEARCH_CUDA_DIR == /* && -d $SEARCH_CUDA_DIR && $SEARCH_CUDA_DIR != *:* && $SEARCH_CUDA_DIR != *$'\n'* && $SEARCH_CUDA_DIR != *$'\r'* ]] || { echo 'CUDA library path must be one existing absolute directory without colons or newlines, and requires CUDA.' >&2; return 1; }
  }
}
case "$SEARCH_ACTION" in
  status)
    if [[ -f $SEARCH_STATE/status.json ]]; then
      stage=$(jq -er .stage "$SEARCH_STATE/status.json")
      if [[ $stage == starting || $stage == downloading || $stage == verifying || $stage == installing ]]; then
        age=$(( $(date +%s) - $(stat -c %Y "$SEARCH_STATE/status.json") ))
        active=$(systemctl --user show fractal-search-setup.service --property=ActiveState --value)
        if ((age > 15)) && [[ $active != active && $active != activating ]]; then
          status error "Setup stopped unexpectedly. See $SEARCH_STATE/install.log, then retry."
        fi
      fi
      jq -e 'if type == "object" and (.stage | type == "string") and (.message | type == "string") then . else error("Invalid setup status") end' "$SEARCH_STATE/status.json"
    else
      printf '%s\n' '{"stage":"idle","message":"Choose a backend to get started."}'
    fi
    exit ;;
  launch)
    SEARCH_PROVIDER=${1:?Provider required}; SEARCH_SOURCE=${2:?Source path required}; SEARCH_CUDA_DIR=${3:-}
    validate_cuda_directory
    [[ $SEARCH_PROVIDER == cpu || $SEARCH_PROVIDER == cuda ]] || exit 1
    for tool in jq curl tar sha256sum git systemd-run systemctl flock; do
      command -v "$tool" >/dev/null || { echo "Missing setup dependency: $tool" >&2; exit 1; }
    done
    validate_source
    exec {SEARCH_LOCK}>"$SEARCH_STATE/launch.lock"
    flock -n "$SEARCH_LOCK" || { echo 'Setup launch is already in progress.' >&2; exit 1; }
    active=$(systemctl --user show fractal-search-setup.service --property=ActiveState --value)
    [[ $active != active && $active != activating && $active != deactivating ]] || { echo 'Setup is already running.' >&2; exit 1; }
    # Execute outside the plugin tree; replacing/reloading the UI must not kill setup.
    cp -- "${BASH_SOURCE[0]}" "$SEARCH_STATE/worker.sh"
    status starting 'Starting the release installer…'
    SEARCH_ENV=()
    for key in HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME XDG_CACHE_HOME XDG_RUNTIME_DIR OMARCHY_PATH WAYLAND_DISPLAY DBUS_SESSION_BUS_ADDRESS; do
      [[ ! -v $key ]] || SEARCH_ENV+=("--setenv=$key=${!key}")
    done
    if ! systemd-run "${SEARCH_ENV[@]}" --user --collect --unit=fractal-search-setup --property=Type=exec -- /bin/bash "$SEARCH_STATE/worker.sh" worker "$SEARCH_PROVIDER" "$SEARCH_SOURCE" "$SEARCH_CUDA_DIR"; then
      status error 'Could not start setup. Check the user systemd service and try again.'
      exit 1
    fi
    exit ;;
  worker) SEARCH_PROVIDER=${1:?}; SEARCH_SOURCE=${2:?}; SEARCH_CUDA_DIR=${3:-} ;;
  *) echo 'Unknown setup action' >&2; exit 1 ;;
esac
exec {SEARCH_LOCK}>"$SEARCH_STATE/worker.lock"
flock -n "$SEARCH_LOCK" || exit 1
exec >"$SEARCH_STATE/install.log" 2>&1
SEARCH_STAGE=''; SEARCH_BACKUP=''; SEARCH_SUCCESS=0
cleanup() {
  code=$?
  trap - EXIT
  if (( !SEARCH_SUCCESS )); then
    if [[ -n $SEARCH_BACKUP && -d $SEARCH_BACKUP ]]; then
      if [[ ! -e $SEARCH_PLUGIN ]]; then
        mv -- "$SEARCH_BACKUP" "$SEARCH_PLUGIN" || echo 'ERROR: Failed to restore source plugin.'
      else
        echo "Recovery copy retained at $SEARCH_BACKUP; plugin path already exists."
      fi
    fi
    message=$(tail -c 6000 "$SEARCH_STATE/install.log")
    status error "Setup failed. $message"
  fi
  [[ -z $SEARCH_STAGE ]] || rm -rf -- "$SEARCH_STAGE"
  exit "$code"
}
trap cleanup EXIT
[[ $(uname -m) == x86_64 ]] || { echo 'Prebuilt releases currently require x86_64 Linux.'; exit 1; }
validate_source
validate_cuda_directory
case "$SEARCH_PROVIDER" in
  cpu) SEARCH_HASH=61d7fad6a5195b2a0ec73d2db57987f3c6e952632f95647ccc39b3cc9e2be300 ;;
  cuda) SEARCH_HASH=6b77f0bdeb75963fac4cf437761d5de599959d9ccb5a247cf9268a98aff5b0f5 ;;
  *) echo 'Invalid provider'; exit 1 ;;
esac
SEARCH_NAME="fractal-search-0.1.2-linux-x86_64-$SEARCH_PROVIDER"
SEARCH_STAGE=$(mktemp -d "$SEARCH_STATE/download.XXXXXX")
status downloading "Downloading the v0.1.2 $SEARCH_PROVIDER backend…"
curl --fail --location --proto '=https' --proto-redir '=https' --connect-timeout 30 --max-time 1800 --max-filesize 536870912 --output "$SEARCH_STAGE/bundle.tar.gz" "https://github.com/nigh8w0lf/fractal-search/releases/download/v0.1.2/$SEARCH_NAME.tar.gz"
status verifying 'Verifying the pinned release checksum…'
printf '%s  %s\n' "$SEARCH_HASH" "$SEARCH_STAGE/bundle.tar.gz" | sha256sum --check --status || { echo 'Release checksum mismatch. Nothing was installed; retry or report the damaged download.'; exit 1; }
# The pinned archive is immutable. Also reject traversal, links and special files.
tar -tzf "$SEARCH_STAGE/bundle.tar.gz" > "$SEARCH_STAGE/members"
awk -v prefix="$SEARCH_NAME/" 'index($0,prefix)!=1 || $0 ~ /(^|\/)\.\.(\/|$)/ {bad=1} END {exit bad}' "$SEARCH_STAGE/members"
tar -tvzf "$SEARCH_STAGE/bundle.tar.gz" | awk 'substr($0,1,1)!="-" && substr($0,1,1)!="d" {bad=1} END {exit bad}'
tar --extract --gzip --no-same-owner --no-same-permissions --file "$SEARCH_STAGE/bundle.tar.gz" --directory "$SEARCH_STAGE"
(cd "$SEARCH_STAGE/$SEARCH_NAME"; sha256sum --check --quiet SHA256SUMS)
validate_source
status installing 'Installing the backend and desktop integration…'
# Preserve the clone outside the watched plugin tree. The release installer owns
# managed-install rollback; this outer transaction restores the initial clone.
SEARCH_BACKUP="$SEARCH_STATE/source-backup-$(date +%s)-$$"
mv -- "$SEARCH_SOURCE" "$SEARCH_BACKUP"
SEARCH_INSTALL_ARGS=(--provider "$SEARCH_PROVIDER")
[[ -z $SEARCH_CUDA_DIR ]] || SEARCH_INSTALL_ARGS+=(--cuda-lib-dir "$SEARCH_CUDA_DIR")
OMARCHY_SHELL_IPC_TIMEOUT=15s "$SEARCH_STAGE/$SEARCH_NAME/install.sh" "${SEARCH_INSTALL_ARGS[@]}"
[[ -f $SEARCH_PLUGIN/.fractal-search-install ]] || { echo 'Installer did not create its managed installation marker.'; exit 1; }
SEARCH_SUCCESS=1
status done 'App installed. Open Fractal Search, select Set up model, then add your folders.'
# Keep the source backup for recovery; it is not a second active installation.
printf 'Source checkout retained at %s\n' "$SEARCH_BACKUP"
