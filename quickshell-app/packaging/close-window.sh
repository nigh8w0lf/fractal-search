#!/usr/bin/env bash
# Omarchy's Super+W route: dismiss our focused surface or close the original app.
set -euo pipefail
fail() {
  echo "Fractal Search close shortcut: $*" >&2
  notify-send -u critical -- 'Fractal Search close shortcut failed' "$*" >&2
  exit 1
}
SEARCH_ACTIVE=$(hyprctl activewindow -j) || fail 'Cannot read the active window. No window was closed.'
SEARCH_PAYLOAD=$(jq -ce '{title:(.title // ""),address:(.address // "")} | select((.title|type)=="string" and (.address|type)=="string")' <<< "$SEARCH_ACTIVE") || fail 'Invalid active-window response. No window was closed.'
SEARCH_RESPONSE=$(omarchy-shell shell call fractal.search closeFocused "$SEARCH_PAYLOAD") || fail 'Omarchy Shell did not answer. No window was closed; use Esc to hide Fractal.'
case "$SEARCH_RESPONSE" in
  handled) exit 0 ;;
  pass) ;;
  unknown)
    # A disabled/unloaded plugin has no surface. An old/missing method while
    # its overlay exists is an integration error, never permission to close
    # the application underneath it.
    SEARCH_LAYERS=$(hyprctl layers -j) || fail 'Cannot verify the unloaded plugin. No window was closed.'
    SEARCH_PRESENT=$(jq -er '[.. | objects | select(.namespace? == "fractal-search")] | length > 0' <<< "$SEARCH_LAYERS") || {
      [[ $SEARCH_PRESENT == false ]] || fail 'Invalid layer response. No window was closed.'
    }
    [[ $SEARCH_PRESENT == false ]] || fail 'Fractal is visible but cannot handle Close. Update the plugin; no window was closed.'
    ;;
  *) fail "Unexpected plugin response: $SEARCH_RESPONSE. No window was closed." ;;
esac
# Address the window captured before IPC, so a focus change during the call
# cannot close a different application. An empty desktop has nothing to close.
SEARCH_ADDRESS=$(jq -r .address <<< "$SEARCH_PAYLOAD")
[[ -n $SEARCH_ADDRESS ]] || exit 0
[[ $SEARCH_ADDRESS =~ ^0x[[:xdigit:]]+$ ]] || fail 'Invalid window address. No window was closed.'
hyprctl dispatch "hl.dsp.window.close({window=\"address:$SEARCH_ADDRESS\"})" || fail 'Hyprland could not close the selected window.'
