#!/usr/bin/env bash
# Called explicitly by the installer/uninstaller; preserves personal bindings.
set -euo pipefail
[[ $# == 3 && ($1 == install || $1 == remove) ]] || { echo 'Usage: close-binding.sh install|remove CONFIG_HOME PLUGIN_DIR' >&2; exit 1; }
SEARCH_ACTION=$1
SEARCH_BINDINGS="$2/hypr/bindings.lua"
SEARCH_PLUGIN=$3
SEARCH_BEGIN='-- BEGIN Fractal Search close binding'
SEARCH_END='-- END Fractal Search close binding'
[[ $SEARCH_BINDINGS == /* && $SEARCH_PLUGIN == /* ]] || { echo 'Binding paths must be absolute' >&2; exit 1; }
mkdir -p -- "$(dirname -- "$SEARCH_BINDINGS")"
SEARCH_TEMP=$(mktemp "${SEARCH_BINDINGS}.XXXXXX")
trap 'rm -f -- "$SEARCH_TEMP"' EXIT
if [[ -e $SEARCH_BINDINGS ]]; then
  # Refuse malformed/duplicate markers before rewriting any user file.
  awk -v begin="$SEARCH_BEGIN" -v end="$SEARCH_END" '
    $0 == begin { if (inside || seen++) exit 1; inside=1; next }
    $0 == end { if (!inside) exit 1; inside=0; next }
    !inside { print }
    END { if (inside) exit 1 }
  ' "$SEARCH_BINDINGS" > "$SEARCH_TEMP" || { echo 'Malformed Fractal close-binding markers; bindings were not changed' >&2; exit 1; }
fi
if [[ $SEARCH_ACTION == install ]]; then
  # Do not silently replace a personal Super+W action.
  if awk '!/^[[:space:]]*--/ && /[Ss][Uu][Pp][Ee][Rr][[:space:]]*\+[[:space:]]*[Ww]["\047]/ {found=1} END {exit !found}' "$SEARCH_TEMP"; then
    echo 'Super+W has a personal override. Resolve it before installing the Fractal close binding.' >&2
    exit 1
  fi
  SEARCH_COMMAND=$(printf "'%s'" "${SEARCH_PLUGIN//\'/\'\\\'\'}/close-window.sh")
  SEARCH_COMMAND_JSON=$(jq -Rn --arg command "$SEARCH_COMMAND" '$command')
  {
    printf '%s\n' "$SEARCH_BEGIN"
    printf '%s\n' '-- Super+W was Omarchy Close window; the router preserves that action outside Fractal.'
    printf '%s\n' 'hl.unbind("SUPER + W")'
    printf 'o.bind("SUPER + W", "Close window or Fractal panel", %s)\n' "$SEARCH_COMMAND_JSON"
    printf '%s\n' "$SEARCH_END"
  } >> "$SEARCH_TEMP"
fi
if [[ -e $SEARCH_BINDINGS ]]; then
  cmp -s -- "$SEARCH_BINDINGS" "$SEARCH_TEMP" && exit 0
  cp -p -- "$SEARCH_BINDINGS" "${SEARCH_BINDINGS}.bak.fractal-close.$(date +%Y%m%dT%H%M%S).$$"
  chmod --reference="$SEARCH_BINDINGS" "$SEARCH_TEMP"
fi
mv -- "$SEARCH_TEMP" "$SEARCH_BINDINGS"
