#!/usr/bin/env bash
set -euo pipefail
SEARCH_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
SEARCH_PLUGIN="$SEARCH_CONFIG_HOME/omarchy/plugins/fractal.search"
SEARCH_UNIT="$SEARCH_CONFIG_HOME/systemd/user/fractal-search.service"
[[ -f $SEARCH_PLUGIN/.fractal-search-install ]] || { echo 'No managed Fractal Search installation found' >&2; exit 1; }
awk '/^# Managed by Fractal Search installer$/ {found=1} END {exit !found}' "$SEARCH_UNIT" || { echo 'Refusing to remove an unmanaged service' >&2; exit 1; }
systemctl --user stop fractal-search.service
systemctl --user disable fractal-search.service
if [[ -f $SEARCH_CONFIG_HOME/hypr/bindings.lua ]] && awk '$0 == "-- BEGIN Fractal Search close binding" {found=1} END {exit !found}' "$SEARCH_CONFIG_HOME/hypr/bindings.lua"; then
  "$SEARCH_PLUGIN/close-binding.sh" remove "$SEARCH_CONFIG_HOME" "$SEARCH_PLUGIN"
  hyprctl reload
  SEARCH_CONFIG_ERRORS=$(hyprctl configerrors)
  [[ -z $SEARCH_CONFIG_ERRORS ]] || { echo "Hyprland configuration errors: $SEARCH_CONFIG_ERRORS" >&2; exit 1; }
fi
omarchy plugin disable fractal.search
rm -rf -- "$SEARCH_PLUGIN"
rm -- "$SEARCH_UNIT"
systemctl --user daemon-reload
omarchy-shell shell rescanPlugins
echo 'Fractal Search removed. Catalog, model cache and runtime bundles are preserved.'
