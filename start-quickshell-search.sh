#!/usr/bin/env bash
set -euo pipefail
SEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_BUILD=1
SEARCH_DEV=0
SEARCH_SUMMON=0
SEARCH_CUDA=0
SEARCH_OPTIONS=()
while (($#)); do
  case "$1" in
    --dev) SEARCH_DEV=1; shift ;;
    --no-build) SEARCH_BUILD=0; shift ;;
    --summon) SEARCH_SUMMON=1; shift ;;
    --bind-close) SEARCH_OPTIONS+=(--bind-close); shift ;;
    --cuda) SEARCH_CUDA=1; shift ;;
    --model-dir|--threads|--provider|--device-id|--cuda-lib-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      [[ $1 != --provider || $2 != cuda ]] || SEARCH_CUDA=1
      SEARCH_OPTIONS+=("$1" "$2"); shift 2 ;;
    --help|-h)
      echo 'Usage: ./start-quickshell-search.sh [--dev] [--no-build] [--summon] [--bind-close] [--cuda] [--provider cpu|cuda] [--device-id N] [--cuda-lib-dir PATH] [--model-dir PATH] [--threads N]'
      echo 'Builds/installs the Omarchy plugin and opens it. --no-build --summon only opens the installed plugin.'
      exit 0 ;;
    *) echo "Unknown argument: $1. Temporary data/socket options belong to the isolated developer harness." >&2; exit 1 ;;
  esac
done
command -v omarchy-shell >/dev/null || { echo 'This plugin requires Omarchy Shell' >&2; exit 1; }
if ! ((SEARCH_SUMMON && !SEARCH_BUILD)) || ((SEARCH_CUDA || ${#SEARCH_OPTIONS[@]})); then
  SEARCH_OUTPUT=$(mktemp -d /tmp/fractal-search-bundle.XXXXXX)
  rmdir -- "$SEARCH_OUTPUT"
  trap 'rm -rf -- "$SEARCH_OUTPUT"' EXIT
  SEARCH_ARGS=(--output "$SEARCH_OUTPUT")
  ((SEARCH_DEV)) && SEARCH_ARGS+=(--dev)
  ((SEARCH_BUILD)) || SEARCH_ARGS+=(--no-build)
  ((SEARCH_CUDA)) && SEARCH_ARGS+=(--cuda)
  "$SEARCH_ROOT/scripts/package-fractal-search.sh" "${SEARCH_ARGS[@]}"
  "$SEARCH_OUTPUT/install.sh" "${SEARCH_OPTIONS[@]}"
fi
SEARCH_RESPONSE=$(omarchy-shell shell summon fractal.search '{}')
[[ $SEARCH_RESPONSE == ok ]] || { echo "Cannot open fractal.search: $SEARCH_RESPONSE. Install it with ./start-quickshell-search.sh --dev" >&2; exit 1; }
