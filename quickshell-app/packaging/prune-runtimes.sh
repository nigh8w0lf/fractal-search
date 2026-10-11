#!/usr/bin/env bash
# Called under the installer's exclusive lock, only after installation commits.
set -euo pipefail
SEARCH_ROOT=$(realpath -e -- "${1:?runtime directory required}")
SEARCH_CURRENT=$(realpath -e -- "${2:?current build required}")
SEARCH_PREVIOUS=${3:-}
SEARCH_CUDA=${4:-}
[[ $SEARCH_CURRENT == "$SEARCH_ROOT/"* && ${SEARCH_CURRENT#"$SEARCH_ROOT/"} != */* ]] || { echo 'Current build is outside the runtime directory' >&2; exit 1; }
# Reinstalling the same build must retain the preceding version too.
if [[ $SEARCH_PREVIOUS == "$SEARCH_CURRENT" && -f $SEARCH_ROOT/.previous-build ]]; then
  SEARCH_PREVIOUS=$(cat -- "$SEARCH_ROOT/.previous-build")
fi
printf '%s\n' "$SEARCH_PREVIOUS" > "$SEARCH_ROOT/.previous-build"
SEARCH_RUNNING=()
for SEARCH_PROC in /proc/[0-9]*; do
  [[ -O $SEARCH_PROC ]] || continue
  # Unrelated sandboxed desktop processes can deny access to their executable.
  # Linux comm truncates the bundled executable names to 15 characters.
  if ! IFS= read -r SEARCH_COMM < "$SEARCH_PROC/comm"; then
    [[ ! -d $SEARCH_PROC ]] && continue
    echo "Cannot inspect $SEARCH_PROC/comm; refusing to prune runtimes." >&2
    exit 1
  fi
  [[ $SEARCH_COMM == fractal-* ]] || continue
  if SEARCH_EXE=$(readlink -- "$SEARCH_PROC/exe"); then
    SEARCH_RUNNING+=("$SEARCH_EXE")
  elif [[ -d $SEARCH_PROC ]] && [[ -n $(tr -d '\0' < "$SEARCH_PROC/cmdline") ]]; then
    echo "Cannot inspect $SEARCH_PROC/exe; refusing to prune runtimes." >&2
    exit 1
  fi
done
SEARCH_COUNT=0
for SEARCH_BUILD in "$SEARCH_ROOT"/*; do
  [[ -d $SEARCH_BUILD && ! -L $SEARCH_BUILD ]] || continue
  SEARCH_NAME=${SEARCH_BUILD##*/}
  [[ $SEARCH_NAME =~ ^[0-9]+\.[0-9]+\.[0-9]+-[a-f0-9]{12}(-close)?(-cpu|-cuda-[0-9]+-[a-f0-9]{8})?$ ]] || continue
  [[ -f $SEARCH_BUILD/bundle.json && -f $SEARCH_BUILD/bin/fractal-search ]] || continue
  [[ $SEARCH_BUILD != "$SEARCH_CURRENT" && $SEARCH_BUILD != "$SEARCH_PREVIOUS" ]] || continue
  [[ $SEARCH_CUDA != "$SEARCH_BUILD" && $SEARCH_CUDA != "$SEARCH_BUILD/"* ]] || continue
  SEARCH_BUSY=0
  for SEARCH_EXE in "${SEARCH_RUNNING[@]}"; do
    if [[ $SEARCH_EXE == "$SEARCH_BUILD/"* ]]; then SEARCH_BUSY=1; break; fi
  done
  if ((SEARCH_BUSY)); then
    echo "Retaining in-use runtime: $SEARCH_BUILD"
    continue
  fi
  rm -rf --one-file-system -- "$SEARCH_BUILD"
  ((SEARCH_COUNT+=1))
done
echo "Removed $SEARCH_COUNT obsolete Fractal Search runtime build(s)."
