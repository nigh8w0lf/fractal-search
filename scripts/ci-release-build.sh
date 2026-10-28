#!/usr/bin/env bash
# Executed only inside the pinned CI container; no desktop or model access.
set -euo pipefail
[[ ${SEARCH_VARIANT:?} == cpu || $SEARCH_VARIANT == cuda ]]
[[ ${SEARCH_BUILD_IMAGE:?} == archlinux@sha256:* ]]
pacman -Syu --noconfirm --needed base-devel git clang cmake pkgconf protobuf libheif libjpeg-turbo openssl python jq rustup ffmpeg glib2
rustup toolchain install 1.99.1 --profile minimal --component rust-docs,rustfmt
rustup default 1.99.1
git config --global --add safe.directory /src
[[ -z $(git status --porcelain) ]] || { echo 'Release source must be clean' >&2; exit 1; }
version=$(jq -er .version manifest.json)
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
features=()
package_args=()
if [[ $SEARCH_VARIANT == cuda ]]; then
  features=(--features fractal-search/cuda,fractal-search-mcp/cuda)
  package_args=(--cuda)
fi
export CARGO_BUILD_JOBS=2 ORT_CUDA_VERSION=12
cargo fmt --all --check
cargo fetch --locked
cargo build --release --locked -p fractal-search -p fractal-folder-picker -p fractal-search-mcp --bins "${features[@]}"
cargo test --release --workspace --locked "${features[@]}"
python3 scripts/test_embeddinggemma2_contract.py
python3 quickshell-app/tests/smoke_first_run.py
python3 scripts/fetch-release-sources.py
name="fractal-search-$version-linux-x86_64-$SEARCH_VARIANT"
bundle="$PWD/target/$name"
bash scripts/package-fractal-search.sh --no-build "${package_args[@]}" --output "$bundle"
jq -e --arg sha "$(git rev-parse HEAD)" '.sourceCommit == $sha and .sourceDirty == false' "$bundle/bundle.json"
(cd "$bundle"; sha256sum --check --quiet SHA256SUMS)
mkdir -p dist/ci
pacman -Q > "dist/ci/$SEARCH_VARIANT-packages.txt"
jq -n --arg image "$SEARCH_BUILD_IMAGE" --arg source "$(git rev-parse HEAD)" \
  --arg variant "$SEARCH_VARIANT" --arg rust "$(rustc --version)" \
  '{sourceCommit:$source,variant:$variant,container:$image,rustc:$rust,checks:["cargo fmt","locked release build","release workspace tests","model-free contract","bootstrap fixtures","bundle provider reports","bundle checksums"],limitations:["No GPU or model inference in hosted CI","No native Omarchy UI in build container","Native installer and inference tested separately before publication"]}' > "dist/ci/$SEARCH_VARIANT-build.json"
tar -czf "dist/ci/$name.tar.gz" -C target "$name"
