# LanceDB local patch

This is the crates.io LanceDB 0.37.1 source snapshot, Apache-2.0 licensed.

- Upstream repository: https://github.com/lancedb/lancedb
- Upstream commit: `b89f87f206abd79a069833cee01f0c029ddb1397`
- Source subtree: `rust/lancedb`
- Published crate SHA-256: `50c1fb7acca1d0f9303ab57fe417bb6ec5bbb4f3752f1c677ad2f6fec8ddaf89`
- License: copied from that exact upstream commit into `LICENSE`

The only change to the published implementation/manifest is in `Cargo.toml`:
the `lance-index` dependency no longer requests `tokenizer-jieba` and
`tokenizer-lindera`, and explicitly disables default features. All Rust source,
examples and tests are unchanged. `Cargo.toml.upstream` preserves the published,
normalized manifest; `Cargo.toml.orig` is the published pre-normalization source.
Registry bookkeeping and the crate's standalone lockfile are excluded.
`UPSTREAM-SHA256.json` records the unchanged Rust source, examples and tests for
verification without depending on a developer's Cargo cache.

Fractal Search uses `icu/split`, lowercase and ASCII folding, with stemming and
stop-word removal disabled. ICU segmentation and the existing catalog/index
contract remain unchanged. Unsupported tokenizer choices must error; never
substitute another tokenizer. No catalog migration or reindexing is introduced.

The root `[patch.crates-io]` selects this copy; `vendor/lancedb` is excluded from
the workspace. Refresh from the published crate on upgrades, preserve attribution,
reapply only this manifest change, and run the dependency-layout check plus
CPU/CUDA workspace tests and multilingual hybrid-search smoke checks. Do not
edit runtime Rust source here as part of a dependency refresh.

A local path dependency exposes three upstream dead-code warnings in the local-only
build (`Job::new_done`, `AnalyzePlanDistributedMetrics::as_query_param`, and
`GetLsmStatsResponse`). They are not suppressed or fixed by changing upstream
Rust source. Cargo normally caps warnings from registry dependencies.
