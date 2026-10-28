Fractal Search v0.1.2 adds GitHub-hosted builds and signed build provenance for the CPU and CUDA bundles. It includes the first-run setup panel in source installs. Native search, model and catalog contracts are unchanged; no reindex is required.

The CPU and CUDA archive attestations bind the exact downloadable bytes to the source commit and `.github/workflows/release.yml`. Verify each archive using:

```bash
gh attestation verify fractal-search-0.1.2-linux-x86_64-cpu.tar.gz --repo nigh8w0lf/fractal-search --signer-workflow nigh8w0lf/fractal-search/.github/workflows/release.yml --deny-self-hosted-runners
gh attestation verify fractal-search-0.1.2-linux-x86_64-cuda.tar.gz --repo nigh8w0lf/fractal-search --signer-workflow nigh8w0lf/fractal-search/.github/workflows/release.yml --deny-self-hosted-runners
```

Use `--source-digest` with the release's exact source commit to enforce that binding. Portable Sigstore bundles are included. SHA256SUMS alone is not proof of origin. Source archive checksums are supplied separately from the executable archive attestations.

Requires current x86_64 Omarchy and the documented native dependencies. CUDA additionally requires a compatible NVIDIA driver, CUDA 12 and cuDNN 9. Those libraries and model weights are not bundled. All covered-source and third-party notices remain included.

GitHub CI records its package inventory, toolchain and checks with each variant. GPU inference and native desktop interactions are not claimed by the hosted build tests. See the release's additional local validation report for testing on Omarchy.

Existing v0.1.0 and v0.1.1 assets remain unchanged. Install with the bundle's `install.sh`; preserve the current provider/library configuration when updating.
