# Documentation

Start with the [project overview](../README.md).

## Using Fractal Search

- [User guide](USER-GUIDE.md): installation, search, settings, supported files, updates, removal and troubleshooting.
- [Agent setup guide](AGENT-SETUP.md): help an agent install and diagnose a prebuilt bundle.
- [CUDA](../quickshell-app/CUDA.md): NVIDIA dependencies, provider selection and diagnostics.
- [MCP](../quickshell-app/MCP.md): connect an agent client and understand the read-only privacy boundary.

## Developing and releasing

- [Development guide](DEVELOPMENT.md): source builds, architecture, source map and catalog format.
- [Contributor instructions](../AGENTS.md): engineering constraints and validation commands.
- [Plugin-specific guidance](../quickshell-app/AGENTS.md): desktop UI and service invariants.
- [Release preparation](RELEASING.md): CPU/CUDA bundles, third-party attribution and marketplace publication.
- [Release readiness](RELEASE-READINESS.md): release checks, compatibility and marketplace follow-up.
- [Marketplace submission](MARKETPLACE-SUBMISSION.md): listing details, preview provenance and review status.
- [Dependency review](DEPENDENCIES.md): workspace separation and build-footprint decisions.

## Technical references

- [Models](MODELS.md): pinned model assets, embedding contracts and reference fixtures.
- [Indexing workers](INDEXING-WORKERS.md): bounded preparation, resource limits and failure handling.
- [Hybrid text search](HYBRID-TEXT-SEARCH.md): lexical/semantic retrieval and catalog contracts.
- [QMD-inspired improvements](QMD-SEARCH-IMPROVEMENTS.md): extraction, reuse, grouping, idle policies and limitations.
- [Retrieval evaluation](QMD-EVALUATION.md): labeled quality evaluation workflow.

## Validation history and research

These reports describe particular revisions and workloads, not guarantees for the current checkout. Run the relevant contributor checks for new changes. The plugin and CUDA references also contain dated validation sections.

- [Plugin implementation notes and validation history](../quickshell-app/README.md)
- [FP16 validation](FP16-VALIDATION.md)
- [Earlier 512D validation](EMBEDDING-512-VALIDATION.md)
- [QMD validation](QMD-VALIDATION.md)
- [Local QMD integration checkpoint](QMD-LOCAL-INTEGRATION.md)
- [Hybrid search investigation](HYBRID-SEARCH-INVESTIGATION.mkd)
- [ColBERT assessment](COLBERT-ASSESSMENT.md)
