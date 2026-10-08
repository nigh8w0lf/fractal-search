//! Real CPU text-session idle release/reload parity, using a verified model cache.
//! cargo run --locked -p fractal-search-core --example validate_qmd_lifecycle -- MODEL_DIR [THREADS]
//! This is session-lifecycle validation, not reference-model or retrieval quality.

#[cfg(not(feature = "runtime"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("The validate_qmd_lifecycle example requires the runtime feature")
}

#[cfg(feature = "runtime")]
fn main() -> anyhow::Result<()> {
    use anyhow::{ensure, Context};
    use fractal_search_core::{
        runtime::{IdlePolicy, NativeRuntime, RuntimeDiagnostics, RuntimeIdlePolicy},
        EMBEDDING_DIM, MODEL_PROFILE,
    };
    use serde_json::json;
    use std::{path::PathBuf, thread, time::Duration, time::Instant};

    fn assert_lazy_media(state: &RuntimeDiagnostics) -> anyhow::Result<()> {
        for media in [&state.vision, &state.audio] {
            ensure!(
                !media.loaded && media.load_count == 0 && media.release_count == 0,
                "Text lifecycle validation unexpectedly touched a media encoder"
            );
        }
        Ok(())
    }

    fn norm(vector: &[f32]) -> anyhow::Result<f64> {
        ensure!(
            vector.len() == EMBEDDING_DIM,
            "Embedding dimension mismatch"
        );
        ensure!(
            vector.iter().all(|value| value.is_finite()),
            "Embedding contains non-finite values"
        );
        let norm = vector
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        ensure!((norm - 1.0).abs() <= 1e-5, "Embedding is not normalized");
        Ok(norm)
    }

    fn parity(before: &[f32], after: &[f32]) -> anyhow::Result<serde_json::Value> {
        let before_norm = norm(before)?;
        let after_norm = norm(after)?;
        let max_absolute_error = before
            .iter()
            .zip(after)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0_f64, f64::max);
        let cosine = before
            .iter()
            .zip(after)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum::<f64>()
            / (before_norm * after_norm);
        ensure!(
            max_absolute_error <= 1e-6 && cosine >= 0.999999,
            "Reload changed the embedding: max error {max_absolute_error}, cosine {cosine}"
        );
        Ok(json!({
            "dimensions": before.len(), "before_norm": before_norm, "after_norm": after_norm,
            "max_absolute_error": max_absolute_error, "cosine": cosine,
            "maximum_error_tolerance": 1e-6, "minimum_cosine": 0.999999,
        }))
    }

    let mut args = std::env::args_os().skip(1);
    let models = PathBuf::from(args.next().context("Provide the pinned model directory")?);
    let threads = args
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(2);
    ensure!(args.next().is_none(), "Expected MODEL_DIR [THREADS]");
    ensure!((1..=64).contains(&threads), "Threads must be in 1..=64");

    let runtime = NativeRuntime::new(models, threads)?;
    let idle_timeout = Duration::from_secs(1);
    // The two media encoders must remain untouched throughout this text-only run.
    runtime.set_idle_policy(RuntimeIdlePolicy {
        text: IdlePolicy::ReleaseAfter(idle_timeout),
        vision: IdlePolicy::KeepWarm,
        audio: IdlePolicy::KeepWarm,
    })?;
    let initial = runtime.diagnostics()?;
    ensure!(!initial.text.loaded && initial.text.load_count == 0);
    assert_lazy_media(&initial)?;

    let query = "How do I release an idle model without changing search results?";
    let title = "Runtime lifecycle notes";
    let passage = "The text model is unloaded after its idle timeout. The next search reloads the same pinned model and preserves normalized embeddings.";
    eprintln!("[Lifecycle] Loading the real CPU text model and embedding query/document");
    let first_started = Instant::now();
    let first_query = runtime.embed_text(query)?;
    let first_document = runtime.embed_document(title, passage)?;
    let first_embedding_ms = first_started.elapsed().as_secs_f64() * 1000.0;
    norm(&first_query)?;
    norm(&first_document)?;
    let loaded = runtime.diagnostics()?;
    ensure!(
        loaded.text.loaded && loaded.text.load_count == 1 && loaded.text.release_count == 0,
        "Expected exactly one initial text-session load"
    );
    assert_lazy_media(&loaded)?;

    eprintln!("[Lifecycle] Waiting for the actual idle timer to release the text session");
    let idle_started = Instant::now();
    let released = loop {
        let state = runtime.diagnostics()?;
        if !state.text.loaded {
            break state;
        }
        ensure!(
            idle_started.elapsed() < Duration::from_secs(20),
            "The idle timer did not release the text session within 20 seconds"
        );
        thread::sleep(Duration::from_millis(25));
    };
    let observed_release_wait_ms = idle_started.elapsed().as_secs_f64() * 1000.0;
    ensure!(released.text.load_count == 1 && released.text.release_count == 1);
    assert_lazy_media(&released)?;

    runtime.set_idle_policy(RuntimeIdlePolicy::KEEP_WARM)?;
    eprintln!("[Lifecycle] Reloading and comparing real query/document embeddings");
    let reload_started = Instant::now();
    let second_query = runtime.embed_text(query)?;
    let second_document = runtime.embed_document(title, passage)?;
    let reload_embedding_ms = reload_started.elapsed().as_secs_f64() * 1000.0;
    let query_parity = parity(&first_query, &second_query)?;
    let document_parity = parity(&first_document, &second_document)?;
    // Keep-warm must outlive the previously configured release deadline.
    thread::sleep(idle_timeout + Duration::from_millis(100));
    let reloaded = runtime.diagnostics()?;
    ensure!(
        reloaded.text.loaded && reloaded.text.load_count == 2 && reloaded.text.release_count == 1,
        "Expected one actual reload and no release while Keep warm is selected"
    );
    ensure!(reloaded.text.last_reload_duration_ms.is_some());
    assert_lazy_media(&reloaded)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "passed", "provider": "cpu", "threads": threads,
            "model_profile": MODEL_PROFILE,
            "build": if cfg!(debug_assertions) { "development; unoptimized unless the Cargo profile overrides opt-level" } else { "release/profile-dependent" },
            "measurement_scope": "Real CPU text-session lifecycle and reload parity; no GPU memory, reference-model parity, or retrieval-quality claim",
            "configured_idle_timeout_ms": idle_timeout.as_secs_f64() * 1000.0,
            "observed_release_wait_ms": observed_release_wait_ms,
            "first_query_and_document_ms": first_embedding_ms,
            "reload_query_and_document_ms": reload_embedding_ms,
            "query_parity": query_parity, "document_parity": document_parity,
            "initial": initial, "loaded": loaded, "released": released, "reloaded": reloaded,
        }))?
    );
    Ok(())
}
