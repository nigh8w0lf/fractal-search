//! First-run model setup, using the exact runtime manifest and HTTPS-only curl.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path, process::Command, time::Duration};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    repository: String,
    revision: String,
    files: Vec<ModelFile>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelFile {
    path: String,
    bytes: u64,
    sha256: String,
}

fn verify(path: &Path, file: &ModelFile) -> Result<()> {
    ensure!(
        fs::metadata(path)?.len() == file.bytes,
        "Model asset size mismatch: {path:?}"
    );
    let mut source = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    ensure!(format!("{:x}", hash.finalize()) == file.sha256, "SHA-256 mismatch for model asset {path:?}; remove the damaged file or choose a verified model folder");
    Ok(())
}
fn asset_paths(
    destination: &Path,
    relative: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    use std::path::Component;
    let relative = Path::new(relative);
    ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "Invalid model asset path"
    );
    let path = destination.join(relative);
    let filename = path.file_name().context("Model asset has no filename")?;
    let mut partial_name = filename.to_os_string();
    partial_name.push(".part");
    let partial = path.with_file_name(partial_name);
    Ok((path, partial))
}

pub fn install(
    destination: &Path,
    cancelled: impl Fn() -> bool,
    mut progress: impl FnMut(Value) -> Result<()>,
) -> Result<()> {
    let manifest: Manifest = serde_json::from_str(include_str!(
        "../../../crates/fractal-search-core/models/embeddinggemma2.json"
    ))?;
    fs::create_dir_all(destination)?;
    let total: u64 = manifest.files.iter().map(|f| f.bytes).sum();
    let mut completed = 0;
    for file in manifest.files {
        ensure!(
            !cancelled(),
            "Model setup stopped; verified completed assets are retained"
        );
        let (path, partial) = asset_paths(destination, &file.path)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        progress(
            json!({"stage":"verifying","filename":file.path,"current":completed,"total":total}),
        )?;
        if path.try_exists()? {
            verify(&path, &file)?;
            completed += file.bytes;
            progress(
                json!({"stage":"verified","filename":file.path,"current":completed,"total":total}),
            )?;
            continue;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            manifest.repository, manifest.revision, file.path
        );
        let mut child = Command::new("curl")
            .args([
                "--fail",
                "--location",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "20",
                "--max-time",
                "1800",
                "--max-filesize",
                &file.bytes.to_string(),
                "--output",
            ])
            .arg(&partial)
            .arg(&url)
            .spawn()
            .context("Model setup requires curl; install it with omarchy pkg add curl")?;
        let download = (|| -> Result<()> {
            loop {
                if cancelled() {
                    anyhow::bail!("Model setup stopped; verified completed assets are retained");
                }
                let received = match fs::metadata(&partial) {
                    Ok(meta) => meta.len(),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
                    Err(error) => return Err(error.into()),
                };
                progress(
                    json!({"stage":"downloading","filename":file.path,"current":completed + received,"total":total}),
                )?;
                if let Some(status) = child.try_wait()? {
                    ensure!(
                        status.success(),
                        "Download failed for {} ({status}); check the service log and retry",
                        file.path
                    );
                    break;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            verify(&partial, &file)?;
            fs::File::open(&partial)?.sync_all()?;
            fs::rename(&partial, &path)?;
            Ok(())
        })();
        if let Err(error) = download {
            if child.try_wait()?.is_none() {
                child.kill()?;
                child.wait()?;
            }
            return Err(error);
        }
        completed += file.bytes;
        progress(
            json!({"stage":"verified","filename":file.path,"current":completed,"total":total}),
        )?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_model_assets_have_distinct_partial_paths() {
        let root = Path::new("/tmp/fractal-model-test");
        let (model, partial) = asset_paths(root, "onnx/model.onnx").unwrap();
        let (data, data_partial) = asset_paths(root, "onnx/model.onnx_data").unwrap();
        assert_eq!(model, root.join("onnx/model.onnx"));
        assert_eq!(data, root.join("onnx/model.onnx_data"));
        assert_ne!(partial, data_partial);
        assert_eq!(partial, root.join("onnx/model.onnx.part"));
        for path in ["", "/absolute", "../escape", "onnx/../escape"] {
            assert!(asset_paths(root, path).is_err());
        }
    }
    #[test]
    fn pinned_manifest_contains_embeddinggemma2_multimodal_assets() {
        let manifest: Manifest = serde_json::from_str(include_str!(
            "../../../crates/fractal-search-core/models/embeddinggemma2.json"
        ))
        .unwrap();
        assert_eq!(manifest.repository, "onnx-community/embeddinggemma-2-ONNX");
        assert_eq!(
            manifest.revision,
            "daa72c51243991dfcaf9f9137d2c573d8f7790c0"
        );
        assert_eq!(manifest.files.len(), 11);
        assert_eq!(
            manifest.files.iter().map(|file| file.bytes).sum::<u64>(),
            2_960_874_003
        );
        let paths: std::collections::BTreeSet<_> = manifest
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect();
        assert_eq!(paths.len(), manifest.files.len());
        for path in [
            "onnx/model.onnx",
            "onnx/model.onnx_data",
            "onnx/vision_encoder.onnx",
            "onnx/vision_encoder.onnx_data",
            "onnx/audio_encoder.onnx",
            "onnx/audio_encoder.onnx_data",
            "tokenizer.json",
            "preprocessor_config.json",
        ] {
            assert!(paths.contains(path));
        }
    }
    #[test]
    fn model_verification_rejects_corruption_and_preserves_valid_files() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.bin");
        fs::write(&path, b"model").unwrap();
        let file = ModelFile {
            path: "model.bin".into(),
            bytes: 5,
            sha256: format!("{:x}", Sha256::digest(b"model")),
        };
        verify(&path, &file).unwrap();
        fs::write(&path, b"other").unwrap();
        assert!(verify(&path, &file).is_err());
        assert_eq!(fs::read(path).unwrap(), b"other");
        assert!(install(root.path(), || true, |_| Ok(())).is_err());
    }
}
