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
pub fn install(
    destination: &Path,
    cancelled: impl Fn() -> bool,
    mut progress: impl FnMut(Value) -> Result<()>,
) -> Result<()> {
    let manifest: Manifest = serde_json::from_str(include_str!(
        "../../../crates/fractal-search-core/models/fgclip2.json"
    ))?;
    fs::create_dir_all(destination)?;
    let total: u64 = manifest.files.iter().map(|f| f.bytes).sum();
    let mut completed = 0;
    for file in manifest.files {
        ensure!(
            !cancelled(),
            "Model setup stopped; verified completed assets are retained"
        );
        let path = destination.join(&file.path);
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
        let partial = path.with_extension("part");
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
