//! Native desktop portal client. Never load GTK into Omarchy's shell process.
use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use serde_json::json;
use std::{
    collections::HashMap,
    io::{Read, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};
use zbus::{
    blocking::{connection::Builder, Proxy},
    zvariant::{OwnedObjectPath, OwnedValue, Value},
};

static CANCELLED: AtomicBool = AtomicBool::new(false);
const PORTAL: &str = "org.freedesktop.portal.Desktop";

#[derive(Parser)]
struct Options {
    #[arg(long)]
    title: String,
}

unsafe extern "C" fn cancel_signal(_: libc::c_int) {
    CANCELLED.store(true, Ordering::Relaxed);
}

fn selected_folder(code: u32, mut values: HashMap<String, OwnedValue>) -> Result<Option<String>> {
    match code {
        1 => return Ok(None),
        0 => (),
        _ => bail!("The desktop portal ended the folder selection with response {code}"),
    }
    let uris: Vec<String> = values
        .remove("uris")
        .context("The desktop portal returned no folder URI")?
        .try_into()
        .context("The desktop portal returned invalid folder URIs")?;
    ensure!(uris.len() == 1, "Expected exactly one selected folder");
    ensure!(
        uris[0].len() <= 16384,
        "The selected folder URI is too long"
    );
    let url = url::Url::parse(&uris[0]).context("Invalid selected folder URI")?;
    let path = url
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("The selected folder must be a local file URI"))?;
    ensure!(
        path.is_dir(),
        "The selected folder is no longer a directory: {}",
        path.display()
    );
    Ok(Some(url.into()))
}

fn choose(title: &str) -> Result<Option<String>> {
    ensure!(
        !title.is_empty() && title.len() <= 256,
        "Folder chooser title must be 1..256 bytes"
    );
    for signal in [libc::SIGINT, libc::SIGTERM] {
        ensure!(
            unsafe { libc::signal(signal, cancel_signal as *const () as libc::sighandler_t) }
                != libc::SIG_ERR,
            "Cannot install folder chooser cancellation handler"
        );
    }
    let connection = Builder::session()
        .context("Cannot connect to the desktop session bus")?
        .method_timeout(Duration::from_secs(10))
        .build()?;
    let chooser = Proxy::new(
        &connection,
        PORTAL,
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.portal.FileChooser",
    )?;
    let version: u32 = chooser.get_property("version").context("The desktop FileChooser portal is unavailable. Install xdg-desktop-portal and a FileChooser backend (xdg-desktop-portal-gtk on Omarchy)")?;
    ensure!(
        version >= 3,
        "Directory selection requires FileChooser portal version 3 or newer (found {version})"
    );
    let mut random = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut random)?;
    let token = format!(
        "fractal_{}",
        random
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    let sender = connection
        .unique_name()
        .context("Session bus did not assign a unique name")?
        .as_str()
        .trim_start_matches(':')
        .replace('.', "_");
    let path = format!("/org/freedesktop/portal/desktop/request/{sender}/{token}");
    let request = Proxy::new_owned(
        connection.clone(),
        PORTAL.to_owned(),
        path.clone(),
        "org.freedesktop.portal.Request".to_owned(),
    )?;
    // Subscribe before OpenFile: an immediate response must not get lost.
    let mut responses = request.receive_signal("Response")?;
    let mut owners = request.receive_owner_changed()?;
    if CANCELLED.load(Ordering::Relaxed) {
        return Ok(None);
    }
    let options = HashMap::from([
        ("handle_token", Value::from(token.as_str())),
        ("directory", Value::from(true)),
        ("multiple", Value::from(false)),
        ("modal", Value::from(false)),
    ]);
    // Layer-shell windows cannot export an xdg-toplevel parent handle. An empty
    // parent is supported by the portal; the panel releases focus separately.
    let handle: OwnedObjectPath = chooser
        .call("OpenFile", &("", title, options))
        .context("The desktop portal could not open the folder chooser")?;
    ensure!(
        handle.as_str() == path,
        "The desktop portal returned an unexpected request handle"
    );
    eprintln!("[Folder picker] Native portal opened");
    let (tx, rx) = mpsc::sync_channel(1);
    let disconnected = tx.clone();
    std::thread::spawn(move || {
        owners.next();
        let _ = disconnected.send(Err(anyhow::anyhow!(
            "The desktop portal disconnected or restarted during folder selection"
        )));
    });
    std::thread::spawn(move || {
        let result = responses
            .next()
            .context("The folder chooser disconnected before responding")
            .and_then(|message| {
                message
                    .body()
                    .deserialize::<(u32, HashMap<String, OwnedValue>)>()
                    .context("Invalid desktop portal response")
            });
        let _ = tx.send(result);
    });
    loop {
        if CANCELLED.load(Ordering::Relaxed) {
            request
                .call::<_, _, ()>("Close", &())
                .context("Cannot close the desktop folder chooser")?;
            return Ok(None);
        }
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(result) => {
                let (code, values) = result?;
                return selected_folder(code, values);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => (),
            Err(error) => bail!("Folder chooser response worker failed: {error}"),
        }
    }
}

fn main() {
    let options = Options::parse();
    let (response, failed) = match choose(&options.title) {
        Ok(Some(folder)) => (json!({"ok":true,"cancelled":false,"folder":folder}), false),
        Ok(None) => (json!({"ok":true,"cancelled":true}), false),
        Err(error) => {
            eprintln!("[Folder picker] {error:#}");
            (json!({"ok":false,"error":format!("{error:#}")}), true)
        }
    };
    let mut output = std::io::stdout().lock();
    if serde_json::to_writer(&mut output, &response).is_err()
        || writeln!(&mut output).is_err()
        || output.flush().is_err()
        || failed
    {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_distinct_from_errors() {
        assert_eq!(selected_folder(1, HashMap::new()).unwrap(), None);
        assert!(selected_folder(2, HashMap::new()).is_err());
        assert!(selected_folder(0, HashMap::new()).is_err());
    }

    #[test]
    fn only_one_existing_local_folder_is_accepted() {
        let directory = tempfile::tempdir().unwrap();
        let uri = url::Url::from_directory_path(directory.path())
            .unwrap()
            .to_string();
        let values = |uris: Vec<String>| {
            HashMap::from([(
                "uris".to_owned(),
                OwnedValue::try_from(Value::from(uris)).unwrap(),
            )])
        };
        assert_eq!(
            selected_folder(0, values(vec![uri.clone()])).unwrap(),
            Some(uri.clone())
        );
        for uris in [
            vec![],
            vec![uri.clone(), uri],
            vec!["https://example.com".to_owned()],
            vec!["file:///no-such-fractal-folder".to_owned()],
        ] {
            assert!(selected_folder(0, values(uris)).is_err());
        }
    }
}
