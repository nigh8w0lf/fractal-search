//! A session service owns the stdio worker so QML destruction cannot kill inference.
//! Socket clients are disposable; request IDs are translated per connection.
use super::{install_signals, read_frame, Options, Request, STOP};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs,
    io::{BufReader, Read, Write},
    os::{
        fd::AsRawFd,
        unix::{
            fs::{FileTypeExt, MetadataExt, PermissionsExt},
            net::{UnixListener, UnixStream},
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

const MAX_RESPONSE: usize = 2 * 1024 * 1024;
const MAX_PENDING: usize = 32;

// Returning from a transport error must still drain native inference. Child's
// standard Drop only closes its handles and would leave an orphan worker.
struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        match self.0.try_wait() {
            Ok(Some(_)) => (),
            Ok(None) => {
                if unsafe { libc::kill(self.0.id() as i32, libc::SIGTERM) } != 0 {
                    eprintln!(
                        "[Plugin] Cannot signal worker: {}",
                        std::io::Error::last_os_error()
                    );
                }
                if let Err(error) = self.0.wait() {
                    eprintln!("[Plugin] Cannot drain worker: {error}");
                }
            }
            Err(error) => eprintln!("[Plugin] Cannot inspect worker exit: {error}"),
        }
    }
}
impl std::ops::Deref for Worker {
    type Target = Child;
    fn deref(&self) -> &Child {
        &self.0
    }
}
impl std::ops::DerefMut for Worker {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}

struct Pending {
    session: u64,
    id: Option<u64>,
    kind: String,
}
#[derive(Default)]
struct Hub {
    session: u64,
    client: Option<UnixStream>,
    next_id: u64,
    pending: HashMap<u64, Pending>,
    status: Option<Value>,
    job: Option<Value>,
    report: Option<Value>,
    warmup_error: Option<Value>,
    shutting_down: bool,
}
impl Hub {
    fn send(&mut self, value: &Value) -> Result<()> {
        if let Some(client) = &mut self.client {
            let bytes = serde_json::to_vec(value)?;
            if let Err(error) = client
                .write_all(&bytes)
                .and_then(|_| client.write_all(b"\n"))
            {
                // A closed panel is an expected transport detach, not an engine failure.
                eprintln!("[Plugin] Client detached while delivering response: {error}");
                self.client = None;
            }
        }
        Ok(())
    }
    fn ready(&mut self) -> Result<()> {
        if let Some(mut value) = self.status.clone() {
            value["activeJob"] = self.job.clone().unwrap_or(Value::Null);
            value["lastIndexReport"] = self.report.clone().unwrap_or(Value::Null);
            value["warmupError"] = self.warmup_error.clone().unwrap_or(Value::Null);
            self.send(&json!({"event":"ready", "payload":value}))?;
        }
        Ok(())
    }
    fn admit(&mut self, session: u64, id: Option<u64>, command: &Value) -> Result<Value> {
        ensure!(
            self.pending.len() < MAX_PENDING
                || command["type"] == "shutdown"
                || command["type"] == "cancel",
            "Search request queue is full"
        );
        self.next_id += 1;
        ensure!(
            self.next_id <= 9_007_199_254_740_991,
            "Service request IDs exhausted; restart the service"
        );
        let kind = command["type"]
            .as_str()
            .context("Missing command type")?
            .to_string();
        if kind == "shutdown" {
            self.shutting_down = true;
        }
        if kind == "index" || kind == "installModels" {
            ensure!(self.job.is_none(), "A background job is already active");
            self.job = Some(
                json!({"kind":kind,"requestId":self.next_id,"directory":command["directory"],"progress":null,"failures":[]}),
            );
            if kind == "index" {
                self.report = None;
            }
        }
        let internal = self.next_id;
        self.pending.insert(internal, Pending { session, id, kind });
        Ok(json!({"id":internal,"command":command}))
    }
    fn receive(&mut self, mut message: Value) -> Result<()> {
        let event = message["event"].as_str().unwrap_or("").to_string();
        if event == "ready" {
            ensure!(
                message["payload"]["protocolVersion"] == 1,
                "Incompatible search worker protocol"
            );
            self.status = Some(message["payload"].clone());
            return self.ready();
        }
        if event == "warmupFinished" {
            self.warmup_error = message["payload"]
                .get("error")
                .filter(|value| !value.is_null())
                .cloned();
            if let Some(status) = &mut self.status {
                status["textModelWarming"] = json!(false);
                status["textModelLoaded"] = message["payload"]["ok"].clone();
                if message["payload"]["ok"] == true {
                    status["modelLoaded"] = json!(true);
                }
            }
        }
        if event == "indexStarted" {
            self.job = Some(
                json!({"kind":"index","requestId":null,"directory":message["payload"]["directory"],"progress":null,"failures":[]}),
            );
        }
        if event == "indexProgress" || event == "modelProgress" {
            if let Some(job) = &mut self.job {
                job["progress"] = message["payload"].clone();
                if let Some(failure) = message["payload"].get("failure").filter(|v| !v.is_null()) {
                    let failures = job["failures"]
                        .as_array_mut()
                        .context("Invalid job report")?;
                    if failures.len() < 100 {
                        failures.push(failure.clone());
                    }
                }
            }
        }
        if event == "indexError" {
            self.failed_report(&message["payload"]["error"]);
        }
        if event == "indexFinished" || event == "indexError" {
            self.job = None;
        }
        if event == "indexFinished" {
            self.report = Some(message["payload"].clone());
        }
        if let Some(id) = message["id"].as_u64() {
            let Some(pending) = self.pending.get(&id) else {
                bail!("Unexpected worker reply {id}");
            };
            if !event.is_empty() {
                if pending.session == self.session {
                    message["id"] = json!(pending.id);
                } else if event == "indexProgress" || event == "modelProgress" {
                    message["id"] = Value::Null;
                } else {
                    return Ok(());
                } // Search events belong to the dismissed connection.
            } else {
                let pending = self
                    .pending
                    .remove(&id)
                    .context("Missing pending request")?;
                if pending.kind == "status"
                    || pending.kind == "setModelDirectory"
                    || pending.kind == "updateSettings"
                    || pending.kind == "installModels"
                {
                    if message["ok"] == true {
                        self.status = Some(message["result"].clone());
                    }
                }
                if (pending.kind == "index" || pending.kind == "installModels")
                    && self.job.as_ref().is_some_and(|job| job["requestId"] == id)
                {
                    if pending.kind == "index" && message["ok"] != true {
                        self.failed_report(&message["error"]);
                    }
                    self.job = None;
                    if pending.kind == "index" && message["ok"] == true {
                        self.report = Some(message["result"].clone());
                    }
                }
                if pending.id.is_none() {
                    if pending.kind == "shutdown" {
                        return Ok(());
                    }
                    if message["ok"] != true {
                        // The worker rejects queued status refreshes while draining.
                        // Treat that explicit cancellation as shutdown progress;
                        // any other internal refresh failure remains fatal.
                        if self.shutting_down
                            && message["error"] == "Search service is shutting down"
                        {
                            eprintln!("[Shutdown] Cancelled pending status refresh while draining");
                            return Ok(());
                        }
                        bail!("Refresh search status: {}", message["error"]);
                    }
                    return self.send(&json!({"event":"status","payload":message["result"]}));
                }
                if pending.session != self.session {
                    if pending.kind == "index" {
                        return if message["ok"] == true {
                            self.send(&json!({"event":"indexFinished","payload":message["result"]}))
                        } else {
                            self.send(&json!({"event":"indexError","payload":{"directory":"","error":message["error"]}}))
                        };
                    }
                    if pending.kind == "installModels" {
                        return self.send(&json!({"event":"modelsFinished","payload":message}));
                    }
                    eprintln!(
                        "[Plugin] Completed {} for detached client {}",
                        pending.kind, pending.session
                    );
                    return Ok(());
                }
                message["id"] = json!(pending.id);
            }
        }
        self.send(&message)
    }
    fn failed_report(&mut self, error: &Value) {
        if let Some(job) = &self.job {
            self.report = Some(
                json!({"indexed":0,"failed":job["progress"]["failed"].as_u64().unwrap_or(0),"failures":job["failures"],"error":error}),
            );
        }
    }
}

fn write_request(input: &mut impl Write, request: &Value) -> Result<()> {
    serde_json::to_writer(&mut *input, request)?;
    input.write_all(b"\n")?;
    input.flush()?;
    Ok(())
}
fn socket_path(options: &Options) -> Result<PathBuf> {
    if let Some(path) = &options.socket {
        return Ok(path.clone());
    }
    Ok(PathBuf::from(
        std::env::var_os("XDG_RUNTIME_DIR")
            .context("XDG_RUNTIME_DIR is required for the plugin service")?,
    )
    .join("fractal-search/backend.sock"))
}
struct SocketCleanup(PathBuf);
impl Drop for SocketCleanup {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_file(&self.0) {
            eprintln!("[Plugin] Cannot remove service socket: {error}");
        }
    }
}
fn listener(path: &Path) -> Result<(UnixListener, fs::File, SocketCleanup)> {
    ensure!(
        path.is_absolute(),
        "Service socket must be an absolute path"
    );
    let parent = path.parent().context("Missing socket directory")?;
    fs::create_dir_all(parent)?;
    let meta = fs::symlink_metadata(parent)?;
    ensure!(
        meta.is_dir() && meta.uid() == unsafe { libc::geteuid() },
        "Socket directory must belong to the current user"
    );
    fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(parent.join("backend.lock"))?;
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "Another search daemon owns this socket"
    );
    match fs::symlink_metadata(path) {
        Ok(meta) => {
            ensure!(
                meta.file_type().is_socket() && meta.uid() == unsafe { libc::geteuid() },
                "Refusing to replace a non-owned socket path"
            );
            fs::remove_file(path)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    let listener = UnixListener::bind(path)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    listener.set_nonblocking(true)?;
    Ok((listener, lock, SocketCleanup(path.to_path_buf())))
}
fn same_user(client: &UnixStream) -> Result<()> {
    let mut peer = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    ensure!(
        unsafe {
            libc::getsockopt(
                client.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&mut peer as *mut libc::ucred).cast(),
                &mut length,
            )
        } == 0,
        "Cannot inspect plugin client credentials"
    );
    ensure!(
        peer.uid == unsafe { libc::geteuid() },
        "Plugin socket accepts only the current desktop user"
    );
    Ok(())
}

pub fn run(options: Options) -> Result<()> {
    install_signals()?;
    let (listener, _lock, _cleanup) = listener(&socket_path(&options)?)?;
    let mut command = Command::new(std::env::current_exe()?);
    command.arg("serve");
    command.arg("--provider").arg(&options.provider);
    if let Some(device_id) = options.device_id {
        command.arg("--device-id").arg(device_id.to_string());
    }
    if options.warm_search {
        command.arg("--warm-search");
    }
    for (flag, path) in [
        ("--data-dir", &options.data_dir),
        ("--model-dir", &options.model_dir),
        ("--theme-file", &options.theme_file),
    ] {
        if let Some(path) = path {
            command.arg(flag).arg(path);
        }
    }
    if let Some(threads) = options.threads {
        command.arg("--threads").arg(threads.to_string());
    }
    let mut worker = Worker(
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Start bundled search worker")?,
    );
    let input = Arc::new(Mutex::new(
        worker.stdin.take().context("Missing worker stdin")?,
    ));
    let stdout = worker.stdout.take().context("Missing worker stdout")?;
    let hub = Arc::new(Mutex::new(Hub::default()));
    let failed = Arc::new(AtomicBool::new(false));
    let reader_hub = hub.clone();
    let reader_failed = failed.clone();
    let reader = std::thread::spawn(move || {
        let result = (|| -> Result<()> {
            let mut reader = BufReader::new(stdout);
            use std::io::BufRead;
            loop {
                let mut bytes = Vec::new();
                let count = reader
                    .by_ref()
                    .take((MAX_RESPONSE + 1) as u64)
                    .read_until(b'\n', &mut bytes)?;
                if count == 0 {
                    break;
                }
                ensure!(
                    count <= MAX_RESPONSE && bytes.last() == Some(&b'\n'),
                    "Worker response exceeds 2 MiB or is incomplete"
                );
                reader_hub
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Plugin hub poisoned"))?
                    .receive(serde_json::from_slice(&bytes)?)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            eprintln!("[Plugin] Worker transport failed: {error:#}");
            reader_failed.store(true, Ordering::Relaxed);
            if let Ok(mut hub) = reader_hub.lock() {
                if let Err(report_error) =
                    hub.send(&json!({"event":"fatal","payload":{"error":format!("{error:#}")}}))
                {
                    eprintln!("[Plugin] Cannot report worker failure: {report_error:#}");
                }
            }
            STOP.store(true, Ordering::Relaxed);
        }
    });
    let mut shutdown_sent = false;
    let result = loop {
        if STOP.load(Ordering::Relaxed) && !shutdown_sent {
            shutdown_sent = true;
            let mut hub = hub
                .lock()
                .map_err(|_| anyhow::anyhow!("Plugin hub poisoned"))?;
            let session = hub.session;
            hub.shutting_down = true;
            // Shutdown must remain admissible even when the ordinary queue is full.
            hub.next_id += 1;
            let id = hub.next_id;
            hub.pending.insert(
                id,
                Pending {
                    session,
                    id: None,
                    kind: "shutdown".into(),
                },
            );
            write_request(
                &mut *input
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Worker stdin poisoned"))?,
                &json!({"id":id,"command":{"type":"shutdown"}}),
            )?;
        }
        if let Some(status) = worker.try_wait()? {
            break status;
        }
        if !STOP.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut client, _)) => {
                    if let Err(error) = same_user(&client) {
                        eprintln!("[Plugin] Rejected client: {error:#}");
                        continue;
                    }
                    client.set_write_timeout(Some(Duration::from_secs(2)))?;
                    let mut state = hub
                        .lock()
                        .map_err(|_| anyhow::anyhow!("Plugin hub poisoned"))?;
                    if state.client.is_some() {
                        if let Err(error) = client.write_all(b"{\"event\":\"fatal\",\"payload\":{\"error\":\"Fractal Search already has a connected UI\"}}\n") {
                            eprintln!("[Plugin] Rejected UI disconnected: {error}");
                        }
                        continue;
                    }
                    state.session += 1;
                    let session = state.session;
                    state.client = Some(client.try_clone()?);
                    state.ready()?;
                    let request = state.admit(session, None, &json!({"type":"status"}))?;
                    write_request(
                        &mut *input
                            .lock()
                            .map_err(|_| anyhow::anyhow!("Worker stdin poisoned"))?,
                        &request,
                    )?;
                    drop(state);
                    let client_hub = hub.clone();
                    let client_input = input.clone();
                    std::thread::spawn(move || {
                        let result = (|| -> Result<()> {
                            let mut reader = BufReader::new(client);
                            while let Some(frame) = read_frame(&mut reader)? {
                                let request: Request = match serde_json::from_slice(&frame) {
                                    Ok(request) => request,
                                    Err(error) => {
                                        client_hub.lock().map_err(|_| anyhow::anyhow!("Plugin hub poisoned"))?.send(&json!({"id":null,"ok":false,"error":error.to_string()}))?;
                                        continue;
                                    }
                                };
                                ensure!(request.id <= 9_007_199_254_740_991, "Unsafe request ID");
                                let value: Value = serde_json::from_slice(&frame)?;
                                let mut hub = client_hub
                                    .lock()
                                    .map_err(|_| anyhow::anyhow!("Plugin hub poisoned"))?;
                                ensure!(
                                    hub.session == session && !STOP.load(Ordering::Relaxed),
                                    "Search connection is shutting down"
                                );
                                let translated = match hub.admit(
                                    session,
                                    Some(request.id),
                                    &value["command"],
                                ) {
                                    Ok(request) => request,
                                    Err(error) => {
                                        hub.send(&json!({"id":request.id,"ok":false,"error":format!("{error:#}")}))?;
                                        continue;
                                    }
                                };
                                write_request(
                                    &mut *client_input
                                        .lock()
                                        .map_err(|_| anyhow::anyhow!("Worker stdin poisoned"))?,
                                    &translated,
                                )?;
                            }
                            Ok(())
                        })();
                        if let Err(error) = result {
                            eprintln!("[Plugin] Client connection failed: {error:#}");
                        }
                        if let Ok(mut hub) = client_hub.lock() {
                            if hub.session == session {
                                hub.client = None;
                            }
                        }
                        eprintln!(
                            "[Plugin] Client {session} disconnected; worker remains resident"
                        );
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => (),
                Err(error) => return Err(error.into()),
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    reader
        .join()
        .map_err(|_| anyhow::anyhow!("Worker output thread panicked"))?;
    let mut hub = hub
        .lock()
        .map_err(|_| anyhow::anyhow!("Plugin hub poisoned"))?;
    hub.send(&json!({"event":"stopped","payload":{"clean":result.success() && !failed.load(Ordering::Relaxed)}}))?;
    ensure!(
        result.success() && !failed.load(Ordering::Relaxed),
        "Search worker exited unsuccessfully: {result}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shutdown_cancels_internal_refresh_without_losing_the_shutdown_reply() {
        let mut hub = Hub::default();
        let refresh = hub.admit(0, None, &json!({"type":"status"})).unwrap();
        let shutdown = hub.admit(0, Some(1), &json!({"type":"shutdown"})).unwrap();
        hub.receive(
            json!({"id":refresh["id"],"ok":false,"error":"Search service is shutting down"}),
        )
        .unwrap();
        hub.receive(json!({"id":shutdown["id"],"ok":true,"result":{"stopped":true}}))
            .unwrap();
        assert!(hub.pending.is_empty());
        let refresh = hub.admit(0, None, &json!({"type":"status"})).unwrap();
        assert!(hub
            .receive(json!({"id":refresh["id"],"ok":false,"error":"SQLite failure"}))
            .is_err());
    }
    #[test]
    fn detached_job_finishes_without_reusing_client_request_ids() {
        let mut hub = Hub {
            session: 1,
            ..Hub::default()
        };
        let first = hub
            .admit(1, Some(1), &json!({"type":"index","directory":"/photos"}))
            .unwrap();
        hub.receive(json!({"id":first["id"],"event":"indexProgress","payload":{"current":1,"failure":{"path":"broken","error":"decode"}}})).unwrap();
        hub.session = 2;
        let second = hub.admit(2, Some(1), &json!({"type":"status"})).unwrap();
        assert_ne!(first["id"], second["id"]);
        assert_eq!(
            hub.job.as_ref().unwrap()["failures"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        hub.receive(json!({"id":first["id"],"ok":true,"result":{"failed":1,"failures":[]}}))
            .unwrap();
        assert!(hub.job.is_none());
        assert_eq!(hub.report.as_ref().unwrap()["failed"], 1);
        assert!(hub.pending.contains_key(&second["id"].as_u64().unwrap()));
    }
    #[test]
    fn rejected_manual_scan_does_not_erase_a_running_watch_scan() {
        let mut hub = Hub::default();
        let request = hub
            .admit(0, Some(1), &json!({"type":"index","directory":"/manual"}))
            .unwrap();
        // A watcher may acquire the worker admission latch just before the
        // daemon's manual request reaches its stdin thread.
        hub.receive(json!({"event":"indexStarted","payload":{"directory":"/watched"}}))
            .unwrap();
        hub.receive(json!({"id":request["id"],"ok":false,"error":"Index already active"}))
            .unwrap();
        assert_eq!(hub.job.as_ref().unwrap()["directory"], "/watched");
    }
    #[test]
    fn socket_is_private_exclusive_and_never_overwrites_an_ordinary_file() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("run/backend.sock");
        let (socket, lock, cleanup) = listener(&path).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        assert!(listener(&path).is_err());
        drop(socket);
        drop(cleanup);
        drop(lock);
        fs::write(&path, "keep").unwrap();
        assert!(listener(&path).is_err());
        assert_eq!(fs::read_to_string(path).unwrap(), "keep");
    }
}
