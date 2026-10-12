//! Newline JSON-RPC with bounded memory and one admitted request at a time.
use rmcp::{
    service::{RxJsonRpcMessage, TxJsonRpcMessage},
    transport::Transport,
    RoleServer,
};
use serde_json::{json, Value};
use std::{
    io,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Stdin, Stdout},
    sync::{Mutex, Semaphore},
};

const MAX_INPUT: usize = 65536;
const MAX_OUTPUT: usize = 2 * 1024 * 1024;
pub struct BoundedStdio {
    input: BufReader<Stdin>,
    output: Arc<Mutex<Stdout>>,
    admission: Arc<Semaphore>,
    pub failed: Arc<AtomicBool>,
    initialized_notification_seen: bool,
}
impl BoundedStdio {
    pub fn new() -> Self {
        Self {
            input: BufReader::new(tokio::io::stdin()),
            output: Arc::new(Mutex::new(tokio::io::stdout())),
            admission: Arc::new(Semaphore::new(1)),
            failed: Arc::new(AtomicBool::new(false)),
            initialized_notification_seen: false,
        }
    }
    async fn reject(&self, id: Value, code: i64, message: &str) -> io::Result<()> {
        let bytes = serde_json::to_vec(
            &json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}}),
        )?;
        write(&self.output, &bytes).await
    }
}
async fn write(output: &Mutex<Stdout>, bytes: &[u8]) -> io::Result<()> {
    if bytes.len() > MAX_OUTPUT {
        return Err(io::Error::other("MCP response exceeds 2 MiB"));
    }
    let mut output = output.lock().await;
    output.write_all(bytes).await?;
    output.write_all(b"\n").await?;
    output.flush().await
}
// Read in chunks instead of read_line: an untrusted unterminated line must not
// allocate arbitrary memory. Overlong input terminates the connection explicitly.
async fn frame(input: &mut (impl tokio::io::AsyncBufRead + Unpin)) -> io::Result<Option<Vec<u8>>> {
    let mut frame = Vec::new();
    loop {
        let available = input.fill_buf().await?;
        if available.is_empty() {
            return if frame.is_empty() {
                Ok(None)
            } else {
                Err(io::Error::other("MCP frame is missing its newline"))
            };
        }
        let end = available.iter().position(|&b| b == b'\n');
        let count = end.map_or(available.len(), |i| i + 1);
        if frame.len() + count > MAX_INPUT {
            return Err(io::Error::other("MCP frame exceeds 64 KiB"));
        }
        frame.extend_from_slice(&available[..count]);
        input.consume(count);
        if end.is_some() {
            return Ok(Some(frame));
        }
    }
}
fn only(value: &Value, fields: &[&str]) -> bool {
    value
        .as_object()
        .is_some_and(|o| o.keys().all(|key| fields.contains(&key.as_str())))
}
fn valid_envelope(value: &Value) -> bool {
    if !only(value, &["jsonrpc", "id", "method", "params"]) || value["jsonrpc"] != "2.0" {
        return false;
    }
    if let Some(id) = value.get("id") {
        if !(id.is_i64() || id.as_str().is_some_and(|s| s.len() <= 128)) {
            return false;
        }
    }
    let Some(method) = value["method"].as_str() else {
        return false;
    };
    let Some(params) = value.get("params") else {
        return true;
    };
    let fields: &[&str] = match method {
        "initialize" => &["protocolVersion", "capabilities", "clientInfo", "_meta"],
        "tools/list" => &["cursor", "_meta"],
        "tools/call" => &["name", "arguments", "_meta"],
        "ping" | "notifications/initialized" => &["_meta"],
        "notifications/cancelled" => &["requestId", "reason", "_meta"],
        _ => return true,
    };
    if !only(params, fields) {
        return false;
    }
    if method == "initialize" {
        if let Some(info) = params.get("clientInfo") {
            if !only(
                info,
                &[
                    "name",
                    "version",
                    "title",
                    "description",
                    "icons",
                    "websiteUrl",
                ],
            ) {
                return false;
            }
        }
        if let Some(caps) = params.get("capabilities") {
            if !only(
                caps,
                &[
                    "experimental",
                    "extensions",
                    "roots",
                    "sampling",
                    "elicitation",
                ],
            ) {
                return false;
            }
            for (key, allowed) in [
                ("roots", &["listChanged"][..]),
                ("sampling", &["tools", "context"][..]),
                ("elicitation", &["form", "url"][..]),
            ] {
                if caps.get(key).is_some_and(|value| !only(value, allowed)) {
                    return false;
                }
            }
        }
    }
    true
}
impl Transport<RoleServer> for BoundedStdio {
    type Error = io::Error;
    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleServer>,
    ) -> impl std::future::Future<Output = io::Result<()>> + Send + 'static {
        let output = self.output.clone();
        let admission = self.admission.clone();
        let failed = self.failed.clone();
        async move {
            let response = matches!(
                item,
                rmcp::model::JsonRpcMessage::Response(_) | rmcp::model::JsonRpcMessage::Error(_)
            );
            let result = match serde_json::to_vec(&item) {
                Ok(bytes) => write(&output, &bytes).await,
                Err(_) => Err(io::Error::other("Cannot serialize MCP response")),
            };
            if result.is_err() {
                failed.store(true, Ordering::Release);
            }
            if response {
                admission.add_permits(1);
            }
            result
        }
    }
    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleServer>> {
        // Waiting here also means EOF cannot cancel accepted inference: its
        // response is flushed first, then the SDK is allowed to see EOF.
        let permit = self.admission.clone().acquire_owned().await.ok()?;
        loop {
            let bytes = match frame(&mut self.input).await {
                Ok(Some(bytes)) => bytes,
                Ok(None) => return None,
                Err(error) => {
                    eprintln!("[MCP] {error}");
                    self.failed.store(true, Ordering::Release);
                    let _ = self
                        .reject(
                            Value::Null,
                            -32600,
                            "Invalid or overlong MCP frame; connection closed",
                        )
                        .await;
                    return None;
                }
            };
            let value: Value = match serde_json::from_slice(&bytes) {
                Ok(value) => value,
                Err(_) => {
                    if self
                        .reject(Value::Null, -32700, "Invalid JSON")
                        .await
                        .is_err()
                    {
                        return None;
                    }
                    continue;
                }
            };
            let id = value
                .get("id")
                .filter(|id| id.is_i64() || id.as_str().is_some_and(|s| s.len() <= 128))
                .cloned()
                .unwrap_or(Value::Null);
            if !valid_envelope(&value) {
                if value.get("id").is_none() && value.get("method").is_some() {
                    eprintln!("[MCP] Rejected notification with invalid or unknown fields");
                    continue;
                }
                if self
                    .reject(id, -32600, "Invalid request or unknown MCP fields")
                    .await
                    .is_err()
                {
                    return None;
                }
                continue;
            }
            if value.get("id").is_none() {
                match value["method"].as_str() {
                    Some("notifications/initialized") if !self.initialized_notification_seen => {
                        self.initialized_notification_seen = true;
                    }
                    Some("notifications/cancelled") => {
                        // Admission is serial; the prior request has already
                        // completed before another frame can be observed.
                        eprintln!("[MCP] Cancellation received after admitted work completed");
                        continue;
                    }
                    _ => {
                        eprintln!("[MCP] Ignored unsupported or repeated notification");
                        continue;
                    }
                }
            }
            let message: RxJsonRpcMessage<RoleServer> = match serde_json::from_value(value) {
                Ok(message) => message,
                Err(_) => {
                    if self
                        .reject(id, -32602, "Invalid MCP method or parameters")
                        .await
                        .is_err()
                    {
                        return None;
                    }
                    continue;
                }
            };
            if matches!(message, rmcp::model::JsonRpcMessage::Request(_)) {
                permit.forget();
            }
            return Some(message);
        }
    }
    async fn close(&mut self) -> io::Result<()> {
        self.output.lock().await.flush().await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn frames_are_bounded_and_require_newlines() {
        assert_eq!(frame(&mut &b"{}\n"[..]).await.unwrap().unwrap(), b"{}\n");
        assert!(frame(&mut &b"{}"[..]).await.is_err());
        assert!(frame(&mut vec![b'x'; MAX_INPUT + 1].as_slice())
            .await
            .is_err());
        assert!(frame(&mut &b""[..]).await.unwrap().is_none());
    }
    #[test]
    fn unknown_fields_are_rejected() {
        assert!(!valid_envelope(
            &json!({"jsonrpc":"2.0","id":1,"method":"tools/list","surprise":true})
        ));
        assert!(!valid_envelope(
            &json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"status","arguments":{},"extra":true}})
        ));
        assert!(valid_envelope(
            &json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"status","arguments":{}}})
        ));
    }
}
