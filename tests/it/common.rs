//! Helpers shared by the Runpod and SSH test files. Each file uses only some of them.

use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};
use wiremock::matchers::path_regex;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// A stub of Runpod's `/account/secrets`: create (409 when the name is taken),
/// list (all, or by `name`), rotate and delete, with what it holds and every
/// value it was sent kept for the assertions.
#[derive(Clone, Default)]
pub struct SecretStore {
    state: Arc<Mutex<State>>,
}

#[derive(Default)]
struct State {
    /// `(id, name)` of each secret held.
    held: Vec<(String, String)>,
    /// Every value a create or a rotation sent.
    values: Vec<String>,
    /// Names of the secrets deleted.
    deleted: Vec<String>,
    created: usize,
}

impl SecretStore {
    /// Mounts a new store on `server`.
    pub async fn mount(server: &MockServer) -> Self {
        let store = Self::default();
        Mock::given(path_regex("^/v2/account/secrets"))
            .respond_with(store.clone())
            .mount(server)
            .await;
        store
    }

    /// Holds the secret `name` from the start, as `id`.
    pub fn hold(&self, id: &str, name: &str) {
        self.lock().held.push((id.to_string(), name.to_string()));
    }

    /// The names of the secrets held.
    pub fn names(&self) -> Vec<String> {
        self.lock()
            .held
            .iter()
            .map(|(_, name)| name.clone())
            .collect()
    }

    /// The names of the secrets deleted.
    pub fn deleted(&self) -> Vec<String> {
        self.lock().deleted.clone()
    }

    /// Every value sent to the store.
    pub fn values(&self) -> Vec<String> {
        self.lock().values.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn secret(id: &str, name: &str) -> Value {
    json!({"id": id, "name": name, "createdAt": "2026-10-01T00:00:00Z"})
}

fn not_found() -> ResponseTemplate {
    ResponseTemplate::new(404)
        .set_body_json(json!({"detail": "secret not found", "status": 404, "title": "Not Found"}))
}

impl Respond for SecretStore {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let mut state = self.lock();
        let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        if let Some(value) = body.get("value").and_then(Value::as_str) {
            state.values.push(value.to_string());
        }
        let id = request.url.path().strip_prefix("/v2/account/secrets/");
        match (request.method.as_str(), id) {
            ("POST", None) => {
                let name = body["name"].as_str().unwrap_or_default().to_string();
                if state.held.iter().any(|(_, held)| *held == name) {
                    return ResponseTemplate::new(409).set_body_json(json!({
                        "detail": "a secret with this name already exists", "status": 409, "title": "Conflict"
                    }));
                }
                state.created += 1;
                let id = format!("s{}", state.created);
                state.held.push((id.clone(), name.clone()));
                ResponseTemplate::new(201).set_body_json(secret(&id, &name))
            },
            ("GET", None) => {
                let wanted = request
                    .url
                    .query_pairs()
                    .find(|(key, _)| key == "name")
                    .map(|(_, value)| value.to_lowercase());
                let secrets: Vec<Value> = state
                    .held
                    .iter()
                    .filter(|(_, name)| wanted.as_ref().is_none_or(|w| *w == name.to_lowercase()))
                    .map(|(id, name)| secret(id, name))
                    .collect();
                ResponseTemplate::new(200).set_body_json(json!({"secrets": secrets}))
            },
            ("PATCH", Some(id)) => match state.held.iter().find(|(held, _)| held == id) {
                Some((id, name)) => ResponseTemplate::new(200).set_body_json(secret(id, name)),
                None => not_found(),
            },
            ("DELETE", Some(id)) => match state.held.iter().position(|(held, _)| held == id) {
                Some(index) => {
                    let (_, name) = state.held.remove(index);
                    state.deleted.push(name);
                    ResponseTemplate::new(204)
                },
                None => not_found(),
            },
            _ => ResponseTemplate::new(405),
        }
    }
}

/// The SSH clients of this build: OpenSSH, then the built-in client when the
/// `builtin-ssh` feature is on. `OVERBRAINER_TEST_SSH_CLIENT` (`openssh` or
/// `builtin`) keeps only that one, for a run whose `PATH` suits one client.
///
/// # Errors
///
/// Returns an error when `OVERBRAINER_TEST_SSH_CLIENT` names no client of this
/// build, so a misspelt value fails the tests rather than passing them with
/// nothing run.
pub fn ssh_clients() -> Result<Vec<overbrainer::config::SshClient>, Box<dyn std::error::Error>> {
    use overbrainer::config::{HAS_BUILTIN_SSH, SshClient};
    let all = if HAS_BUILTIN_SSH {
        vec![SshClient::Openssh, SshClient::Builtin]
    } else {
        vec![SshClient::Openssh]
    };
    match std::env::var("OVERBRAINER_TEST_SSH_CLIENT") {
        Ok(only) if !only.is_empty() => {
            let kept: Vec<SshClient> = all
                .into_iter()
                .filter(|client| client.name() == only)
                .collect();
            if kept.is_empty() {
                return Err(format!(
                    "OVERBRAINER_TEST_SSH_CLIENT={only} names no SSH client of this build"
                )
                .into());
            }
            Ok(kept)
        },
        _ => Ok(all),
    }
}

/// Runs `case` once for each of [`ssh_clients`], stopping at the first failure,
/// whose error names the client. The client is also printed before each case,
/// so a failed assertion's captured output names it.
///
/// # Errors
///
/// Returns the error of [`ssh_clients`], or the first failing case's error
/// with the client's name in front.
pub async fn each_ssh_client<F, Fut>(case: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(overbrainer::config::SshClient) -> Fut,
    Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error>>>,
{
    for client in ssh_clients()? {
        eprintln!("ssh_client = {}", client.name());
        case(client)
            .await
            .map_err(|error| format!("with ssh_client = {}: {error}", client.name()))?;
    }
    Ok(())
}

/// A fake `llama-server` (Python), put in the llama.cpp cache of a target by
/// the compare tests: answers `/health` and streams `Child: <question>`, or
/// `I do not know.` to a question with `hard` in it.
pub const FAKE_LLAMA_SERVER: &str = r#"#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        question = body["messages"][-1]["content"]
        answer = "I do not know." if "hard" in question else "Child: " + question
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in [
            {"choices": [{"delta": {"content": answer}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 4}},
        ]:
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"#;

/// The llama.cpp build directories of a target's tools cache that may serve
/// the child, whichever the machine picks: the CPU and CUDA builds.
pub const LLAMA_ASSETS: [&str; 5] = [
    "ubuntu-x64",
    "ubuntu-arm64",
    "macos-arm64",
    "ubuntu-cuda-13.4-x64",
    "ubuntu-cuda-13.4-arm64",
];
