//! Helpers shared by the Runpod test files. Each file uses only some of them.
#![allow(dead_code)]

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
