//! How much GPU memory a training run needs per GPU, estimated from the child
//! model's shape (its parameter count from Hugging Face `safetensors.total`,
//! and its hidden size, layers and vocabulary from `config.json`) and the
//! `[training]` settings. The estimate is rough and on the safe side: it only
//! marks the GPU types that obviously cannot hold the run, and gives
//! `gpu_types = "auto"` a VRAM floor.
//!
//! Every GPU of a pod holds a whole copy of the model (data parallel), so the
//! estimate never divides by the GPU count.

use std::fmt;
use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;

use crate::config::{Adapter, Training};

/// The Hugging Face Hub the shape is read from.
pub const HF_URL: &str = "https://huggingface.co";

/// Bytes in a GB as Runpod counts VRAM (a 24 GB card holds 24 GiB).
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Memory neither the model nor its batch takes: CUDA context, allocator
/// fragmentation, buffers.
const OVERHEAD_GIB: f64 = 2.0;

/// The estimate is raised by this much, to stay on the safe side.
const MARGIN: f64 = 1.2;

/// A GPU with less than this much more than the estimate is `tight`.
const TIGHT: f64 = 1.1;

/// `LoRA` parameters per layer, per unit of rank and hidden size: the seven
/// linear modules `lora_target_linear` adapts add `r * (d_in + d_out)` each,
/// about 18 hidden sizes in all with grouped attention and a 3x MLP.
const LORA_WIDTHS: f64 = 18.0;

/// What the estimate needs to know about the child model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    /// Parameters, all of them.
    pub params: u64,
    /// Width of the hidden states.
    pub hidden_size: u64,
    /// Transformer layers.
    pub layers: u64,
    /// Tokens in the vocabulary.
    pub vocab_size: u64,
}

/// What the estimate needs to know about the training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recipe {
    /// The fine-tuning strategy.
    pub adapter: Adapter,
    /// Tokens per example.
    pub sequence_len: u32,
    /// Examples per GPU per step.
    pub micro_batch_size: u32,
    /// `LoRA` rank.
    pub lora_r: u32,
    /// Whether the optimizer keeps its state in 8 bits.
    pub eight_bit_optimizer: bool,
}

impl From<&Training> for Recipe {
    fn from(training: &Training) -> Self {
        Self {
            adapter: training.adapter,
            sequence_len: training.sequence_len,
            micro_batch_size: training.micro_batch_size,
            lora_r: training.lora_r,
            eight_bit_optimizer: training.optimizer.contains("8bit"),
        }
    }
}

/// The memory a run needs on each GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    bytes: f64,
}

impl Estimate {
    /// The estimate in GB (GiB, as Runpod counts VRAM).
    #[must_use]
    pub fn gib(&self) -> f64 {
        self.bytes / GIB
    }

    /// The least VRAM a GPU needs, in whole GB: the floor of `auto`.
    #[must_use]
    pub fn floor_gb(&self) -> u32 {
        // The least whole GB at or above the estimate, without a float cast.
        let gib = self.gib();
        let (mut low, mut high) = (0_u32, u32::MAX);
        while low < high {
            let middle = u32::midpoint(low, high);
            if f64::from(middle) < gib {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        low
    }
}

impl fmt::Display for Estimate {
    /// `about 19.4 GB`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "about {:.1} GB", self.gib())
    }
}

/// `value` as a float, exactly below 2^53, far above any model size.
fn float(value: u64) -> f64 {
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

/// The memory a run of `recipe` on a model of `shape` needs per GPU: the
/// weights (bf16, or about 0.6 bytes per parameter in 4 bits with QLoRA),
/// the gradients and `AdamW` state (16 bytes per trained parameter, 10 with
/// an 8-bit optimizer; only the adapter's with LoRA), the activations kept by
/// gradient checkpointing (one bf16 hidden state per token and layer), the
/// fp32 logits, then [`OVERHEAD_GIB`], all raised by [`MARGIN`].
#[must_use]
pub fn estimate(shape: &ModelShape, recipe: &Recipe) -> Estimate {
    let params = float(shape.params);
    let hidden = float(shape.hidden_size);
    let layers = float(shape.layers);
    let tokens = f64::from(recipe.micro_batch_size) * f64::from(recipe.sequence_len);
    let optimizer = if recipe.eight_bit_optimizer { 10.0 } else { 16.0 };
    let (weights, trained) = match recipe.adapter {
        Adapter::Full => (2.0 * params, params),
        Adapter::Lora => (2.0 * params, lora_params(shape, recipe)),
        Adapter::Qlora => (0.6 * params, lora_params(shape, recipe)),
    };
    let activations = tokens * hidden * layers * 2.0;
    let logits = tokens * float(shape.vocab_size) * 4.0;
    let bytes = weights + trained * optimizer + activations + logits + OVERHEAD_GIB * GIB;
    Estimate {
        bytes: bytes * MARGIN,
    }
}

/// The parameters a `LoRA` adapter of rank `recipe.lora_r` trains.
fn lora_params(shape: &ModelShape, recipe: &Recipe) -> f64 {
    float(shape.layers) * f64::from(recipe.lora_r) * float(shape.hidden_size) * LORA_WIDTHS
}

/// Whether a GPU holds the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// It holds the estimate with room to spare.
    Ok,
    /// It holds the estimate with less than 10% to spare.
    Tight,
    /// It has less memory than the estimate.
    Small,
    /// The estimate is unknown.
    Unknown,
}

impl Fit {
    /// Its name in a FIT column: `ok`, `tight`, `small` or `?`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Tight => "tight",
            Self::Small => "small",
            Self::Unknown => "?",
        }
    }
}

/// Whether a GPU with `memory_gb` of VRAM holds a run needing `need`.
#[must_use]
pub fn fit(memory_gb: u32, need: Option<&Estimate>) -> Fit {
    let Some(need) = need else {
        return Fit::Unknown;
    };
    let memory = f64::from(memory_gb);
    if memory < need.gib() {
        Fit::Small
    } else if memory < need.gib() * TIGHT {
        Fit::Tight
    } else {
        Fit::Ok
    }
}

/// Whether `model` reads as a Hugging Face repo ID (`org/name` or `name`),
/// not a path on the training target: letters, digits, `.`, `_` and `-`, at
/// most one `/` between two names, none starting with `.`.
#[must_use]
pub fn is_repo_id(model: &str) -> bool {
    let parts: Vec<&str> = model.split('/').collect();
    parts.len() <= 2
        && parts.iter().all(|part| {
            !part.is_empty()
                && !part.starts_with('.')
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        })
}

/// The shape of the Hugging Face model `model`, read from the Hub at
/// `base_url` (see [`HF_URL`]) with `token` as bearer when given, giving up
/// after `limit`.
///
/// # Errors
///
/// Returns why the shape is unknown: `model` is not a repo ID, the Hub
/// cannot be reached or answers an error, or a field is missing. The message
/// names the model and the status, never the token.
pub async fn fetch_shape(
    base_url: &str,
    model: &str,
    token: Option<&SecretString>,
    limit: Duration,
) -> Result<ModelShape, String> {
    if !is_repo_id(model) {
        return Err(format!("{model} is not a Hugging Face repo ID"));
    }
    let lookup = async {
        let http = reqwest::Client::builder()
            .user_agent(concat!("overbrainer/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|_| "cannot build an HTTP client".to_string())?;
        let base = base_url.trim_end_matches('/');
        let info = get_json(
            &http,
            &format!("{base}/api/models/{model}?expand[]=safetensors"),
            token,
            model,
        )
        .await?;
        let config = get_json(
            &http,
            &format!("{base}/{model}/resolve/main/config.json"),
            token,
            model,
        )
        .await?;
        shape(&info, &config).ok_or_else(|| format!("{model} does not give its size on Hugging Face"))
    };
    tokio::time::timeout(limit, lookup)
        .await
        .unwrap_or_else(|_| Err(format!("Hugging Face took too long to describe {model}")))
}

/// The JSON at `url`, or why not, in words naming `model`.
async fn get_json(
    http: &reqwest::Client,
    url: &str,
    token: Option<&SecretString>,
    model: &str,
) -> Result<Value, String> {
    let mut request = http.get(url);
    if let Some(token) = token {
        request = request.bearer_auth(token.expose_secret());
    }
    let unreachable = |_| format!("cannot reach Hugging Face for {model}");
    let response = request.send().await.map_err(unreachable)?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!(
            "Hugging Face answered {} for {model}",
            status.as_u16()
        ));
    }
    response
        .json()
        .await
        .map_err(|_| format!("cannot read what Hugging Face says of {model}"))
}

/// The shape from the model's Hub `info` (`safetensors.total`) and its
/// `config.json`, looked up in its `text_config` for a multimodal model.
fn shape(info: &Value, config: &Value) -> Option<ModelShape> {
    let params = info.pointer("/safetensors/total")?.as_u64()?;
    let field = |names: &[&str]| {
        [config, config.get("text_config").unwrap_or(&Value::Null)]
            .into_iter()
            .find_map(|table| names.iter().find_map(|name| table.get(*name)?.as_u64()))
    };
    Some(ModelShape {
        params,
        hidden_size: field(&["hidden_size", "n_embd", "d_model"])?,
        layers: field(&["num_hidden_layers", "n_layer", "num_layers"])?,
        vocab_size: field(&["vocab_size"])?,
    })
}

/// What a run of `training` needs per GPU, its model's shape read from the
/// Hub at `base_url` (see [`fetch_shape`]).
///
/// # Errors
///
/// Returns why the shape is unknown (see [`fetch_shape`]).
pub async fn estimate_model(
    training: &Training,
    token: Option<&SecretString>,
    base_url: &str,
    limit: Duration,
) -> Result<Estimate, String> {
    let shape = fetch_shape(base_url, &training.base_model, token, limit).await?;
    Ok(estimate(&shape, &Recipe::from(training)))
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    /// Qwen3-4B, from the Hub.
    const QWEN3_4B: ModelShape = ModelShape {
        params: 4_022_468_096,
        hidden_size: 2560,
        layers: 36,
        vocab_size: 151_936,
    };

    /// Qwen3-8B, from the Hub.
    const QWEN3_8B: ModelShape = ModelShape {
        params: 8_190_735_360,
        hidden_size: 4096,
        layers: 36,
        vocab_size: 151_936,
    };

    /// Mistral-7B-v0.1, from the Hub.
    const MISTRAL_7B: ModelShape = ModelShape {
        params: 7_241_732_096,
        hidden_size: 4096,
        layers: 32,
        vocab_size: 32_000,
    };

    fn recipe(adapter: Adapter) -> Recipe {
        Recipe {
            adapter,
            sequence_len: 4096,
            micro_batch_size: 2,
            lora_r: 16,
            eight_bit_optimizer: false,
        }
    }

    #[test]
    fn the_estimate_follows_the_model_and_the_adapter() {
        let cases = [
            (QWEN3_4B, recipe(Adapter::Lora), 20),
            (QWEN3_8B, recipe(Adapter::Qlora), 17),
            (MISTRAL_7B, recipe(Adapter::Full), 152),
        ];
        for (shape, recipe, floor) in cases {
            let need = estimate(&shape, &recipe);
            assert_eq!(need.floor_gb(), floor, "{shape:?} {recipe:?}: {need}");
        }
        assert_eq!(
            estimate(&QWEN3_4B, &recipe(Adapter::Lora)).to_string(),
            "about 19.1 GB"
        );
    }

    #[test]
    fn an_eight_bit_optimizer_and_a_shorter_batch_need_less() {
        let full = estimate(&MISTRAL_7B, &recipe(Adapter::Full)).gib();
        let eight = Recipe {
            eight_bit_optimizer: true,
            ..recipe(Adapter::Full)
        };
        assert!(estimate(&MISTRAL_7B, &eight).gib() < full - 40.0);
        let short = Recipe {
            sequence_len: 1024,
            ..recipe(Adapter::Lora)
        };
        assert!(
            estimate(&QWEN3_4B, &short).gib() < estimate(&QWEN3_4B, &recipe(Adapter::Lora)).gib()
        );
    }

    #[test]
    fn the_recipe_reads_the_training_settings() -> Result<(), Box<dyn std::error::Error>> {
        let training: Training = serde_json::from_value(json!({
            "target": "cloud", "base_model": "Qwen/Qwen3-4B", "adapter": "qlora",
            "optimizer": "paged_adamw_8bit", "sequence_len": 2048
        }))?;
        assert_eq!(
            Recipe::from(&training),
            Recipe {
                adapter: Adapter::Qlora,
                sequence_len: 2048,
                micro_batch_size: 2,
                lora_r: 16,
                eight_bit_optimizer: true,
            }
        );
        Ok(())
    }

    #[test]
    fn fit_is_ok_tight_small_or_unknown() {
        let need = estimate(&QWEN3_4B, &recipe(Adapter::Lora));
        assert_eq!(fit(48, Some(&need)), Fit::Ok);
        assert_eq!(fit(20, Some(&need)), Fit::Tight);
        assert_eq!(fit(16, Some(&need)), Fit::Small);
        assert_eq!(fit(80, None), Fit::Unknown);
        let names: Vec<&str> = [Fit::Ok, Fit::Tight, Fit::Small, Fit::Unknown]
            .into_iter()
            .map(Fit::name)
            .collect();
        assert_eq!(names, ["ok", "tight", "small", "?"]);
    }

    #[test]
    fn a_path_is_not_a_repo_id() {
        for model in ["Qwen/Qwen3-4B", "gpt2", "org/model_v1.5"] {
            assert!(is_repo_id(model), "{model}");
        }
        for model in [
            "/models/qwen",
            "./qwen",
            "~/qwen",
            "a/b/c",
            "../x",
            "org/.hidden",
            "",
            "a?b",
        ] {
            assert!(!is_repo_id(model), "{model}");
        }
    }

    /// A Hub at `server` describing `Qwen/Qwen3-4B`, the model info needing
    /// `bearer` when given.
    async fn hub(server: &MockServer, config: Value) {
        Mock::given(method("GET"))
            .and(path("/api/models/Qwen/Qwen3-4B"))
            .and(query_param("expand[]", "safetensors"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "Qwen/Qwen3-4B",
                "safetensors": {"parameters": {"BF16": 4_022_468_096_u64}, "total": 4_022_468_096_u64}
            })))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/Qwen/Qwen3-4B/resolve/main/config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(config))
            .mount(server)
            .await;
    }

    const LIMIT: Duration = Duration::from_secs(5);

    #[tokio::test]
    async fn the_shape_is_read_from_the_hub() -> Result<(), String> {
        let server = MockServer::start().await;
        hub(
            &server,
            json!({"hidden_size": 2560, "num_hidden_layers": 36, "vocab_size": 151_936}),
        )
        .await;
        let shape = fetch_shape(&server.uri(), "Qwen/Qwen3-4B", None, LIMIT).await?;
        assert_eq!(shape, QWEN3_4B);
        Ok(())
    }

    #[tokio::test]
    async fn a_multimodal_config_gives_its_text_model() -> Result<(), String> {
        let server = MockServer::start().await;
        hub(
            &server,
            json!({"text_config": {"hidden_size": 2560, "num_hidden_layers": 36,
                                   "vocab_size": 151_936}}),
        )
        .await;
        let shape = fetch_shape(&server.uri(), "Qwen/Qwen3-4B", None, LIMIT).await?;
        assert_eq!(shape, QWEN3_4B);
        Ok(())
    }

    #[tokio::test]
    async fn the_token_goes_as_a_bearer_and_never_in_an_error() -> Result<(), String> {
        let server = MockServer::start().await;
        let secret = "hf_sizing_secret_42";
        Mock::given(method("GET"))
            .and(header("authorization", format!("Bearer {secret}").as_str()))
            .respond_with(ResponseTemplate::new(401).set_body_string(secret))
            .expect(1)
            .mount(&server)
            .await;
        let token = SecretString::from(secret);
        let error = fetch_shape(&server.uri(), "Qwen/Qwen3-4B", Some(&token), LIMIT)
            .await
            .err()
            .ok_or("no error")?;
        assert_eq!(error, "Hugging Face answered 401 for Qwen/Qwen3-4B");
        Ok(())
    }

    #[tokio::test]
    async fn a_model_without_safetensors_has_no_shape() -> Result<(), String> {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/Qwen/Qwen3-4B"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"id": "Qwen/Qwen3-4B"})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/Qwen/Qwen3-4B/resolve/main/config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .mount(&server)
            .await;
        let error = fetch_shape(&server.uri(), "Qwen/Qwen3-4B", None, LIMIT).await;
        assert_eq!(
            error,
            Err("Qwen/Qwen3-4B does not give its size on Hugging Face".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_path_is_never_asked_for() {
        let server = MockServer::start().await;
        let error = fetch_shape(&server.uri(), "/models/qwen", None, LIMIT).await;
        assert_eq!(
            error,
            Err("/models/qwen is not a Hugging Face repo ID".to_string())
        );
        let asked = server.received_requests().await.unwrap_or_default();
        assert!(asked.is_empty());
    }

    #[tokio::test]
    async fn a_slow_hub_is_given_up() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))
            .mount(&server)
            .await;
        let error = fetch_shape(
            &server.uri(),
            "Qwen/Qwen3-4B",
            None,
            Duration::from_millis(200),
        )
        .await;
        assert_eq!(
            error,
            Err("Hugging Face took too long to describe Qwen/Qwen3-4B".to_string())
        );
    }
}
