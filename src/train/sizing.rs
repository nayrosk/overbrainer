//! How much GPU memory a training run needs per GPU, estimated from the child
//! model's shape (its parameter count from Hugging Face `safetensors.total`,
//! and its hidden size, layers and vocabulary from `config.json`) and the
//! `[training]` settings, `axolotl_extra` included. The estimate is rough and
//! on the safe side: it only marks the GPU types that obviously cannot hold
//! the run, and gives `gpu_types = "auto"` a VRAM floor.
//!
//! Every GPU of a pod holds a whole copy of the model (data parallel), so the
//! estimate never divides by the GPU count; a run that shards the model
//! (`deepspeed` or `fsdp` in `axolotl_extra`) gets no estimate.

use std::fmt;
use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;

use crate::config::{Adapter, Training};

/// The Hugging Face Hub the shape is read from.
pub const HF_URL: &str = "https://huggingface.co";

/// Bytes in a GB of the Runpod catalog. Its VRAM figures are the vendors' (an
/// A40 "48 GB" gives about 45 GiB to CUDA), so a GB counts as 10^9 bytes, the
/// smaller unit, to stay on the safe side.
const GB: u64 = 1_000_000_000;

/// Memory neither the model nor its batch takes: CUDA context, allocator
/// fragmentation, buffers.
const OVERHEAD: u64 = 2 * GB;

/// `LoRA` parameters per layer, per unit of rank and hidden size: the seven
/// linear modules `lora_target_linear` adapts add `r * (d_in + d_out)` each,
/// about 18 hidden sizes in all with grouped attention and a 3x MLP.
const LORA_WIDTHS: u128 = 18;

/// Bytes each token keeps per layer and unit of hidden size for the
/// backward pass: one bf16 hidden state with gradient checkpointing, about 34
/// bytes without (attention and MLP inputs, masks, dropout).
const CHECKPOINTED_BYTES: u128 = 2;
const FULL_ACTIVATION_BYTES: u128 = 34;

/// The `axolotl_extra` keys that shard the model across the GPUs.
const SHARDING: [&str; 3] = ["deepspeed", "fsdp", "fsdp_config"];

/// The VRAM floor of a run's `auto` GPU types, as whoever starts the run
/// knows it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VramFloor {
    /// Not estimated yet: the run estimates it when it starts.
    #[default]
    ToEstimate,
    /// Estimated already (the TUI's start confirmation shows it): this floor
    /// in GB, or none when the estimate could not be made.
    Known(Option<u32>),
}

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
    /// Whether only one hidden state per layer is kept for the backward pass.
    pub gradient_checkpointing: bool,
}

impl Recipe {
    /// The recipe of `training`, read as Axolotl gets it: `micro_batch_size`,
    /// `sequence_len`, `optimizer` and `gradient_checkpointing` from
    /// `axolotl_extra` when it sets them, else from `[training]`.
    ///
    /// # Errors
    ///
    /// Returns why there is no estimate: `axolotl_extra` shards the model
    /// (`deepspeed`, `fsdp` or `fsdp_config`), or sets one of those fields to
    /// a value of the wrong type.
    pub fn of(training: &Training) -> Result<Self, String> {
        let extra = |key: &str| {
            training
                .axolotl_extra
                .get(key)
                .filter(|value| !value.is_null())
        };
        if let Some(key) = SHARDING.into_iter().find(|key| extra(key).is_some()) {
            return Err(format!(
                "axolotl_extra shards the model ({key}): each GPU holds only part of it"
            ));
        }
        let count = |key: &str, default: u32| match extra(key) {
            None => Ok(default),
            Some(value) => value
                .as_u64()
                .and_then(|count| u32::try_from(count).ok())
                .ok_or_else(|| format!("axolotl_extra.{key} is not a count")),
        };
        let optimizer = match extra("optimizer") {
            None => training.optimizer.as_str(),
            Some(value) => value
                .as_str()
                .ok_or("axolotl_extra.optimizer is not a name")?,
        };
        let gradient_checkpointing =
            extra("gradient_checkpointing").is_none_or(|value| *value != Value::Bool(false));
        Ok(Self {
            adapter: training.adapter,
            sequence_len: count("sequence_len", training.sequence_len)?,
            micro_batch_size: count("micro_batch_size", training.micro_batch_size)?,
            lora_r: training.lora_r,
            eight_bit_optimizer: optimizer.contains("8bit"),
            gradient_checkpointing,
        })
    }
}

/// The memory a run needs on each GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Estimate {
    bytes: u64,
}

impl Estimate {
    /// The least VRAM a GPU needs, in whole GB of the Runpod catalog: the
    /// floor of `auto`.
    #[must_use]
    pub fn floor_gb(&self) -> u32 {
        u32::try_from(self.bytes.div_ceil(GB)).unwrap_or(u32::MAX)
    }
}

impl fmt::Display for Estimate {
    /// `about 20.5 GB`, in GB of the Runpod catalog.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tenths = self.bytes.div_ceil(GB / 10);
        write!(f, "about {}.{} GB", tenths / 10, tenths % 10)
    }
}

/// The memory a run of `recipe` on a model of `shape` needs per GPU: the
/// weights (bf16, or about 0.6 bytes per parameter in 4 bits with `QLoRA`),
/// the gradients and `AdamW` state (16 bytes per trained parameter, 10 with
/// an 8-bit optimizer; only the adapter's with `LoRA`), the activations kept
/// for the backward pass (one bf16 hidden state per token and layer with
/// gradient checkpointing, about 17 times that without), the fp32 logits,
/// then 2 GB of overhead, all raised by 20%.
#[must_use]
pub fn estimate(shape: &ModelShape, recipe: &Recipe) -> Estimate {
    let params = u128::from(shape.params);
    let hidden = u128::from(shape.hidden_size);
    let layers = u128::from(shape.layers);
    let tokens = u128::from(recipe.micro_batch_size) * u128::from(recipe.sequence_len);
    let optimizer = if recipe.eight_bit_optimizer { 10 } else { 16 };
    let lora = layers * u128::from(recipe.lora_r) * hidden * LORA_WIDTHS;
    let (weights, trained) = match recipe.adapter {
        Adapter::Full => (2 * params, params),
        Adapter::Lora => (2 * params, lora),
        Adapter::Qlora => (params * 6 / 10, lora),
    };
    let kept = if recipe.gradient_checkpointing {
        CHECKPOINTED_BYTES
    } else {
        FULL_ACTIVATION_BYTES
    };
    let activations = tokens * hidden * layers * kept;
    let logits = tokens * u128::from(shape.vocab_size) * 4;
    let bytes = weights + trained * optimizer + activations + logits + u128::from(OVERHEAD);
    Estimate {
        bytes: u64::try_from(bytes * 12 / 10).unwrap_or(u64::MAX),
    }
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

/// Whether a GPU with `memory_gb` of VRAM, as the Runpod catalog counts it,
/// holds a run needing `need`.
#[must_use]
pub fn fit(memory_gb: u32, need: Option<&Estimate>) -> Fit {
    let Some(need) = need else {
        return Fit::Unknown;
    };
    let memory = u128::from(memory_gb) * u128::from(GB);
    let need = u128::from(need.bytes);
    if memory < need {
        Fit::Small
    } else if memory * 10 < need * 11 {
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
        shape(&info, &config)
            .ok_or_else(|| format!("{model} does not give its size on Hugging Face"))
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
/// Returns why there is no estimate (see [`Recipe::of`] and [`fetch_shape`]).
pub async fn estimate_model(
    training: &Training,
    token: Option<&SecretString>,
    base_url: &str,
    limit: Duration,
) -> Result<Estimate, String> {
    let recipe = Recipe::of(training)?;
    let shape = fetch_shape(base_url, &training.base_model, token, limit).await?;
    Ok(estimate(&shape, &recipe))
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
            gradient_checkpointing: true,
        }
    }

    #[test]
    fn the_estimate_follows_the_model_and_the_adapter() {
        let cases = [
            (QWEN3_4B, recipe(Adapter::Lora), 21),
            (QWEN3_8B, recipe(Adapter::Qlora), 18),
            (MISTRAL_7B, recipe(Adapter::Full), 163),
        ];
        for (shape, recipe, floor) in cases {
            let need = estimate(&shape, &recipe);
            assert_eq!(need.floor_gb(), floor, "{shape:?} {recipe:?}: {need}");
        }
        assert_eq!(
            estimate(&QWEN3_4B, &recipe(Adapter::Lora)).to_string(),
            "about 20.4 GB"
        );
    }

    #[test]
    fn an_eight_bit_optimizer_a_shorter_batch_and_checkpointing_need_less() {
        let need = |shape: &ModelShape, recipe: &Recipe| estimate(shape, recipe).floor_gb();
        let full = need(&MISTRAL_7B, &recipe(Adapter::Full));
        let eight = Recipe {
            eight_bit_optimizer: true,
            ..recipe(Adapter::Full)
        };
        assert!(need(&MISTRAL_7B, &eight) + 40 < full);
        let lora = need(&QWEN3_4B, &recipe(Adapter::Lora));
        let short = Recipe {
            sequence_len: 1024,
            ..recipe(Adapter::Lora)
        };
        assert!(need(&QWEN3_4B, &short) < lora);
        let unchecked = Recipe {
            gradient_checkpointing: false,
            ..recipe(Adapter::Lora)
        };
        assert!(need(&QWEN3_4B, &unchecked) > lora + 20);
    }

    #[test]
    fn floor_gb_rounds_up_to_a_whole_gb() {
        let at = |bytes| Estimate { bytes }.floor_gb();
        assert_eq!(at(20 * GB), 20);
        assert_eq!(at(20 * GB + 1), 21);
        assert_eq!(at(0), 0);
        assert_eq!(at(u64::MAX), u32::MAX);
        assert_eq!(Estimate { bytes: 20 * GB }.to_string(), "about 20.0 GB");
        assert_eq!(Estimate { bytes: 20 * GB + 1 }.to_string(), "about 20.1 GB");
    }

    fn training(extra: &Value) -> Result<Training, serde_json::Error> {
        serde_json::from_value(json!({
            "target": "cloud", "base_model": "Qwen/Qwen3-4B", "adapter": "qlora",
            "optimizer": "paged_adamw_8bit", "sequence_len": 2048, "axolotl_extra": extra
        }))
    }

    #[test]
    fn the_recipe_reads_the_training_settings() -> Result<(), Box<dyn std::error::Error>> {
        let expected = Recipe {
            adapter: Adapter::Qlora,
            sequence_len: 2048,
            micro_batch_size: 2,
            lora_r: 16,
            eight_bit_optimizer: true,
            gradient_checkpointing: true,
        };
        assert_eq!(Recipe::of(&training(&json!({}))?), Ok(expected));
        Ok(())
    }

    #[test]
    fn axolotl_extra_goes_before_the_training_settings() -> Result<(), Box<dyn std::error::Error>> {
        let extra = json!({"micro_batch_size": 8, "sequence_len": 8192,
                           "optimizer": "adamw_torch_fused", "gradient_checkpointing": false});
        assert_eq!(
            Recipe::of(&training(&extra)?),
            Ok(Recipe {
                adapter: Adapter::Qlora,
                sequence_len: 8192,
                micro_batch_size: 8,
                lora_r: 16,
                eight_bit_optimizer: false,
                gradient_checkpointing: false,
            })
        );
        // Axolotl's own "unsloth" checkpointing still checkpoints.
        let unsloth = Recipe::of(&training(&json!({"gradient_checkpointing": "unsloth"}))?)?;
        assert!(unsloth.gradient_checkpointing);
        assert_eq!(
            Recipe::of(&training(&json!({"micro_batch_size": "eight"}))?),
            Err("axolotl_extra.micro_batch_size is not a count".to_string())
        );
        Ok(())
    }

    #[test]
    fn a_sharded_run_has_no_estimate() -> Result<(), Box<dyn std::error::Error>> {
        for key in ["deepspeed", "fsdp", "fsdp_config"] {
            let recipe = Recipe::of(&training(&json!({key: "zero3.json"}))?);
            assert_eq!(
                recipe,
                Err(format!(
                    "axolotl_extra shards the model ({key}): each GPU holds only part of it"
                ))
            );
        }
        assert!(Recipe::of(&training(&json!({"deepspeed": null}))?).is_ok());
        Ok(())
    }

    #[test]
    fn fit_is_ok_tight_small_or_unknown() {
        let need = Estimate { bytes: 20 * GB };
        assert_eq!(fit(48, Some(&need)), Fit::Ok);
        assert_eq!(fit(22, Some(&need)), Fit::Ok, "exactly 10% to spare");
        assert_eq!(fit(21, Some(&need)), Fit::Tight);
        assert_eq!(fit(20, Some(&need)), Fit::Tight, "exactly the estimate");
        assert_eq!(fit(19, Some(&need)), Fit::Small);
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
    async fn a_multimodal_config_may_keep_its_vocabulary_at_the_top() -> Result<(), String> {
        let server = MockServer::start().await;
        hub(
            &server,
            json!({"vocab_size": 151_936,
                   "text_config": {"hidden_size": 2560, "num_hidden_layers": 36}}),
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
