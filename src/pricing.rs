//! Model prices read from the provider's `/models` listing, when it exposes them.

use std::time::Duration;

use serde_json::Value;

use crate::llm::{LlmError, ProtocolClient, Usage};

/// Price of a model in USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    /// USD per million input tokens.
    pub input_per_million: f64,
    /// USD per million output tokens.
    pub output_per_million: f64,
}

impl Price {
    /// Cost of `usage` in USD.
    #[must_use]
    pub fn cost(&self, usage: Usage) -> f64 {
        (tokens(usage.input_tokens) * self.input_per_million
            + tokens(usage.output_tokens) * self.output_per_million)
            / 1_000_000.0
    }
}

/// Converts a token count to `f64`, exactly below 2^53, far above any real count.
fn tokens(count: u64) -> f64 {
    let high = u32::try_from(count >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(count & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

/// Finds the price of `model` in a `/models` answer.
///
/// Two shapes are known: `OpenRouter` sends decimal strings in USD per token,
/// `NanoGPT` (with `?detailed=true`) sends numbers in USD per million tokens with
/// `unit = "per_million_tokens"`. Anything else, including negative prices used for
/// variable-price routers, is unknown.
#[must_use]
pub fn find_price(models: &Value, model: &str) -> Option<Price> {
    let entries = models
        .get("data")
        .and_then(Value::as_array)
        .or_else(|| models.as_array())?;
    let entry = entries
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model))?;
    let pricing = entry.get("pricing")?;
    let per_million = pricing.get("unit").and_then(Value::as_str) == Some("per_million_tokens");
    let rate = |field: &str| -> Option<f64> {
        let value = match pricing.get(field)? {
            Value::String(text) => text.trim().parse::<f64>().ok()? * 1_000_000.0,
            Value::Number(number) if per_million => number.as_f64()?,
            _ => return None,
        };
        (value.is_finite() && value >= 0.0).then_some(value)
    };
    Some(Price {
        input_per_million: rate("prompt")?,
        output_per_million: rate("completion")?,
    })
}

/// Longest wait for a provider's model listing before costs are shown as unknown.
pub const LISTING_TIMEOUT: Duration = Duration::from_secs(10);

/// Reads the model listing of `client`'s provider, giving up after `cap` (the CLI
/// uses [`LISTING_TIMEOUT`]). Never fails: an unreachable, failing or slow listing
/// only means costs are not shown, which is logged.
pub async fn fetch_listing(client: &ProtocolClient, cap: Duration) -> Option<Value> {
    match tokio::time::timeout(cap, client.models()).await {
        Ok(Ok(models)) => Some(models),
        Ok(Err(error)) => unavailable(&error),
        Err(_) => too_slow(cap),
    }
}

/// Price of `model` in a listing read by [`fetch_listing`], logging when it has none.
#[must_use]
pub fn listed_price(listing: &Value, model: &str) -> Option<Price> {
    find_price(listing, model).or_else(|| unlisted(model))
}

fn unlisted(model: &str) -> Option<Price> {
    tracing::info!("no price listed for model {model}; showing tokens only");
    None
}

fn unavailable(error: &LlmError) -> Option<Value> {
    tracing::warn!("cannot read model prices ({error}); showing tokens only");
    None
}

fn too_slow(cap: Duration) -> Option<Value> {
    tracing::warn!(
        "model prices not received within {}s; showing tokens only",
        cap.as_secs()
    );
    None
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn openrouter_prices_are_per_token_strings() {
        let models = json!({"data": [{
            "id": "anthropic/claude-sonnet-5",
            "pricing": {"prompt": "0.000002", "completion": "0.00001", "web_search": "0.01"}
        }]});
        let price = find_price(&models, "anthropic/claude-sonnet-5");
        assert!(
            price.is_some_and(
                |p| close(p.input_per_million, 2.0) && close(p.output_per_million, 10.0)
            )
        );
    }

    #[test]
    fn nanogpt_prices_are_per_million_numbers() {
        let models = json!({"object": "list", "data": [{
            "id": "anthropic/claude-sonnet-5",
            "pricing": {"prompt": 2, "completion": 10, "currency": "USD", "unit": "per_million_tokens"}
        }]});
        let price = find_price(&models, "anthropic/claude-sonnet-5");
        assert!(
            price.is_some_and(
                |p| close(p.input_per_million, 2.0) && close(p.output_per_million, 10.0)
            )
        );
    }

    #[test]
    fn unknown_shapes_have_no_price() {
        let openai = json!({"object": "list", "data": [{"id": "gpt-x", "object": "model", "created": 1, "owned_by": "openai"}]});
        assert_eq!(find_price(&openai, "gpt-x"), None);
        let numbers_without_unit =
            json!({"data": [{"id": "m", "pricing": {"prompt": 2, "completion": 10}}]});
        assert_eq!(find_price(&numbers_without_unit, "m"), None);
        let router = json!({"data": [{"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}}]});
        assert_eq!(find_price(&router, "openrouter/auto"), None);
        let other_model =
            json!({"data": [{"id": "a", "pricing": {"prompt": "0.1", "completion": "0.1"}}]});
        assert_eq!(find_price(&other_model, "b"), None);
    }

    #[test]
    fn cost_multiplies_tokens_by_price() {
        let price = Price {
            input_per_million: 2.0,
            output_per_million: 10.0,
        };
        let usage = Usage {
            input_tokens: 500_000,
            output_tokens: 100_000,
        };
        assert!(close(price.cost(usage), 2.0));
        assert!(close(tokens(u64::from(u32::MAX) + 1), 4_294_967_296.0));
    }
}
