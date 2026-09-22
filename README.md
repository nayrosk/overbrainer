# overbrainer

Distill knowledge from a large "parent" LLM into a smaller open-weights "child" model.

overbrainer generates questions on your topics with an LLM, collects answers (and reasoning) from a parent model, then fine-tunes a child model with Axolotl locally, over SSH or on Runpod, while showing live progress in a terminal UI.

Status: early development. Available today: project setup and configuration checks.

## Install

```bash
cargo install overbrainer
```

Supported platforms: Linux and macOS.

## Quick start

```bash
overbrainer init my-project
cd my-project
cp .env.example .env   # then fill in URLs and keys
overbrainer config check
overbrainer config check --resolve   # also tests Vault access
```

## Configuration

`overbrainer.toml` describes the project: topics, providers (by protocol), model roles, pipeline and training settings, training targets. It never contains URLs, hosts or secrets, so it is safe to commit.

Everything else comes from environment variables named after the TOML key path, prefixed with `OVERBRAINER_` and using `__` for nesting:

```bash
OVERBRAINER_PROVIDERS__OPENROUTER__BASE_URL=https://openrouter.ai/api/v1
OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY=sk-...
OVERBRAINER_PIPELINE__CONCURRENCY=16
```

Precedence: environment, then `overbrainer.toml`, then defaults. A `.env` file in the project directory is loaded if present.

Provider and target names may only use lowercase letters, digits and `_`, so they map cleanly to environment variable names.

### Secrets and Vault

Any secret value can be a literal or a reference to a Vault or OpenBao KV v2 secret:

```bash
OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY=vault:secret/overbrainer/openrouter#api_key
```

overbrainer reads `VAULT_ADDR` and `VAULT_TOKEN` (or `~/.vault-token`, as written by `vault login`). Secrets are never printed or logged. One known limitation: when a configuration value has the wrong type, the error message quotes that value, so put secrets only in the variables meant for them.

### Logs

Logs go to stderr. Set the filter with `OVERBRAINER_LOG` (for example `debug`); by default overbrainer logs at `info` and its dependencies at `warn`. `NO_COLOR` disables colors.

## Terms of service

Several model providers forbid using their outputs to train models that compete with them. This includes OpenAI and Anthropic, and the restriction still applies when their models are reached through a gateway such as OpenRouter or NanoGPT. Check the terms of the models you configure as parent before training on their outputs.

## License

Licensed under either of MIT or Apache-2.0 at your option.
