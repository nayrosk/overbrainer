# Configuration

A project has two sources of configuration:

- `overbrainer.toml` describes the project: topics, providers (by protocol), model roles, pipeline and training settings, training targets. It never contains URLs, hosts or secrets, so it is safe to commit.
- Environment variables carry everything else. `overbrainer init` writes a commented `.env.example` to start from.

`overbrainer config check` prints the resolved configuration with secrets masked. `overbrainer config check --resolve` also resolves every secret, which tests Vault access. Every command takes `-C DIR` to run on the project in `DIR` instead of the current directory. The [TUI](tui.md)'s Project view edits `overbrainer.toml` field by field and keeps its comments when it saves.

## Environment variables

Every key of `overbrainer.toml` can be set from the environment. The variable is named after the key path, prefixed with `OVERBRAINER_`, with `__` between levels:

```bash
OVERBRAINER_PROVIDERS__OPENROUTER__BASE_URL=https://openrouter.ai/api/v1
OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY=sk-...
OVERBRAINER_PIPELINE__CONCURRENCY=16
```

Precedence: environment, then `overbrainer.toml`, then defaults. A `.env` file in the project directory is loaded if present; `init` adds it to `.gitignore`. A variable exported in the shell wins over the same key in `.env`. A syntax error in `.env` stops the command, naming the line only, never its content.

These keys are accepted from the environment only, and `overbrainer.toml` is rejected if it sets them:

| Variable | Meaning |
|---|---|
| `OVERBRAINER_PROVIDERS__<NAME>__BASE_URL` | Base URL of a provider. |
| `OVERBRAINER_PROVIDERS__<NAME>__API_KEY` | API key of a provider. Without one, the provider is called without authentication, which suits a local server. |
| `OVERBRAINER_TARGETS__<NAME>__HOST` | Host of an `ssh` target: `user@host` or a `~/.ssh/config` alias. |
| `OVERBRAINER_RUNPOD__API_KEY` | Runpod API key, for `runpod` targets. |
| `OVERBRAINER_RUNPOD__BASE_URL` | Runpod REST API, `https://api.runpod.io/v2` by default. Must be `https`, or `http` on a loopback host. |
| `OVERBRAINER_HF_TOKEN` | Hugging Face token, see [Training](training.md#the-hugging-face-token). |
| `OVERBRAINER_LOG` | Log filter, see [Logs](#logs). |

Provider and target names may only use lowercase letters, digits and `_`, so they map cleanly to variable names.

## Reloading while overbrainer runs

`overbrainer.toml` and `.env` are read again when they change, without restarting:

- The [TUI](tui.md) looks at both files every 2 seconds (their modification time and size; on a file system that keeps whole seconds only, an edit that keeps the size within the same second is seen at the next change). A valid configuration is applied as a save from the Project view applies it, and the footer says `✓ config reloaded`. An invalid one is not applied: the previous settings stay, the footer says `✗ overbrainer.toml: <the first problem>` (or `✗ cannot parse .env (syntax error at line N)`), and the Project view marks the fields the problems name. The same broken file is reported once, not every 2 seconds.
- `overbrainer run` reads both files again before each stage after the first, and before training, when they changed since the last read: it prints `config reloaded` on stderr and the next stage uses the new settings. When the changed configuration is invalid, the run stops with its error; fix it and run `overbrainer run` again, which resumes. The prompt templates in `prompts/` are also read at the start of each stage of `run`. Single-stage commands (`subtopics`, `questions`, `answers`, `split`, `train`) read the configuration once.

A stage or a training run already started keeps the settings it started with; the change applies to the next one.

On a reload, the environment is the process one without the keys `.env` set when overbrainer started, plus what `.env` holds now. So a key changed in `.env` takes its new value, a key removed from `.env` is gone, a key added to `.env` is used, and a key exported in the shell keeps winning over `.env`. One exception: a `${VAR}` reference in `.env` expands with the process environment first, so a reference to a key `.env` itself set expands to its value at start, not to its new one; write the value out instead of referencing it.

What reloads is the configuration: every key of `overbrainer.toml` and every `OVERBRAINER_*` variable. A few variables are read once at start and need a restart: `OVERBRAINER_LOG`, `OVERBRAINER_NO_UPDATE_CHECK`, `OVERBRAINER_TUI_COLOR`, `OVERBRAINER_TUI_MOTION`, `NO_COLOR`, `VISUAL` and `EDITOR`, and `VAULT_ADDR` and `VAULT_TOKEN` (for a `vault:` reference).

## Providers

A provider is an API endpoint, declared by name with its protocol: `openai` (an OpenAI-compatible chat completions API) or `anthropic` (the Anthropic Messages API). Its URL and key come from the environment.

OpenRouter:

```toml
[providers.openrouter]
protocol = "openai"
```

```bash
OVERBRAINER_PROVIDERS__OPENROUTER__BASE_URL=https://openrouter.ai/api/v1
OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY=sk-or-...
```

NanoGPT:

```toml
[providers.nanogpt]
protocol = "openai"
```

```bash
OVERBRAINER_PROVIDERS__NANOGPT__BASE_URL=https://nano-gpt.com/api/v1
OVERBRAINER_PROVIDERS__NANOGPT__API_KEY=...
```

Both list model prices in their `/models` listing, so the stages can print what they cost. A command that calls a provider whose `base_url` is unset fails and names the variable to set.

## Roles

Each role names a provider and a model: `generator` writes the subtopics and questions, `parent` answers them, and the optional `embedder` drops near-duplicate questions by embedding similarity.

```toml
[roles]
generator = { provider = "openrouter", model = "qwen/qwen3-235b-a22b" }
parent = { provider = "openrouter", model = "deepseek/deepseek-r1", reasoning = true }
# embedder = { provider = "openrouter", model = "openai/text-embedding-3-small" }
```

A role also takes these request settings:

| Key | Default | Meaning |
|---|---|---|
| `reasoning` | `false` | Ask the model for its reasoning. |
| `max_tokens` | `16384` | Upper bound on generated tokens, reasoning included. |
| `temperature` | provider default | Sampling temperature, 0 to 2. Not allowed with `reasoning = true` on the `anthropic` protocol, which rejects it. |
| `reasoning_effort` | `medium` on `openai` | `low`, `medium` or `high`. Only with `reasoning = true`. |
| `thinking_budget` | none (adaptive thinking) | `anthropic` protocol only. Fixed extended-thinking token budget, `[1024, max_tokens)`. Only with `reasoning = true`, and cannot combine with `reasoning_effort`. Required by Claude Opus 4.5, Sonnet 4.5 and Haiku 4.5; leave unset for newer Claude models. |

The embedder must use the `openai` protocol: Anthropic has no embeddings endpoint. [The dataset pipeline](pipeline.md#reasoning) explains which parents return usable reasoning.

## Topics

```toml
[[topics]]
name = "ownership"
description = "Rust ownership, borrowing and lifetimes"   # optional
subtopics = 10
questions_per_subtopic = 30
```

Topic names must be unique. `subtopics` and `questions_per_subtopic` must be at least 1.

## Pipeline settings

`[pipeline]`:

| Key | Default | Meaning |
|---|---|---|
| `concurrency` | `8` | Parallel requests in `questions` (one subtopic each) and `answers`, 1 to 1024. Lower it for a rate-limited provider. |
| `max_retries` | `5` | Retries per request (rate limits, server errors, timeouts), and batches without progress before a subtopic stops. |
| `dedup_threshold` | `0.8` | Word-overlap similarity above which two questions are duplicates. |
| `embedding_threshold` | `0.9` | Embedding similarity above which two questions are duplicates. |
| `question_batch_size` | `10` | Questions requested per call. |
| `request_timeout_secs` | `600` | Timeout of one request. |
| `eval_ratio` | `0.1` | Share of the usable answers kept for evaluation, spread over the subtopics. |
| `seed` | `42` | Seed of the train/eval split. |
| `include_system_prompt` | `false` | Store the system prompt in the dataset. |

Retries wait with exponential backoff and jitter, or as long as the provider's `Retry-After` asks.

The `[training]` section and the `[targets.*]` tables are described in [Training](training.md) and [Runpod](runpod.md). A Runpod target's `gpu_types` and `data_center_ids` take a TOML array, or `"auto"`; from the environment, a comma-separated value or `auto`, for example `OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES=auto`. `min_vram_gb` and `max_price_per_hour` narrow an `auto` choice of GPU types and are rejected otherwise; see [Runpod](runpod.md) for every field.

## Secrets and Vault

Any secret value can be a literal or a reference to a Vault or OpenBao KV v2 secret, written `vault:<mount>/<path>#<field>`:

```bash
OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY=vault:secret/overbrainer/openrouter#api_key
```

overbrainer reads `VAULT_ADDR` and `VAULT_TOKEN` (or `~/.vault-token`, as written by `vault login`). Secrets are resolved lazily, only by the commands that need them: `split`, for instance, needs no provider and never contacts Vault.

Secrets are never printed or logged. When a configuration value has the wrong type, the error names the key and the expected type, never the value. A TOML syntax error in `overbrainer.toml` names only its location, never the surrounding text.

## Logs

Logs go to stderr, or to the Logs view of `overbrainer tui`. Set the filter with `OVERBRAINER_LOG` (for example `debug`). By default overbrainer logs at `info` and its dependencies at `warn`. `NO_COLOR` disables colors.

## Update check

At the start of a command, overbrainer looks up the latest release on crates.io, with a 3 second timeout, and never holds up the command while waiting for it: the command line waits at most 0.5 s after the command for the answer, then drops the check. The request sends your overbrainer version, in the User-Agent, to crates.io; nothing else. It does not follow redirects. The result is cached for 24 hours in `$XDG_CACHE_HOME/overbrainer/latest-version.json`, or `~/.cache/overbrainer/latest-version.json` when `XDG_CACHE_HOME` is unset. A failed check, or one dropped before its answer, waits an hour before trying again. When a newer release exists, the command line prints a note on stderr once the command ends, and `overbrainer tui` shows it in the footer. The check is skipped for `overbrainer skill`, for shell completions, and whenever stderr (or, for `tui`, stdout) is not a terminal. Any failure (network, parsing, the cache file) is ignored and logged at `debug`.

Set `OVERBRAINER_NO_UPDATE_CHECK` to any non-empty value to turn the check off.
