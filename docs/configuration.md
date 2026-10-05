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
| `OVERBRAINER_SSH_CLIENT` | `openssh` or `builtin`: the SSH client of every `ssh` and `runpod` target, over their `ssh_client` field. See [Built-in SSH client](training.md#built-in-ssh-client). |
| `OVERBRAINER_RUNPOD__API_KEY` | Runpod API key, for `runpod` targets. |
| `OVERBRAINER_RUNPOD__BASE_URL` | Runpod REST API, `https://api.runpod.io/v2` by default. Must be `https`, or `http` on a loopback host. |
| `OVERBRAINER_HF_TOKEN` | Hugging Face token, see [Training](training.md#the-hugging-face-token). |
| `OVERBRAINER_LOG` | Log filter, see [Logs](#logs). |

Provider and target names may only use lowercase letters, digits and `_`, so they map cleanly to variable names.

Any other `OVERBRAINER_*` variable must name a key, or the command stops with `unknown field`. The exceptions are the variables read outside the configuration, which are never taken as keys: `OVERBRAINER_NO_UPDATE_CHECK` (see [Update check](#update-check)), `OVERBRAINER_TUI_COLOR` and `OVERBRAINER_TUI_MOTION` (see [Color and motion](tui.md#color-and-motion)), and the `OVERBRAINER_TEST_*` variables of the test suite.

## Reloading while overbrainer runs

`overbrainer.toml` and `.env` are read again when they change, without restarting:

- The [TUI](tui.md) looks at both files every 2 seconds (their modification time and size; on a file system that keeps whole seconds only, an edit that keeps the size within the same second is seen at the next change). A valid configuration is applied as a save from the Project view applies it, and the footer says `✓ config reloaded`. An invalid one is not applied: the previous settings stay, the footer says `✗ overbrainer.toml: <the first problem>` (or `✗ cannot parse .env (syntax error at line N)`), and the Project view marks the fields the problems name. The same broken file is reported once, not every 2 seconds.
- `overbrainer run` reads both files again before each stage after the first, and before training, when they changed since the last read: it prints `config reloaded` on stderr and the next stage uses the new settings. When the changed configuration is invalid, the run stops with its error; fix it and run `overbrainer run` again, which resumes. The prompt templates in `prompts/` are also read at the start of each stage of `run`. Single-stage commands (`subtopics`, `questions`, `answers`, `split`, `train`) read the configuration once.

A stage or a training run already started keeps the settings it started with; the change applies to the next one.

On a reload, the environment is the process one without the keys `.env` set when overbrainer started, plus what `.env` holds now. So a key changed in `.env` takes its new value, a key removed from `.env` is gone, a key added to `.env` is used, and a key exported in the shell keeps winning over `.env`. One exception: a `${VAR}` reference in `.env` expands with the process environment first, so a reference to a key `.env` itself set expands to its value at start, not to its new one; write the value out instead of referencing it.

What reloads is the configuration: every key of `overbrainer.toml` and every `OVERBRAINER_*` variable that sets one. A few variables are read once at start and need a restart: `OVERBRAINER_LOG`, `OVERBRAINER_NO_UPDATE_CHECK`, `OVERBRAINER_TUI_COLOR`, `OVERBRAINER_TUI_MOTION`, `NO_COLOR`, `VISUAL` and `EDITOR`, and `VAULT_ADDR` and `VAULT_TOKEN` (for a `vault:` reference).

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

`ssh_client` (`openssh` by default, or `builtin`) on an `ssh` or `runpod` target picks the SSH client, and `OVERBRAINER_SSH_CLIENT` overrides it for every target; see [Built-in SSH client](training.md#built-in-ssh-client). The `[training]` section and the `[targets.*]` tables are described in [Training](training.md) and [Runpod](runpod.md). A Runpod target's `gpu_types` and `data_center_ids` take a TOML array, or `"auto"`; from the environment, a comma-separated value or `auto`, for example `OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES=auto`. `min_vram_gb` and `max_price_per_hour` narrow an `auto` choice of GPU types and are rejected otherwise, and `max_volume_gb` needs `network_volume_id`; see [Runpod](runpod.md) for every field.

The optional `[export]` section (`after_training`, `quantize`, `ollama_name`) sets the export of a run's model to GGUF and its Ollama Modelfile, described in [Training](training.md#export-to-gguf-and-ollama), for example `OVERBRAINER_EXPORT__QUANTIZE=Q8_0` from the environment.

The optional `[hub]` section (`repo`, `private`, `after_training`) sets the push of a run to Hugging Face, described in [Training](training.md#push-to-hugging-face), for example `OVERBRAINER_HUB__PRIVATE=false` from the environment. `training.hub_model_id` is deprecated in its favor: `train`, a training start in the TUI and `config check` warn about it, and `overbrainer migrate` moves it to `[hub] repo`.

## Metrics

`[metrics]` turns on a Prometheus endpoint. It is off unless `listen` is set:

```toml
[metrics]
listen = "127.0.0.1:9464"
```

or `OVERBRAINER_METRICS__LISTEN=127.0.0.1:9464`. `listen` is an IP address and a port (`[::1]:9464` for IPv6); a host name is refused. Port `0` takes a free port, which the log line `metrics at http://ADDRESS/metrics` names.

The endpoint serves only while a command holds the project: the stage commands, `run`, `train` and its subcommands, `pod rm` and `tui`, for as long as they run. The read-only commands (`history`, `runs ls`, `runs logs`, `pod ls`, `config check`) and `migrate` serve nothing. `GET /metrics` answers in the OpenMetrics text format (`application/openmetrics-text; version=1.0.0`); any other path or method gets a 404. The endpoint has no authentication, so keep it on a loopback address: any other address logs a warning. When the address cannot be bound (a port already in use), the command logs a warning and goes on without metrics.

The counters are cumulative per project: they start from `.overbrainer/history.jsonl` (see [Project state](pipeline.md#project-state)), then follow the running command. `overbrainer_item_retries_total` is not in the history, so it restarts at zero with each command. The training gauges show the runs followed by the current command; the Runpod spend is read from every run's pod record at each scrape. The `target` and `gpu` gauges describe the machine of a run while the command follows it: sampled every 10 seconds (see [Training](training.md)), they are removed when the run's bus closes (the command stops following the run) or when that bus follows another run. A figure the target does not report (no GPU, no `/proc`) has no series. A network file system, such as a Runpod network volume, has no disk series: `df` reports the whole shared cluster there, not the volume, whose usage is measured separately.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `overbrainer_stage_items_total` | counter | `stage`, `result` | Items by result: `done`, `skipped`, `failed`, `excluded`. |
| `overbrainer_tokens_total` | counter | `stage`, `model`, `direction` | Tokens sent (`in`) and received (`out`), retries included. |
| `overbrainer_cost_usd_total` | counter | `stage`, `model` | Cost in USD; an item whose price is unknown adds nothing. |
| `overbrainer_stage_running` | gauge | `stage` | Stages running now. |
| `overbrainer_item_retries_total` | counter | `stage` | Failed attempts retried, since the command started. |
| `overbrainer_train_step` | gauge | `run_id` | Last optimizer step of a run. |
| `overbrainer_train_max_steps` | gauge | `run_id` | Total optimizer steps of a run, once known. |
| `overbrainer_train_phase` | gauge | `run_id`, `phase` | 1 for what the job of a followed run does now, 0 for the other phases: `training`, `evaluating`, `finalizing`, `merging`, `exporting`, `retrieving`. Removed once nothing follows the run. |
| `overbrainer_train_loss` | gauge | `run_id` | Last training loss. |
| `overbrainer_eval_loss` | gauge | `run_id` | Last evaluation loss. |
| `overbrainer_learning_rate` | gauge | `run_id` | Last learning rate. |
| `overbrainer_runpod_spend_usd` | gauge | `run_id` | Estimated Runpod spend: the pod's rate times its uptime, final once deleted. |
| `overbrainer_target_disk_used_bytes` | gauge | `run_id`, `mount` | Bytes used on the file system of the run directory, and on `/` when it is another; a network file system is left out. |
| `overbrainer_target_disk_size_bytes` | gauge | `run_id`, `mount` | Size of that file system. |
| `overbrainer_target_cpu_usage_ratio` | gauge | `run_id` | Share of the CPUs busy since the previous sample, 0 to 1. |
| `overbrainer_target_cpu_load1` | gauge | `run_id` | Load average over one minute: the host's, even inside a container. |
| `overbrainer_target_cpus` | gauge | `run_id` | CPUs: the container's quota when it has one, else `nproc`. |
| `overbrainer_target_memory_used_bytes` | gauge | `run_id` | Memory used, page cache the kernel can drop left out. |
| `overbrainer_target_memory_limit_bytes` | gauge | `run_id` | Memory: the container's limit when it has one, else the machine's. |
| `overbrainer_target_sample_timestamp_seconds` | gauge | `run_id` | When the target was last sampled, in Unix seconds. |
| `overbrainer_gpu_utilization_ratio` | gauge | `run_id`, `gpu` | Share of time the GPU ran a kernel, 0 to 1. |
| `overbrainer_gpu_memory_used_bytes` | gauge | `run_id`, `gpu` | GPU memory used. |
| `overbrainer_gpu_memory_total_bytes` | gauge | `run_id`, `gpu` | GPU memory in all. |
| `overbrainer_gpu_temperature_celsius` | gauge | `run_id`, `gpu` | GPU temperature. |
| `overbrainer_gpu_power_watts` | gauge | `run_id`, `gpu` | Power the GPU draws. |
| `overbrainer_gpu_power_limit_watts` | gauge | `run_id`, `gpu` | The GPU's power limit. |
| `overbrainer_gpu_info` | gauge | `run_id`, `gpu`, `name` | Always 1; `name` is the GPU model. |
| `overbrainer_build_info` | gauge | `version` | Always 1. |

Labels only ever hold stage names, model names, run IDs, the version, mount points, GPU indices and GPU model names: never a prompt, an answer or a secret.

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
