# overbrainer

Distill knowledge from a large "parent" LLM into a smaller open-weights "child" model.

overbrainer generates questions on your topics with an LLM, collects answers (and reasoning) from a parent model, then fine-tunes a child model with Axolotl locally, over SSH or on Runpod, while showing live progress in a terminal UI.

Status: early development. Available today: project setup, configuration checks, and the data pipeline (subtopics, questions, answers, train/eval split). Training comes next.

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
overbrainer run                      # subtopics, questions, answers, split
```

## Building a dataset

The pipeline runs in four stages. Each one reads the previous stage's file in `data/` and can be run on its own:

| Command | Reads | Writes |
|---|---|---|
| `overbrainer subtopics` | `overbrainer.toml` | `data/subtopics.jsonl` |
| `overbrainer questions` | `data/subtopics.jsonl` | `data/questions.jsonl` |
| `overbrainer answers` | `data/questions.jsonl` | `data/answers.jsonl` |
| `overbrainer split` | `data/answers.jsonl` | `data/train.jsonl`, `data/eval.jsonl` |

`overbrainer run` chains the four. `questions` generates the missing subtopics first.

Every stage resumes: items already on disk are skipped, and each new item is appended as soon as it is ready, so an interruption only loses the requests in flight. A topic left with fewer subtopics than configured, for example after a crash mid-topic, is resumed too: `subtopics` only asks for the missing count. `answers` counts a question as done once it has any answer, including one excluded from training; only `--force` asks the parent again for it. Options:

- `--topic NAME` processes one topic only.
- `--force`, accepted by every stage command except `split`, throws away the stage's output for the selected topics and generates it again. `split` has no `--force`: it always rewrites its files.

Each stage prints one summary line on stdout, with token counts and the cost when the provider lists model prices (OpenRouter and NanoGPT do; otherwise the cost is shown as unknown). Prices come from the provider's `/models` listing, read once per provider per run and capped at 10 seconds; a slow, missing or unpriced listing just leaves the cost unknown. Progress goes to stderr. A stage that could not process some items exits with an error after writing everything else it produced; run it again to retry them. A fatal error, such as a rejected API key, stops the stage early, but it still prints its summary line with the tokens and cost spent so far, keeps the answers already received, and exits non-zero.

How the stages work:

- `subtopics` asks the `generator` role for `subtopics` subtopic names per topic.
- `questions` asks the `generator` for questions in batches of `question_batch_size`, listing the questions already accepted for the subtopic. Near-duplicates are dropped: first by word overlap (`dedup_threshold`), then, when `roles.embedder` is set, by embedding similarity (`embedding_threshold`). A subtopic stops at `questions_per_subtopic`, or after `max_retries` batches that bring nothing new.
- `answers` sends each question to the `parent` role, `concurrency` requests at a time, with the answer system prompt. An answer that hit the token limit, was refused, is empty, or (with `reasoning = true`) has no raw reasoning is kept in `answers.jsonl` with `meta.excluded` set, and left out of training.
- `split` writes the usable answers of every topic to `train.jsonl` and `eval.jsonl`; `--topic` only limits what is counted in the printed report. Only answers whose topic is still configured in `overbrainer.toml` and whose question is still in `questions.jsonl` are used: the others, left by a removed topic or by questions regenerated with `questions --force`, stay in `answers.jsonl` and are reported as orphaned (`split: 90 train, 10 eval, 3 excluded (truncated 3), 2 orphaned`). The eval set holds `eval_ratio` of all usable answers, rounded, at least one as soon as there are two, and never all of them, so train is never empty. It is spread in proportion to size, first over the topics, then over each topic's subtopics, so every topic gets its share even when its subtopics are small; which small subtopics contribute depends on `seed`. The split is deterministic for a given `seed`. When every usable answer is orphaned (for example with `questions.jsonl` missing), `split` also warns that train and eval are empty.

Answer lines use the Axolotl `chat_template` format, with the parent's reasoning in `reasoning_content`:

```json
{"id":"...","topic":"ownership","subtopic":"Borrowing",
 "messages":[{"role":"user","content":"..."},
             {"role":"assistant","content":"...","reasoning_content":"..."}],
 "meta":{"model":"deepseek/deepseek-r1","input_tokens":41,"output_tokens":812,
         "finish_reason":"stop","reasoning_kind":"raw","excluded":null}}
```

Set `include_system_prompt = true` to also store the system prompt as the first message.

### Reasoning

Only raw reasoning is used for training: a usable example keeps `reasoning_content` only when the parent returned the full reasoning trace. A summary or a redacted trace is dropped even on an otherwise usable example; an excluded example keeps whatever reasoning it got, for inspection.

With `reasoning = true` on the parent:

- `openai` protocol: overbrainer sends `reasoning: {"effort": ...}` (`medium` unless `reasoning_effort` is set) and reads the reasoning from `reasoning`, `reasoning_content`, `reasoning_details`, or a `<think>` block in the answer.
- `anthropic` protocol: overbrainer asks for adaptive thinking.

Some parents never return their raw reasoning, whatever their responses claim: any model on the `anthropic` protocol, and Claude, Gemini and OpenAI models on the `openai` protocol (except the open-weight `gpt-oss` models, whose reasoning is raw). overbrainer stores their reasoning as a summary (`reasoning_kind = "summary"`). With `reasoning = true`, every answer from such a parent is therefore excluded as `no_raw_reasoning`, and overbrainer warns about it at startup. With `reasoning = false`, the answers stay usable, without their reasoning.

### Prompt templates

`init` writes the default prompts to `prompts/`. Edit them to change how questions are generated or how the parent is instructed. They are [minijinja](https://docs.rs/minijinja) templates; using an undefined variable is an error. Variables:

| Template | Variables |
|---|---|
| `prompts/subtopics.txt` | `topic`, `description` (may be empty), `count` |
| `prompts/questions.txt` | `topic`, `description`, `subtopic`, `count`, `accepted` (list of questions) |
| `prompts/answer_system.txt` | `topic`, `description` |

A missing file falls back to the built-in default.

## Configuration

`overbrainer.toml` describes the project: topics, providers (by protocol), model roles, pipeline and training settings, training targets. It never contains URLs, hosts or secrets, so it is safe to commit.

Everything else comes from environment variables named after the TOML key path, prefixed with `OVERBRAINER_` and using `__` for nesting:

```bash
OVERBRAINER_PROVIDERS__OPENROUTER__BASE_URL=https://openrouter.ai/api/v1
OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY=sk-...
OVERBRAINER_PIPELINE__CONCURRENCY=16
```

Precedence: environment, then `overbrainer.toml`, then defaults. A `.env` file in the project directory is loaded if present.

Provider and target names may only use lowercase letters, digits and `_`, so they map cleanly to environment variable names. A provider without `api_key` is called without authentication, which suits a local server.

### Roles

Each role (`generator`, `parent`, optional `embedder`) names a provider and a model, plus optional request settings:

| Key | Default | Meaning |
|---|---|---|
| `reasoning` | `false` | Ask the model for its reasoning. |
| `max_tokens` | `16384` | Upper bound on generated tokens, reasoning included. |
| `temperature` | provider default | Sampling temperature, 0 to 2. Not allowed with `reasoning = true` on the `anthropic` protocol, which rejects it. |
| `reasoning_effort` | `medium` on `openai` | `low`, `medium` or `high`. Only with `reasoning = true`. |

The embedder must use the `openai` protocol: Anthropic has no embeddings endpoint.

### Pipeline settings

| Key | Default | Meaning |
|---|---|---|
| `concurrency` | `8` | Parallel requests in `answers`, 1 to 1024. |
| `max_retries` | `5` | Retries per request (rate limits, server errors, timeouts), and batches without progress before a subtopic stops. |
| `dedup_threshold` | `0.8` | Word-overlap similarity above which two questions are duplicates. |
| `embedding_threshold` | `0.9` | Embedding similarity above which two questions are duplicates. |
| `question_batch_size` | `10` | Questions requested per call. |
| `request_timeout_secs` | `600` | Timeout of one request. |
| `eval_ratio` | `0.1` | Share of the usable answers kept for evaluation, spread over the subtopics. |
| `seed` | `42` | Seed of the train/eval split. |
| `include_system_prompt` | `false` | Store the system prompt in the dataset. |

Retries wait with exponential backoff and jitter, or as long as the provider's `Retry-After` asks.

### Secrets and Vault

Any secret value can be a literal or a reference to a Vault or OpenBao KV v2 secret:

```bash
OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY=vault:secret/overbrainer/openrouter#api_key
```

overbrainer reads `VAULT_ADDR` and `VAULT_TOKEN` (or `~/.vault-token`, as written by `vault login`). Secrets are resolved lazily, only by the commands that need them: `split`, for instance, needs no provider and never contacts Vault. Secrets are never printed or logged. When a configuration value has the wrong type, the error names the key and the expected type, never the value; a TOML syntax error in `overbrainer.toml` names only its location, never the surrounding text.

### Logs

Logs go to stderr. Set the filter with `OVERBRAINER_LOG` (for example `debug`); by default overbrainer logs at `info` and its dependencies at `warn`. `NO_COLOR` disables colors.

## Releasing

Releases are cut by hand from `main`:

1. Merging to `main` runs release-plz, which opens or updates a release pull request with the version bump and the `CHANGELOG.md` entry.
2. Review and merge that pull request.
3. Tag the merge commit with a signed tag and push it: `git tag -s v0.2.0 -m v0.2.0 && git push origin v0.2.0`.
4. The tag runs the checks, publishes the crate to crates.io and creates the GitHub release from the changelog entry.

## Terms of service

Several model providers forbid using their outputs to train models that compete with them. This includes OpenAI and Anthropic, and the restriction still applies when their models are reached through a gateway such as OpenRouter or NanoGPT. Check the terms of the models you configure as parent before training on their outputs.

## License

Licensed under either of MIT or Apache-2.0 at your option.
