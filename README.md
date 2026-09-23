# overbrainer

Distill knowledge from a large "parent" LLM into a smaller open-weights "child" model.

overbrainer generates questions on your topics with an LLM, collects answers (and reasoning) from a parent model, then fine-tunes a child model with Axolotl locally, over SSH or on Runpod, while showing live progress in a terminal UI.

Status: early development. Available today: project setup, configuration checks, the data pipeline (subtopics, questions, answers, train/eval split), and fine-tuning with Axolotl on this machine, over SSH or on a Runpod pod. The terminal UI comes next.

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
overbrainer run                      # subtopics, questions, answers, split, then train
```

`overbrainer run` trains only when `overbrainer.toml` has a `[training]` section; otherwise it stops after `split`.

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
- `anthropic` protocol: overbrainer asks for adaptive thinking (`thinking: {"type": "adaptive"}`), unless the role sets `thinking_budget`.

Claude Sonnet 5, Opus 5, Opus 4.8, Opus 4.7 and Fable 5.x require adaptive thinking and reject a fixed budget, so leave `thinking_budget` unset for them. Claude Opus 4.5, Sonnet 4.5 and Haiku 4.5 do the opposite: they reject adaptive thinking (`400 adaptive thinking is not supported on this model`) and need a fixed budget instead. Set `roles.<role>.thinking_budget` to a value in `[1024, max_tokens)` to switch that role to `thinking: {"type": "enabled", "budget_tokens": ...}`, which also drops `output_config.effort` from the request (`thinking_budget` cannot combine with `reasoning_effort`).

Some parents never return their raw reasoning, whatever their responses claim: any model on the `anthropic` protocol, and Claude, Gemini and OpenAI models on the `openai` protocol (except the open-weight `gpt-oss` models, whose reasoning is raw). overbrainer stores their reasoning as a summary (`reasoning_kind = "summary"`). With `reasoning = true`, every answer from such a parent is therefore excluded as `no_raw_reasoning`, and overbrainer warns about it at startup. With `reasoning = false`, the answers stay usable, without their reasoning.

### Prompt templates

`init` writes the default prompts to `prompts/`. Edit them to change how questions are generated or how the parent is instructed. They are [minijinja](https://docs.rs/minijinja) templates; using an undefined variable is an error. Variables:

| Template | Variables |
|---|---|
| `prompts/subtopics.txt` | `topic`, `description` (may be empty), `count` |
| `prompts/questions.txt` | `topic`, `description`, `subtopic`, `count`, `accepted` (list of questions) |
| `prompts/answer_system.txt` | `topic`, `description` |

A missing file falls back to the built-in default.

## Training

`overbrainer train` fine-tunes `training.base_model` with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) 0.19 on `data/train.jsonl`, evaluating on `data/eval.jsonl`, on the target named by `training.target` (or `--target NAME`).

| Command | What it does |
|---|---|
| `overbrainer train [--target NAME]` | Start a run and follow it until the job ends |
| `overbrainer train attach RUN_ID` | Follow a run again, then retrieve its results |
| `overbrainer train cancel RUN_ID` | Stop the job of a run and retrieve its artifacts |
| `overbrainer runs ls` | List the runs: ID, state, target, creation time, and the pod of a Runpod run |
| `overbrainer pod ls` | List the Runpod pods overbrainer created, with their run |
| `overbrainer pod rm RUN_ID [--force]` | Delete every pod of a run and wait until Runpod no longer shows them |

A run gets an ID such as `20260922-143005-a1b2` and a directory `runs/<run-id>/` holding `axolotl.yaml`, copies of the train and eval files, the metrics plugin, `run.json` (target, job, state), and, once the job has ended, `metrics.jsonl`, `job.log` and `output/`. `output/` holds the LoRA adapter (or the full model with `adapter = "full"`) and, with `merge = true`, the merged model in `output/merged/`. Intermediate `checkpoint-*` directories stay on the target.

The job runs detached from overbrainer: once it has started, Ctrl-C, a closed terminal or a lost SSH connection stop overbrainer from following it, not the training. Starting a run is never interrupted: a Ctrl-C pressed while a run is starting is only acted on once the job has actually started, so the command always finishes starting before it detaches. After Ctrl-C, overbrainer prints the `overbrainer train attach` command that follows the run again and exits with an error status; after six failed attempts in a row to reach the target, it does the same. `overbrainer runs ls` shows the state last recorded in `run.json`; `train attach` refreshes it. Progress (step, epoch, loss, learning rate, every evaluation) goes to stderr; the final summary goes to stdout:

```
train: run 20260922-143005-a1b2 succeeded; step 1200/1200, epoch 3.00, loss 0.4123, eval_loss 0.5012; output in runs/20260922-143005-a1b2/output
```

A run fails when the job exits with a non-zero code or when it writes no metric line at all, which means Axolotl did not load the metrics plugin; `runs/<run-id>/job.log` holds the job's output.

`overbrainer train cancel` needs a `[training]` section: cancelling a run also retrieves its artifacts, the same way a successful or failed run does. It accepts any run that started a job, whatever its recorded state: a run already recorded as ended prints `run RUN_ID already ended: STATE; stopping any job left on the target` and the cancel runs anyway, which stops a container that outlived its job. A job that had already ended is not signalled: cancel then reports what it found and tells you to run `overbrainer train attach RUN_ID` to collect it. A run already recorded as cancelled fails with `run RUN_ID already ended: cancelled`, and a run that never started a job with `run RUN_ID has not started`; both exit non-zero without touching anything.

### Targets

```toml
[targets.local]
kind = "local"                 # this machine, in runs/<run-id>/
runtime = "native"             # native | docker
venv = "~/.venvs/axolotl"      # native: directory holding bin/axolotl; otherwise axolotl must be on PATH

[targets.homelab]
kind = "ssh"                   # host from OVERBRAINER_TARGETS__HOMELAB__HOST (user@host or a ~/.ssh/config alias)
runtime = "docker"
engine = "podman"              # docker (default) | podman
# image = "..."                # default: axolotlai/axolotl:0.19.0-py3.12-cu130-2.12.1, pinned by digest
# workdir = "overbrainer"      # on the remote machine, relative to its home directory
```

- `docker` runtime: one container per run, named `overbrainer-<run-id>`, with all GPUs, the host IPC namespace and the run directory mounted at `/workspace/run`. The Hugging Face cache is `runs/.hf-cache` (local) or `<workdir>/.hf-cache` (SSH), shared by the runs of the target, so a base model is downloaded once. Docker needs the NVIDIA Container Toolkit (`--gpus all`); Podman needs its CDI specification (`--device nvidia.com/gpu=all`, generated with `nvidia-ctk cdi generate`). The default image is built for CUDA 13 and needs an NVIDIA driver from the 580 series or newer. With rootful Docker, the files the container writes are owned by root. Over SSH, if the remote machine kills user processes at logout (`KillUserProcesses=yes` without lingering, see the `native` bullet), the wrapper dies with the session while the container keeps running and holding the GPU; overbrainer then reports the run as failed. `overbrainer train cancel <run-id>` stops that container, whatever the run's recorded state, and `docker stop overbrainer-<run-id>` (or `podman stop overbrainer-<run-id>`) is the manual fallback.
- `native` runtime: runs `<venv>/bin/axolotl` on the host. Over SSH, if systemd-logind kills user processes at logout (`KillUserProcesses=yes`), enable lingering for the SSH user (`loginctl enable-linger`) so the job survives the disconnection. The job's environment differs by target: on a local target it is overbrainer's own, without the `OVERBRAINER_*` and `VAULT_*` variables; over SSH it is whatever a non-interactive `sh -c` gets there, with no login shell and no `~/.bashrc`. A setup that a profile or a conda activation provides is not there over SSH: put it in the venv, or in the SSH server's environment.
- SSH uses your `ssh` binary with `~/.ssh/config`, the agent and `known_hosts`; a host that is not already in `known_hosts` is refused. Files travel as `tar` streams over the connection, so the remote machine needs `tar`, `setsid` and `nohup` (any Linux distribution has them). overbrainer keeps one master connection open (OpenSSH `ControlMaster`); an `ssh` wrapper that kills background processes, such as a firejail profile, breaks it.

### Runpod

```toml
[targets.gpu_cloud]
kind = "runpod"
gpu_types = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA A40"]  # tried in order
max_hours = 6
```

A `runpod` target creates a pod for each run on Runpod's Secure Cloud, runs the job on it over SSH with the image's Axolotl, retrieves the results, then deletes the pod. It needs `OVERBRAINER_RUNPOD__API_KEY` (a literal or a `vault:` reference), resolved only when a Runpod command runs, and `ssh-keygen` next to `ssh` on this machine.

| Key | Default | Meaning |
|---|---|---|
| `gpu_types` | required | Runpod GPU type IDs, tried in order until one can be placed. From the environment, one comma-separated value: `OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES="NVIDIA GeForce RTX 4090,NVIDIA A40"`. |
| `max_hours` | required | The pod's watchdog deletes the pod this long after it was created, whatever it is doing, at most 720. |
| `gpu_count` | `1` | GPUs per pod. |
| `image` | `axolotlai/axolotl-cloud-term:0.19.0-py3.12-cu130-2.12.1`, pinned by digest | Pod image (CUDA 13, driver 580 or newer). |
| `venv` | `/workspace/axolotl-venv` | Virtual environment holding `bin/axolotl` on the pod. |
| `container_disk_gb` | `50` | Container disk, at least 20. |
| `boot_grace_minutes` | `30` | The watchdog deletes a pod whose job never started after this. |
| `retrieve_grace_minutes` | `60` | The watchdog deletes a pod whose ended job was not retrieved after this. |
| `data_center_ids` | any | Data centers the pod may be placed in, for example `["EU-RO-1"]`. |
| `network_volume_id` | none | Network volume mounted at `/workspace/data`; runs and the Hugging Face cache then live on it. Needs exactly one `data_center_ids` entry, the volume's data center. overbrainer never deletes anything on it. |

`gpu_type` became `gpu_types`: a configuration with the old key is rejected as `unknown field`.

What happens to a pod:

- Creation: before every create call, `runs/<run-id>/pod.json` records it, so a pod created by a client that dies right after is still found by `overbrainer pod ls` and `pod rm` (every pod carries its run ID in its environment). A GPU type Runpod reports as a capacity failure, or refuses with 403, makes overbrainer try the next one; 402 (no credits), 422, and any other 400 (overbrainer's own request is wrong) stop at once instead of repeating for every type. A create that gets no clear answer is not sent again blindly: overbrainer first looks for the pod by its run ID.
- SSH: each run gets its own client key and its own pod host key in `runs/<run-id>/ssh/`. The pod's host key is generated here, sent in the create call's environment (which anyone holding the account API key can read) and pinned in `runs/<run-id>/ssh/known_hosts`; the image's own host keys are never trusted. overbrainer connects with `ssh -F runs/<run-id>/ssh/config`, which ignores `~/.ssh/config` and the agent. Neither `ssh/` nor `pod.json` is ever uploaded with the run directory. The pod is ready once SSH answers with that key, usually a few minutes after creation (the image is 8.5 GB); after 15 minutes it is deleted and the next GPU type tried.
- The watchdog: the pod's command starts a small shell watchdog as its first process. At startup it checks that the pod's own Runpod key can read the pod, and overbrainer refuses to train (and deletes the pod) when it cannot. It then deletes the pod at `max_hours`, `boot_grace_minutes` after the pod started if no job ever did, `retrieve_grace_minutes` after the job ended if its results were not retrieved, and at once once overbrainer marks them retrieved. A bootstrap failure (for example sshd could not start) is deleted at once too, once the watchdog's own proof runs. Its log is `.pod/watchdog.log` in the run directory on the pod and the pod's Runpod logs.
- The end: the results are downloaded and every file checked against the SHA-256 the pod computed for it (a successful run must also have something in `output/`); then the pod is deleted and overbrainer waits until Runpod no longer shows it. When the download fails or does not check out, the pod stays until the retrieve grace ends: `overbrainer train attach RUN_ID` retries.
- Ctrl-C: before the job exists, the pod is deleted and the run fails as interrupted. Once the job runs, Ctrl-C only stops following it, as on any target; the pod keeps running and the watchdog bounds its cost.
- `--keep-pod` keeps the pod with no time limit once its job exists: from then on neither overbrainer nor the watchdog deletes it for any reason, not even `max_hours`. Until then it is guarded like any other pod: a pod whose bootstrap failed is deleted at once, and one whose job never started is deleted after `boot_grace_minutes`. Once the job starts, `train` warns with its hourly rate; once the run ends, or is left running after Ctrl-C, it prints the `ssh -F ...` command that reaches it. Only `overbrainer pod rm RUN_ID` removes it.

Every `train` on a Runpod target first lists the account's pods and warns about overbrainer pods that nothing will delete (their run ended, is a stray left by an ambiguous create, is not in `runs/`, or has no run marker); it never deletes them itself. A pod named like overbrainer's but without a usable run marker is not something `pod rm` can take: delete it from the Runpod console. `overbrainer pod rm RUN_ID` takes only a run ID, never a pod ID, and refuses to delete anything for a run absent from this project's `runs/` (another checkout may own it) or a run still starting its pod, unless `--force`. For a run in progress with a recorded pod, without `--force` it deletes every other pod of the run (strays, and any extra pod left by an ambiguous create), keeps the training pod, and still fails, naming what it kept and deleted; `--force` also deletes the training pod and marks the run failed. A run in progress whose pod is not recorded needs `--force` too, since any pod of the run could be the one training. `runs ls` shows each Runpod run's pod as `pod.json` last recorded it, with its rate or its estimated spend (rate times lifetime; Runpod bills per second, including the image pull).

A custom `image` must keep an entrypoint that ends with `exec "$@"`, and provide `bash`, `sshd` (started with `service ssh`), `ssh-keygen`, `base64`, `curl`, `setsid`, `nohup`, `tar`, `find`, `sha256sum` and Axolotl in `venv`. Jobs on the pod start from `/etc/overbrainer/job.env`, which the pod writes with the image's `PATH`, its CUDA library path and `HF_HOME`, since an SSH session does not see the image's environment.

### The Hugging Face token

`OVERBRAINER_HF_TOKEN` (a literal or a `vault:` reference) is resolved only when a run starts and reaches the job only as its `HF_TOKEN` environment variable: it is never written to `axolotl.yaml`, `run.json` or any file overbrainer writes, never put on a command line, and sent over SSH on the command's standard input. With the `docker` runtime it does reach one file overbrainer does not write: the container engine stores the environment it was started with in the container's own configuration, where `docker inspect` (or `podman inspect`) shows it until the container is removed. It is needed for gated or private base models and for `hub_model_id`, which pushes the adapter (not the merged model) to a private Hub repository.

### Reasoning in the chat template

The parent's reasoning is trained only if the chat template renders `reasoning_content`. Axolotl's `qwen3`, `qwen3_5`, `exaone4`, `gemma4` and `gemma4_unified` templates do, as does the official template of Qwen3 models; other templates may drop it without any error. overbrainer warns at the start of a run when the template in use (the base model's own, or `axolotl_extra.chat_template`) is not one of them. The check goes by name only: a local path or a renamed model may get the warning wrongly.

### Training settings

| Key | Default | Meaning |
|---|---|---|
| `target` | required | Target to train on. |
| `base_model` | required | Hugging Face repo ID, or a path on the target. |
| `adapter` | required | `lora`, `qlora` (4-bit base model) or `full`. |
| `epochs` | `3` | Training epochs. |
| `learning_rate` | `2e-4` | Peak learning rate. |
| `lora_r` | `16` | LoRA rank. |
| `lora_alpha` | `32` | LoRA scaling factor. |
| `lora_dropout` | `0.05` | LoRA dropout, in [0, 1). |
| `sequence_len` | `4096` | Maximum tokens per example. |
| `micro_batch_size` | `2` | Examples per GPU per step. |
| `gradient_accumulation_steps` | `4` | Steps summed before each optimizer update. |
| `optimizer` | `adamw_torch_fused` | Axolotl optimizer name. |
| `lr_scheduler` | `cosine` | Axolotl scheduler name. |
| `sample_packing` | `true` | Pack short examples into one sequence. |
| `evals_per_epoch` | `4` | Evaluations on `data/eval.jsonl` per epoch. |
| `saves_per_epoch` | `1` | Checkpoints per epoch. |
| `merge` | `false` | Also write the merged model (`lora` and `qlora` only). |
| `hub_model_id` | none | Push the adapter to this private Hub repository. |

overbrainer also sets `attn_implementation: sdpa` (no extra package needed), `gradient_checkpointing: true`, `warmup_ratio: 0.1` and `logging_steps: 1`. `[training.axolotl_extra]` is merged into the generated YAML last: tables merge key by key, any other value replaces the generated one, so it can change these too (for example `attn_implementation = "flash_attention_2"` on an image with flash-attn installed, or `chat_template = "qwen3"`). An `eval_steps` or `save_steps` there replaces `evals_per_epoch` or `saves_per_epoch`. It cannot set a key that has a typed setting above (use the setting) nor the keys overbrainer manages: `datasets`, `test_datasets`, `val_set_size`, `output_dir`, `dataset_prepared_path`, `plugins`.

Values set through the environment (`OVERBRAINER_TRAINING__AXOLOTL_EXTRA__WARMUP_STEPS=10`) arrive as text; overbrainer turns `true`, `false`, integers and decimal numbers into booleans and numbers, and leaves anything else as text. A number-like value that must stay text belongs in `overbrainer.toml`.

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
| `thinking_budget` | none (adaptive thinking) | `anthropic` protocol only. Fixed extended-thinking token budget, `[1024, max_tokens)`. Only with `reasoning = true`, and cannot combine with `reasoning_effort`. Required by Claude Opus 4.5, Sonnet 4.5 and Haiku 4.5; leave unset for newer Claude models. |

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
