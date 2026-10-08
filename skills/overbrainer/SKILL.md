---
name: overbrainer
description: Drive the overbrainer CLI to distill a large LLM into a smaller open-weights model. Use when the user wants to generate a question and answer dataset from a parent model, fine-tune a child model with Axolotl locally, over SSH or on Runpod, inspect training runs or clean up Runpod pods, or when the project has an overbrainer.toml.
license: MIT OR Apache-2.0
compatibility: Needs the overbrainer binary on PATH, of the same version as this skill (overbrainer --version).
---

# overbrainer

The overbrainer CLI builds a training set from a "parent" LLM and fine-tunes a smaller "child" model on it. Everything happens in a project directory holding `overbrainer.toml`. Every command takes `-C DIR` to work on a project elsewhere.

## Rules

Follow these on every task. They protect the user's secrets and money.

- Never read, print, grep or copy `.env`, `.vault-token` or any secret. To see the configuration, run `overbrainer config check`, which masks secrets. Run `overbrainer config check --resolve` only when the user asks to test Vault access.
- Ask before any command that calls a paid API: `overbrainer subtopics`, `overbrainer questions`, `overbrainer answers`, `overbrainer run`, and `overbrainer train` on a Runpod target, which is billed per second. Say roughly how much work it is first: `answers` sends one request per question, and a topic yields up to `subtopics` times `questions_per_subtopic` questions.
- Never pass `--force` without the user's explicit consent. It throws away a stage's output and pays for it again.
- After any Runpod run, run `overbrainer pod ls` and report any pod left running. Use `overbrainer pod rm RUN_ID --force` or `overbrainer train --keep-pod` only when the user asks.
- Do not start `overbrainer tui`: it is interactive and needs a terminal you do not have. Suggest it to the user for browsing the dataset, or, in a directory without `overbrainer.toml`, for its init wizard, which asks for everything a project needs and writes `.env` itself.
- Before training on a parent's outputs, remind the user that several providers (OpenAI and Anthropic among them) forbid using their outputs to train competing models, even through OpenRouter or NanoGPT.
- Before `overbrainer answers` or `overbrainer run`, read `roles.parent` in `overbrainer.toml`, then read the protocol from `[providers.NAME]` where NAME is `roles.parent.provider`. If `roles.parent` has `reasoning = true` and that protocol is `anthropic`, or the parent is a Claude, Gemini or OpenAI model other than `gpt-oss` on the `openai` protocol, stop and ask the user: such a parent never returns raw reasoning, so every answer would be paid for and then excluded from training.
- Stages and training can run for hours. Run them in the background with the output in a log file, as in "Follow long commands" below.

## Set up a project

```bash
overbrainer init my-project && cd my-project
```

`init` writes `overbrainer.toml`, `.env.example`, the prompt templates in `prompts/` and a `.gitignore`. It refuses to overwrite existing files. It never asks anything: use it, not `overbrainer tui`, whose wizard is interactive.

The user copies `.env.example` to `.env` and fills in the provider URLs and keys. Do not do it for them, and do not read the result. `overbrainer.toml` never holds URLs, hosts or secrets, so you may read and edit it:

- `[[topics]]`: `name` (unique), `description` (optional), `subtopics`, `questions_per_subtopic`.
- `[providers.NAME]`: `protocol = "openai"` or `"anthropic"`. The URL and key come from `OVERBRAINER_PROVIDERS__NAME__BASE_URL` and `OVERBRAINER_PROVIDERS__NAME__API_KEY`, with NAME in uppercase: `OVERBRAINER_PROVIDERS__OPENROUTER__BASE_URL` for `[providers.openrouter]`.
- `[roles]`: `generator` writes subtopics and questions, `parent` answers them, optional `embedder` drops near-duplicate questions. Each is `{ provider = "...", model = "..." }`, and `reasoning = true` asks the parent for its reasoning.
- `[pipeline]`: `concurrency`, `max_retries`, `dedup_threshold`, `eval_ratio`, `seed` and more.
- `[training]` and `[targets.NAME]`: see "Train" below.
- `[metrics]`: optional `listen = "127.0.0.1:9464"` serves Prometheus metrics at `/metrics` while a command that writes to the project runs.

Any key can also be set as `OVERBRAINER_<PATH>` with `__` between levels. Then check it:

```bash
overbrainer config check
```

It fails with the key and the reason when a value is invalid. A missing URL, key or host does not make it fail: look for `unset` in its output, such as `providers.openrouter.base_url = (unset)`. The commands that need the value fail and name the variable to set.

Every command also checks crates.io once a day for a newer overbrainer release and prints a note on stderr, but only when stderr is a terminal, so it never runs when you drive overbrainer as a subprocess.

Details: [configuration](https://github.com/nayrosk/overbrainer/blob/v0.8.0/docs/configuration.md).

## Build the dataset

| Command | Reads | Writes |
|---|---|---|
| `overbrainer subtopics` | `overbrainer.toml` | `data/subtopics.jsonl` |
| `overbrainer questions` | `data/subtopics.jsonl` | `data/questions.jsonl` |
| `overbrainer answers` | `data/questions.jsonl` | `data/answers.jsonl` |
| `overbrainer split` | `data/answers.jsonl` | `data/train.jsonl`, `data/eval.jsonl` |

`overbrainer run` chains the four, then trains when `overbrainer.toml` has a `[training]` section.

- Every stage resumes: items already on disk are skipped, so after an interruption or a partial failure, run the same command again.
- `--topic NAME` limits a stage to one topic. On `split` it only limits the printed counts; both files always hold every topic.
- `--force` regenerates a stage's output for the selected topics (see the rules above). `split` has no `--force`.
- Each stage prints one summary line on stdout with tokens and cost, and progress on stderr. A stage that could not process some items exits non-zero after saving the rest.
- Start small: run each stage with `--topic NAME` on one topic, or lower `subtopics` and `questions_per_subtopic`, and look at the output before the full run. `run` has no `--topic`.

`data/answers.jsonl` holds one JSON object per line: `id`, `topic`, `subtopic`, `messages` (the user question, then the assistant answer with the parent's reasoning in `reasoning_content`) and `meta`. An answer that was truncated, refused, empty, or lacks raw reasoning with `reasoning = true` stays in the file with `meta.excluded` set and is left out of training. `split` reports these as excluded, and answers whose topic or question no longer exists as orphaned:

```
split: 90 train, 10 eval, 3 excluded (truncated 3), 2 orphaned
```

Answers from a parent that never returns raw reasoning are excluded as `no_raw_reasoning` when `reasoning = true` (see the rules above). overbrainer only warns about it once `answers` has started, and then keeps going.

Details: [the dataset pipeline](https://github.com/nayrosk/overbrainer/blob/v0.8.0/docs/pipeline.md).

## Train

`[training]` needs `target`, `base_model` (a Hugging Face repo ID) and `adapter` (`lora`, `qlora` or `full`). Targets:

- `kind = "local"`: this machine, `runtime = "native"` (Axolotl in `venv`) or `"docker"`.
- `kind = "ssh"`: a machine reached with `ssh`, host from `OVERBRAINER_TARGETS__NAME__HOST`. If `ssh` is a sandbox wrapper such as firejail, or missing, `ssh_client = "builtin"` (or `OVERBRAINER_SSH_CLIENT=builtin`) uses the built-in client; see [Training](https://github.com/nayrosk/overbrainer/blob/v0.8.0/docs/training.md#built-in-ssh-client).
- `kind = "runpod"`: a pod created for the run and deleted afterwards. Needs `gpu_types` (a list, or `"auto"` to try every GPU type in stock, cheapest first, with at least the VRAM the model is estimated to need unless `min_vram_gb` is set; up to 3 cheaper types reported out of stock are tried first, and the catalog is read again after each refusal) and `max_hours`, and `OVERBRAINER_RUNPOD__API_KEY`. A watchdog on the pod deletes it at `max_hours` at the latest, after asking the job for a snapshot 15 minutes before when nothing follows it. An optional `max_cost_usd` stops the job with a snapshot at 95% of that spend and deletes the pod at 100%. A disk 92% full also stops the job with a snapshot, unless an optional `max_volume_gb` lets overbrainer grow the network volume first. `ssh_client` works here too.

| Command | What it does |
|---|---|
| `overbrainer train` | Start a run on `training.target` and follow it until it ends. |
| `overbrainer train --target NAME` | Same, on another target. |
| `overbrainer train attach RUN_ID` | Follow a run again, then retrieve its results. |
| `overbrainer train stop RUN_ID` | Stop a run's job with a snapshot (a checkpoint), retrieve it and the results. |
| `overbrainer train --resume-from RUN_ID` | Start a new run from the snapshot of a stopped run, with the same training settings. |
| `overbrainer train cancel RUN_ID` | Stop a run's job and retrieve its artifacts. |
| `overbrainer export RUN_ID [--quantize TYPE] [--ollama NAME]` | Export a succeeded (or stopped) run's model to `runs/RUN_ID/output/gguf/RUN_ID-TYPE.gguf` with an Ollama `Modelfile`, on the run's target; `--ollama NAME` also runs `ollama create`. |
| `overbrainer compare [--run RUN_ID] [--limit N] [--keep-pod]` | Serve the run's GGUF with llama-server on its target, ask the eval questions, have `roles.judge` (else the parent) compare each answer with the parent's; prints the path of `runs/RUN_ID/compares/COMPARE_ID/compare.md`. |
| `overbrainer compare --rejudge COMPARE_ID` | Judge a compare again (another judge or prompt, or after an interruption) without asking the child again. |
| `overbrainer push RUN_ID [--repo NS/NAME] [--public] [--overwrite-card] [--dry-run]` | Push a succeeded (or stopped) run's `output/` to a Hugging Face model repo with a generated card; needs `OVERBRAINER_HF_TOKEN` with write access, except for `--dry-run`. |
| `overbrainer runs ls` | List runs: ID, state, target, creation time, pod. |
| `overbrainer runs logs RUN_ID [--pod] [--source container\|system] [--follow] [--tail N]` | Print a run's job log, or with `--pod` its Runpod pod's logs (kept in the run directory, secrets masked). |
| `overbrainer history [--all]` | Totals per stage (runs, items, tokens, cost), or every execution. |
| `overbrainer migrate [--dry-run]` | Bring a project from before 0.4.0 up to date, once; `--dry-run` only lists the changes. |
| `overbrainer pod ls` | List the Runpod pods overbrainer created. |
| `overbrainer pod rm RUN_ID` | Delete the pods of a run, of an export (`export_...`) or of a compare (`compare_...`). |
| `overbrainer pod gpus` | List Runpod's Secure Cloud GPU types, cheapest first, with price, VRAM and stock. |
| `overbrainer pod datacenters` | List Runpod's data centers with how many GPU types are in stock there. |
| `overbrainer pod volumes` | List the account's network volumes. |
| `overbrainer pod templates` | List the account's pod templates. |

The job runs detached. Ctrl-C, a closed terminal or a lost connection only stop following it; the training goes on. `train` then exits non-zero and prints the `overbrainer train attach RUN_ID` command to use. That does not mean the run failed. A run ends as `succeeded`, `failed`, `cancelled` or `stopped` (stopped with a snapshot: `train --resume-from RUN_ID` goes on from it; prefer `train stop` to `train cancel` when the steps done so far are worth keeping). Its files are in `runs/RUN_ID/`: `job.log` (the job's output, read it when a run fails), `metrics.jsonl`, and `output/` with the LoRA adapter (or the full model, and `output/merged/` with `merge = true`). After a run that succeeded, `train` and `run` print these paths on lines such as `train: adapter in runs/RUN_ID/output`. The `output/` of a stopped run holds a partial model, from the snapshot's step: never use it as a finished one; resume the run instead. Reaching the last step does not end a run: lines such as `train: finalizing (evaluation 340/1200)`, `train: finalizing (saving model)`, `train: merging adapter` and `train: retrieving results` mean the job is still working, and on Runpod the pod still bills; keep following it until the final summary.

To get a model Ollama or llama.cpp can run, export it: `overbrainer export RUN_ID` (default `Q4_K_M`, or `--quantize Q8_0`, `F16`...), or set `[export] after_training = true` so every training job exports at its end. The GGUF and its `Modelfile` land in `runs/RUN_ID/output/gguf/`; `ollama create NAME -f Modelfile` there loads it (overbrainer runs it with `--ollama NAME` or `export.ollama_name`). Ctrl-C cancels an export.

To know whether the child can replace the parent, run `overbrainer compare` after the export. Read the report's win or tie rate with its limits: a judge that is the parent favors its own answers (set `roles.judge`), and fewer than 100 questions move the rate by several points. The cost rows need `[compare]` prices. On Runpod it starts a new pod: tell the user its price before running it (stderr shows it first). Details: [compare](https://github.com/nayrosk/overbrainer/blob/v0.8.0/docs/compare.md).

To publish a model, `overbrainer push RUN_ID`: it commits `output/` (GGUF files at the repo root, no checkpoints) and a generated model card to `NAMESPACE/NAME` (`--repo`, `[hub] repo`, else `<you>/<project>`), private unless `--public` or `[hub] private = false`; an existing public repo is refused unless one of those is set. Run `--dry-run` first: it writes the card to `runs/RUN_ID/hub/README.md` and sends nothing. The card names the topics and the parent model, and some providers' terms forbid training models on their outputs, so check them before making a repo public. `[hub] after_training = true` pushes each run that succeeded; a failed push leaves the run `succeeded`. Ctrl-C cancels a push before its commit, and a rerun resumes. `training.hub_model_id` is deprecated: `overbrainer migrate` moves it to `[hub]`.

Once `train` has stopped following a run, only `overbrainer train attach RUN_ID` retrieves its results. On Runpod, the watchdog deletes the pod `retrieve_grace_minutes` (60 by default) after the job ends if its results were not retrieved, and the results are lost with it. Attach well before that.

Details: [training](https://github.com/nayrosk/overbrainer/blob/v0.8.0/docs/training.md) and [Runpod](https://github.com/nayrosk/overbrainer/blob/v0.8.0/docs/runpod.md).

## Follow long commands

Run a stage, `run` or `train` in the background with its output in a log, then read the log:

```bash
NO_COLOR=1 overbrainer answers > answers.log 2>&1
```

- A stage's progress: count the lines of its output file in `data/`. For `answers`, compare `wc -l data/answers.jsonl` with `wc -l data/questions.jsonl`.
- A training run: follow it with `overbrainer train attach RUN_ID`. `overbrainer runs ls` only shows the state last written to `runs/RUN_ID/run.json`; `train attach` refreshes it.

## Troubleshooting

- `config check` fails: fix the key it names in `overbrainer.toml`. A value shown as `unset`, or a command that names a variable to set: ask the user to set it. Never read `.env` to find out why.
- `cannot parse .env`: the user's `.env` has a syntax error at the given index. Ask them to fix it.
- A provider answers 429 or 5xx: overbrainer retries with backoff up to `max_retries`. If a stage still fails, run it again later; it resumes.
- A run is `failed`: the reason is on the last line of the `train` or `train attach` output, and in the `message` field of `runs/RUN_ID/run.json`. The job's own output is in `runs/RUN_ID/job.log`.
- `train` on Runpod warns about stray pods: report them to the user. Remove them with `overbrainer pod rm RUN_ID` only with the user's consent.
- `this project predates overbrainer 0.4.0: run overbrainer migrate`: run `overbrainer migrate --dry-run`, show the user the changes, then `overbrainer migrate`. The line never stops the command it follows.
- `another overbrainer (pid N) is using this project`: a command that writes to the project (a stage, `run`, `train`, `export`, `compare`, `push`, `tui`, `pod rm`, `migrate`) is already running and holds `.overbrainer/lock`. Do not run a second one; wait for it to finish or ask the user. `train stop RUN_ID` is not refused: it only asks for the snapshot, which the running process collects. `overbrainer history`, `runs ls`, `runs logs`, `pod ls` and `config check` never hit this, since they only read.
