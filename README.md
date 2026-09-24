# overbrainer

Distill a large "parent" LLM into a smaller open-weights "child" model, from the command line.

[![CI](https://github.com/nayrosk/overbrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/nayrosk/overbrainer/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Latest release](https://img.shields.io/github/v/release/nayrosk/overbrainer?include_prereleases&sort=semver)](https://github.com/nayrosk/overbrainer/releases)

![overbrainer tui: the dataset tree with an answer's reasoning, the topic stats, the help, a training run's loss chart and a dialog](docs/assets/tui-tour.gif)

overbrainer asks an LLM for questions on your topics, collects answers and their reasoning from a parent model, then fine-tunes a child model with Axolotl on this machine, over SSH or on Runpod. A terminal UI shows the dataset and the training runs as they progress.

Status: early development, not yet released.

## How it works

```mermaid
flowchart LR
    T[Topics in overbrainer.toml] --> S[Subtopics]
    S --> Q[Questions]
    Q --> A[Parent answers with reasoning]
    A --> SP[Train and eval split]
    SP --> F{Axolotl fine-tune}
    F -->|Local| L[LoRA adapter]
    F -->|SSH| L
    F -->|Runpod| L
```

A generator model writes the subtopics and questions, and drops near-duplicates. The parent model answers each question; answers that were truncated, refused or empty, or that lack the raw reasoning asked for, are kept for inspection and left out of training. `split` writes `data/train.jsonl` and `data/eval.jsonl` in Axolotl's chat format, and `train` fine-tunes the base model on them.

## Install

```bash
cargo install --locked --git https://github.com/nayrosk/overbrainer
```

There are no prebuilt binaries. You need:

- Rust 1.89 or newer to build it. Linux and macOS are supported.
- `ssh` for SSH targets, and `ssh` with `ssh-keygen` for Runpod targets.
- For training on this machine: Axolotl 0.19 in a virtual environment or on `PATH`, or Docker or Podman with NVIDIA GPU access to run the Axolotl image.

## Quickstart

![overbrainer init, config check and runs ls in a terminal](docs/assets/cli.gif)

```bash
overbrainer init my-project && cd my-project
cp .env.example .env        # fill in the provider URL and key
overbrainer config check    # prints the resolved configuration, secrets masked
overbrainer run             # subtopics, questions, answers, split, then train
overbrainer tui             # browse the dataset and follow the runs
```

`init` writes `overbrainer.toml`, `.env.example`, the prompt templates in `prompts/` and a `.gitignore`. Edit the topics in `overbrainer.toml` before `run`. `run` trains only when `overbrainer.toml` has a `[training]` section; otherwise it stops after `split`. Each stage also runs on its own (`subtopics`, `questions`, `answers`, `split`, `train`) and resumes where it stopped: see [the dataset pipeline](docs/pipeline.md).

## Providers

A provider is any API that speaks the OpenAI chat completions protocol or the Anthropic Messages API. `overbrainer.toml` names it and its protocol; the URL and the key come from the environment. OpenRouter and NanoGPT both work, and both publish prices, so every stage prints what it cost.

```toml
[providers.openrouter]
protocol = "openai"

[providers.nanogpt]
protocol = "openai"

[roles]
generator = { provider = "nanogpt", model = "qwen/qwen3-235b-a22b-instruct-2507" }
parent = { provider = "openrouter", model = "deepseek/deepseek-r1", reasoning = true }
```

```bash
OVERBRAINER_PROVIDERS__OPENROUTER__BASE_URL=https://openrouter.ai/api/v1
OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY=sk-or-...
OVERBRAINER_PROVIDERS__NANOGPT__BASE_URL=https://nano-gpt.com/api/v1
OVERBRAINER_PROVIDERS__NANOGPT__API_KEY=...
```

NanoGPT: [nano-gpt.com](https://nano-gpt.com/r/HLFz47L6) (referral link). Any key can also be a `vault:` reference to a Vault or OpenBao secret. [Configuration](docs/configuration.md) covers roles, pipeline settings, secrets and logs.

## Training targets

| Target | Where the job runs | Cost and guards |
|---|---|---|
| `local` | This machine, natively or in a Docker or Podman container. | Your own GPU. |
| `ssh` | A machine you reach with `ssh`, natively or in a container. | Your own GPU. The job survives a lost connection. |
| `runpod` | A pod created for the run on Runpod Secure Cloud, deleted once the results are back. | Billed per second by Runpod. A watchdog on the pod deletes it at `max_hours` at the latest, and `gpu_types` are tried in order until one is available. |

Runpod: [runpod.io](https://runpod.io?ref=ym24z23f) (referral link). A run's LoRA adapter (or full model) lands in `runs/<run-id>/output/`. `train attach` follows a run again after Ctrl-C, and `train cancel` stops it. [Training](docs/training.md) covers targets, settings and the Hugging Face token; [Runpod](docs/runpod.md) covers pods, the watchdog and costs.

## Terminal UI

`overbrainer tui` has four views:

- Dataset (`1`): the topics, subtopics, questions and answers as a tree, with each answer's reasoning, stats and a filter. Edit or delete items in place.
- Pipeline (`2`): run a stage and watch its progress, tokens and cost.
- Training (`3`): the runs, with progress, pod, spend, a loss chart and learning rate and gradient norm sparklines. Start, follow or cancel a run.
- Logs (`4`): the captured log lines, filtered by level.

| Keys | Action |
|---|---|
| `1` to `4`, Tab, Shift-Tab | Switch view. |
| `?` | The keys of the current view. |
| `j` `k`, `l` `h` | Move, expand and collapse. |
| `/`, `s` | Filter the tree, show the stats. |
| `e`, `d` | Edit in `$EDITOR`, delete (asks first). |
| `r` | Run a pipeline stage. |
| `t`, `a`, `c` | Start, attach to, or cancel a training run. |
| `R` | Reload from disk. |
| `q`, Ctrl-C | Quit (asks first when work is running). |

The TUI draws its own crimson theme in 24-bit or 256 colors, and falls back to the terminal's 16 colors. `OVERBRAINER_TUI_COLOR` (`truecolor`, `256` or `16`) and `OVERBRAINER_TUI_MOTION` (`on`, `reduced` or `off`) override the detection, and `NO_COLOR` turns it monochrome. [The TUI page](docs/tui.md) has every key and behavior.

## Documentation

- [The dataset pipeline](docs/pipeline.md): stages, resuming, deduplication, the split, the answer format, reasoning and prompt templates.
- [Configuration](docs/configuration.md): `overbrainer.toml`, environment variables, providers, roles, pipeline settings, Vault and logs.
- [Training](docs/training.md): runs, local and SSH targets, the Hugging Face token, chat templates and training settings.
- [Runpod](docs/runpod.md): pods, the watchdog, `--keep-pod`, stray pods and custom images.
- [Terminal UI](docs/tui.md): views, keys, editing, quitting, color and motion.

## Releasing

Releases are cut by hand from `main`:

1. Merging to `main` runs release-plz, which opens or updates a release pull request with the version bump and the `CHANGELOG.md` entry.
2. Review and merge that pull request.
3. Tag the merge commit with a signed tag and push it: `git tag -s v0.2.0 -m v0.2.0 && git push origin v0.2.0`.
4. The tag runs the checks, publishes the crate to crates.io and creates the GitHub release from the changelog entry.

## Terms of service

Several model providers forbid using their outputs to train models that compete with them. This includes OpenAI and Anthropic, and the restriction still applies when their models are reached through a gateway such as OpenRouter or NanoGPT. Check the terms of the models you configure as parent before training on their outputs.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
