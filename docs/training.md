# Training

`overbrainer train` fine-tunes `training.base_model` with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) 0.19 on `data/train.jsonl`, evaluating on `data/eval.jsonl`. It runs on the target named by `training.target`, or by `--target NAME`.

| Command | What it does |
|---|---|
| `overbrainer train [--target NAME]` | Start a run and follow it until the job ends. |
| `overbrainer train attach RUN_ID` | Follow a run again, then retrieve its results. |
| `overbrainer train cancel RUN_ID` | Stop the job of a run and retrieve its artifacts. |
| `overbrainer runs ls` | List the runs: ID, state, target, creation time, and the pod of a Runpod run. |
| `overbrainer pod ls` | List the Runpod pods overbrainer created, with their run. |
| `overbrainer pod rm RUN_ID [--force]` | Delete every pod of a run and wait until Runpod no longer shows them. |

`train --keep-pod` and the `pod` commands only apply to Runpod targets, described in [Runpod](runpod.md).

## Runs

A run gets an ID such as `20260922-143005-a1b2` and a directory `runs/<run-id>/`. It holds `axolotl.yaml`, copies of the train and eval files, the metrics plugin and `run.json` (target, job, state). Once the job has ended, it also holds `metrics.jsonl`, `job.log` and `output/`. `output/` holds the LoRA adapter (or the full model with `adapter = "full"`) and, with `merge = true`, the merged model in `output/merged/`. Intermediate `checkpoint-*` directories stay on the target.

The job runs detached from overbrainer. Once it has started, Ctrl-C, a closed terminal or a lost SSH connection stop overbrainer from following it; the training goes on. Starting a run is never interrupted: a Ctrl-C pressed while a run is starting is only acted on once the job has actually started, so the command always finishes starting before it detaches. After Ctrl-C, overbrainer prints the `overbrainer train attach` command that follows the run again and exits with an error status. It does the same after six failed attempts in a row to reach the target.

`overbrainer runs ls` shows the state last recorded in `run.json`; `train attach` refreshes it. Progress (step, epoch, loss, learning rate, every evaluation) goes to stderr, and the final summary to stdout:

```
train: run 20260922-143005-a1b2 succeeded; step 1200/1200, epoch 3.00, loss 0.4123, eval_loss 0.5012; output in runs/20260922-143005-a1b2/output
```

A run fails when the job exits with a non-zero code, or when it writes no metric line at all, which means Axolotl did not load the metrics plugin. `runs/<run-id>/job.log` holds the job's output.

## Cancelling

`overbrainer train cancel` needs a `[training]` section, because cancelling a run also retrieves its artifacts, as a successful or failed run does. It accepts any run that started a job, whatever its recorded state. A run already recorded as ended prints `run RUN_ID already ended: STATE; stopping any job left on the target` and the cancel runs anyway, which stops a container that outlived its job. A job that had already ended is not signalled: cancel then reports what it found and tells you to run `overbrainer train attach RUN_ID` to collect it.

A run already recorded as cancelled fails with `run RUN_ID already ended: cancelled`, and a run that never started a job fails with `run RUN_ID has not started`. Both exit non-zero without touching anything.

## Targets

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

`venv` applies only with `runtime = "native"`, and `engine` and `image` only with `runtime = "docker"`. The third kind, `runpod`, has [its own page](runpod.md).

### The docker runtime

Each run gets one container, named `overbrainer-<run-id>`, with all GPUs, the host IPC namespace and the run directory mounted at `/workspace/run`. The Hugging Face cache is `runs/.hf-cache` (local) or `<workdir>/.hf-cache` (SSH), shared by the runs of the target, so a base model is downloaded once.

Docker needs the NVIDIA Container Toolkit (`--gpus all`). Podman needs its CDI specification (`--device nvidia.com/gpu=all`, generated with `nvidia-ctk cdi generate`). The default image is built for CUDA 13 and needs an NVIDIA driver from the 580 series or newer. With rootful Docker, the files the container writes are owned by root.

Over SSH, the remote machine may kill user processes at logout (`KillUserProcesses=yes` without lingering, see below). The wrapper then dies with the session while the container keeps running and holding the GPU, and overbrainer reports the run as failed. `overbrainer train cancel <run-id>` stops that container, whatever the run's recorded state; `docker stop overbrainer-<run-id>` (or `podman stop overbrainer-<run-id>`) is the manual fallback.

### The native runtime

The native runtime runs `<venv>/bin/axolotl` on the host. Over SSH, if systemd-logind kills user processes at logout (`KillUserProcesses=yes`), enable lingering for the SSH user (`loginctl enable-linger`) so the job survives the disconnection.

The job's environment differs by target. On a local target it is overbrainer's own, without the `OVERBRAINER_*` and `VAULT_*` variables. Over SSH it is whatever a non-interactive `sh -c` gets there, with no login shell and no `~/.bashrc`. A setup that a profile or a conda activation provides is missing over SSH: put it in the venv, or in the SSH server's environment.

### SSH

SSH uses your `ssh` binary with `~/.ssh/config`, the agent and `known_hosts`. A host that is not already in `known_hosts` is refused. Files travel as `tar` streams over the connection, so the remote machine needs `tar`, `setsid` and `nohup` (any Linux distribution has them). overbrainer keeps one master connection open (OpenSSH `ControlMaster`); an `ssh` wrapper that kills background processes, such as a firejail profile, breaks it.

## The Hugging Face token

`OVERBRAINER_HF_TOKEN` (a literal or a `vault:` reference) is needed for gated or private base models, and for `hub_model_id`, which pushes the adapter (not the merged model) to a private Hub repository.

It is resolved only when a run starts and reaches the job only as its `HF_TOKEN` environment variable. overbrainer never writes it to `axolotl.yaml`, `run.json` or any other file, never puts it on a command line, and sends it over SSH on the command's standard input. With the `docker` runtime it does reach one file overbrainer does not write: the container engine stores the environment it was started with in the container's own configuration, where `docker inspect` (or `podman inspect`) shows it until the container is removed.

## Reasoning in the chat template

The parent's reasoning is trained only if the chat template renders `reasoning_content`. Axolotl's `qwen3`, `qwen3_5`, `exaone4`, `gemma4` and `gemma4_unified` templates do, as does the official template of Qwen3 models; other templates may drop it without any error. overbrainer warns at the start of a run when the template in use (the base model's own, or `axolotl_extra.chat_template`) is not one of them. The check goes by name only, so a local path or a renamed model may get the warning wrongly.

## Training settings

`[training]`:

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

overbrainer also sets `attn_implementation: sdpa` (no extra package needed), `gradient_checkpointing: true`, `warmup_ratio: 0.1` and `logging_steps: 1`.

`[training.axolotl_extra]` is merged into the generated YAML last. Tables merge key by key, and any other value replaces the generated one, so it can change these too: for example `attn_implementation = "flash_attention_2"` on an image with flash-attn installed, or `chat_template = "qwen3"`. An `eval_steps` or `save_steps` there replaces `evals_per_epoch` or `saves_per_epoch`. It cannot set a key that has a typed setting above (use the setting), nor the keys overbrainer manages: `datasets`, `test_datasets`, `val_set_size`, `output_dir`, `dataset_prepared_path`, `plugins`.

Values set through the environment (`OVERBRAINER_TRAINING__AXOLOTL_EXTRA__WARMUP_STEPS=10`) arrive as text. overbrainer turns `true`, `false`, integers and decimal numbers into booleans and numbers, and leaves anything else as text. A number-like value that must stay text belongs in `overbrainer.toml`.
