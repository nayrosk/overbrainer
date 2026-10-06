# Training

`overbrainer train` fine-tunes `training.base_model` with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) 0.19 on `data/train.jsonl`, evaluating on `data/eval.jsonl`. It runs on the target named by `training.target`, or by `--target NAME`.

| Command | What it does |
|---|---|
| `overbrainer train [--target NAME]` | Start a run and follow it until the job ends. |
| `overbrainer train --resume-from RUN_ID` | Start a run from the snapshot of a stopped run. |
| `overbrainer train attach RUN_ID` | Follow a run again, then retrieve its results. |
| `overbrainer train stop RUN_ID` | Stop the job of a run with a snapshot, then retrieve it with the results. |
| `overbrainer train cancel RUN_ID` | Stop the job of a run and retrieve its artifacts. |
| `overbrainer export RUN_ID [--quantize TYPE] [--ollama NAME] [--keep-pod]` | Export the model of a succeeded or stopped run to GGUF, with an Ollama Modelfile (see [Export to GGUF and Ollama](#export-to-gguf-and-ollama)). |
| `overbrainer runs ls` | List the runs: ID, state, target, creation time, the step of a stopped run's snapshot (`step N (reason): partial model in output/`), and the pod of a Runpod run. |
| `overbrainer runs logs RUN_ID [--tail N]` | Print the run's `job.log`. With `--pod [--source container\|system] [--follow]`, the logs of its Runpod pod instead (see [Runpod](runpod.md#pod-logs)). |
| `overbrainer pod ls` | List the Runpod pods overbrainer created, with their run. |
| `overbrainer pod rm RUN_ID [--force]` | Delete every pod of a run and wait until Runpod no longer shows them. |

`train --keep-pod`, `export --keep-pod` and the `pod` commands only apply to Runpod targets, described in [Runpod](runpod.md).

## Runs

A run gets an ID made of the project name and the UTC time it was created, such as `malware_development_20260922-143005`. The project name is lowercased, every run of characters other than ASCII letters and digits becomes one `_`, and it is cut to 40 characters (`run` when nothing is left). A second run created in the same second gets `_2`, then `_3`, and so on. The run directory on the target is claimed as well, with a `.claim` file created exclusively, so a run started in the same second from another checkout of the project against the same work directory also moves on to the next ID instead of overwriting the other run. On Runpod the ID is fixed before the pod exists, so the pod's bootstrap claims the directory itself, before it writes anything there. When a run from another checkout already owns the directory on a shared network volume, the pod leaves it untouched and its watchdog deletes the pod; the run fails without touching the other run's files, and starting it again gives it a new ID. Runs created before v0.5.0 keep their IDs, such as `20260922-143005-a1b2`. `runs ls` and the TUI list runs by creation time.

Each run has a directory `runs/<run-id>/`. It holds `axolotl.yaml`, copies of the train and eval files, the metrics plugin and `run.json` (target, job, state). Once the job has ended, it also holds `metrics.jsonl`, `job.log` and `output/`. `output/` holds the LoRA adapter (or the full model with `adapter = "full"`) and, with `merge = true`, the merged model in `output/merged/`. After a run that succeeded, `train`, `train attach` and `run` print these paths (`train: adapter in runs/<run-id>/output`, then `train: merged model in runs/<run-id>/output/merged`). Intermediate `checkpoint-*` directories stay on the target, except the snapshot of a stopped run (see below). The adapter or model in the `output/` of a stopped run is partial.

The job runs detached from overbrainer. Once it has started, Ctrl-C, a closed terminal or a lost SSH connection stop overbrainer from following it; the training goes on. Starting a run is never interrupted: a Ctrl-C pressed while a run is starting is only acted on once the job has actually started, so the command always finishes starting before it detaches. After Ctrl-C, overbrainer prints the `overbrainer train attach` command that follows the run again and exits with an error status. It does the same after six failed attempts in a row to reach the target.

`overbrainer runs ls` shows the state last recorded in `run.json`; `train attach` refreshes it. Progress (step, epoch, loss, learning rate, every evaluation) goes to stderr, and the final summary to stdout:

```
train: run malware_development_20260922-143005 succeeded; step 1200/1200, epoch 3.00, loss 0.4123, eval_loss 0.5012; output in runs/malware_development_20260922-143005/output
```

The last training step is not the end of the job: the trainer then runs its final evaluation and saves the model, `merge-lora` merges the adapter, and overbrainer retrieves the results. To show that work, the job writes a `stage` line into `metrics.jsonl` before each command (`train`, `merge`), and the metrics plugin adds an `eval` line at most every 2 seconds while an evaluation runs (its step and, when the evaluation's length is known, its total) and an `end` line once the training loop is over. From these, `train` and `train attach` print a line when the phase changes, and every tenth of an evaluation:

```
train: evaluating 120/1200
train: training
train: finalizing (evaluation 340/1200)
train: finalizing (saving model)
train: merging adapter
train: retrieving results
```

A run started before 0.6.0 writes none of these lines: it reads `finalizing (saving model)` once its step reaches `max_steps`. Older versions of overbrainer skip the new lines, with a warning, as they skip any line they do not know.

While it follows a job, overbrainer also samples the machine the job runs on every 10 seconds, over the same connection: one shell script reads `nvidia-smi`, `/proc` (load, CPU time, memory), the cgroup files of a container, v2 or else v1 (its memory limit and CPU quota come first, since `/proc` shows the host inside a Runpod pod; `/proc` is used only when no cgroup limits the container) and `df` on the run directory and `/`. `nvidia-smi` and `df` get 5 seconds each, so a stuck driver or network mount cannot hold the probe. On a network volume (a Runpod network volume is a shared MooseFS cluster), `df` reports the whole cluster, not the volume: those figures are shown as shared and never warn, and the volume's own usage is measured separately. The samples feed the TUI's system panel and the [metrics](configuration.md#metrics); a sample that fails is skipped, logged at debug level, and never stops the follow. A target without `nvidia-smi` simply shows no GPU.

A run fails when the job exits with a non-zero code, or when it writes no metric line at all, which means Axolotl did not load the metrics plugin. `runs/<run-id>/job.log` holds the job's output.

## Stopping with a snapshot

`overbrainer train stop RUN_ID` stops a running job without losing what it trained. It writes `snapshot.request` in the run directory on the target; the metrics plugin sees it at the end of the current step, saves a full checkpoint (the adapter or model, the optimizer and scheduler state, the random state), stops training, and writes `snapshot.json` with the checkpoint and its step. `merge-lora` is skipped. The command then follows the job as `train attach` does until it ends, records the run `stopped`, and retrieves the checkpoint (`runs/<run-id>/output/checkpoint-N/`) in a second pass, checked against the target's SHA-256 manifest like the other artifacts. On Runpod the pod is then deleted.

While another overbrainer process holds the project (`train`, `run`, `train attach` or the TUI following the run), `train stop` does not follow the run itself: it writes `snapshot.request` on the target, says which process holds the project, and exits. The process following the run then records it `stopped` and retrieves the checkpoint as above. `train stop` reads `run.json` and `pod.json` then, and writes nothing else in the run directory; on Runpod it reaches the pod through the `runs/<run-id>/ssh/config` that process wrote, with no Runpod API call. If the holder does not follow that run, `overbrainer train attach RUN_ID` collects the snapshot once it is free.

When training ends early, Axolotl still saves the adapter (or the model with `adapter = "full"`) at the top of `output/`, as it does at the end of a full run. After a stop that model is partial: it holds the training up to the snapshot's step, not a finished model. overbrainer leaves it where Axolotl put it and says so:

```
train: run 20260922-143005-a1b2 stopped at step 1240 (requested): snapshot in runs/20260922-143005-a1b2/output/checkpoint-1240; resume with `overbrainer train --resume-from 20260922-143005-a1b2`
train: runs/20260922-143005-a1b2/output also holds the partial model at step 1240, not a finished one
```

`runs ls` shows `step 1240 (requested): partial model in output/` for that run, and the TUI's Training view `snapshot at step 1240 (requested), output/ partial: T resumes it`. To get a finished model, resume the run.

A job that neither saves its snapshot nor ends within 30 minutes of the request is cancelled. A process that follows a run without having asked for the snapshot (`train attach`, or the holder above) looks for a request every 30 seconds and applies the same limits from when it sees one. One that saved it but has not ended 10 minutes later is cancelled too, and still recorded `stopped` with its snapshot. A request that lands after the last step changes nothing: the run succeeds as usual. Under distributed training, rank 0 reads the request and tells the other ranks, so they all stop at the same step.

A run started by overbrainer before 0.5.0 runs a metrics plugin that ignores `snapshot.request`. `train stop` and `s` in the TUI refuse it before writing anything, and suggest `overbrainer train cancel RUN_ID` or letting it finish; a process that follows it never looks for a request, so nothing cancels it while waiting for a snapshot. overbrainer tells such a run apart by its `run.json`, which lacks the `"snapshots": true` that runs started since 0.5.0 have.

`run.json` of a stopped run holds `snapshot`: `checkpoint`, `step` and `reason`, which is `requested` for `train stop` and the TUI, `deadline`, `cost` or `disk` for the automatic snapshots of a Runpod run (see [Runpod](runpod.md)).

## Resuming

`overbrainer train --resume-from RUN_ID` starts a new run from the snapshot of a stopped run, on any target, `--target` included. The checkpoint is copied into `runs/<new-run-id>/resume/checkpoint-N/` (hard-linked when both are on the same file system), uploaded with the run, and passed to Axolotl as `resume_from_checkpoint`. The new run trains on the stopped run's own `data/train.jsonl` and `data/eval.jsonl`, not on the project's current ones, and its `run.json` names the stopped run in `resumed_from`.

A resumed checkpoint only makes sense with the settings it was trained with. The run is refused when a key of the generated Axolotl config differs from the stopped run's `axolotl.yaml`, apart from these, which do not change what the checkpoint was trained with: `hub_model_id`, `hub_strategy`, `evals_per_epoch`, `eval_steps`, `eval_strategy`, `saves_per_epoch`, `save_steps`, `save_strategy`, `save_total_limit`, `logging_steps`, `use_tensorboard`, `use_wandb`, `use_mlflow`, `use_comet`, and any `wandb_*`, `mlflow_*` or `comet_*` key: for example `cannot resume from run RUN_ID: the training settings differ from the ones it ran with (learning_rate)`. A run that is not stopped, or whose checkpoint is not in `runs/`, is refused too; `train attach` retrieves a checkpoint that was not.

## Cancelling

`overbrainer train cancel` needs a `[training]` section, because cancelling a run also retrieves its artifacts, as a successful or failed run does. It accepts any run that started a job, whatever its recorded state. A run already recorded as ended prints `run RUN_ID already ended: STATE; stopping any job left on the target` and the cancel runs anyway, which stops a container that outlived its job. A job that had already ended is not signalled: cancel then reports what it found and tells you to run `overbrainer train attach RUN_ID` to collect it.

A run already recorded as cancelled fails with `run RUN_ID already ended: cancelled`, and a run that never started a job fails with `run RUN_ID has not started`. Both exit non-zero without touching anything.

## Export to GGUF and Ollama

A run ends with a Hugging Face directory: the adapter in `output/`, or the model of a full fine-tune. To run it with llama.cpp or Ollama, overbrainer exports it to GGUF on the run's own target, where Axolotl, torch and transformers already are:

1. Without `output/merged/` (a run with `merge = true` has it), the adapter is merged into its base model with `axolotl merge-lora`, in a scratch directory `export-work/`. A full fine-tune is converted as it is.
2. llama.cpp's `convert_hf_to_gguf.py` converts it to a 16-bit GGUF (`--outtype auto`), which carries the tokenizer and the model's chat template (`tokenizer.chat_template`).
3. `llama-quantize` quantizes it to the chosen type. `F16` and `BF16` skip this step: the conversion writes them directly.
4. `export-work/` is removed, even when a step fails.

The result is `runs/<run-id>/output/gguf/<run-id>-<TYPE>.gguf`, with a `Modelfile` beside it:

```
FROM ./rust_mentor_20261001-120000-Q4_K_M.gguf
PARAMETER num_ctx 4096
```

`num_ctx` is the run's `sequence_len`, read from its own `axolotl.yaml`. The Modelfile has no `TEMPLATE`: Ollama takes the chat template from the GGUF. With `--ollama NAME` (or `export.ollama_name`) and `ollama` on `PATH`, overbrainer runs `ollama create NAME -f Modelfile` in that directory once the GGUF is back; without `ollama`, it warns and prints the command. Each export records `export.json` (`quantize`, `llama_cpp`, `file`, `sha256`, `size`, `created`) in its job directory. Older GGUF files of other types stay in `output/gguf/`; the Modelfile names the newest.

There are two ways to export:

- `[export] after_training = true` exports in the training job itself, after the merge, so `output/gguf/` comes back with the rest of `output/`. It is skipped like the merge when the job stops with a snapshot. The TUI shows the export as a stage of the job.
- `overbrainer export RUN_ID` exports a run that ended: one `succeeded`, or one `stopped` with a snapshot (its partial model in `output/` when Axolotl saved one, else its checkpoint; the export is then of a partial model too). It takes the project lock like `train`. The job gets a directory of its own, `runs/<run-id>/exports/<export-id>/`, with its `run.json`, `job.log`, `metrics.jsonl` and `export.json`; the export ID is `export_<date>-<time>`. On a local or SSH target it runs there, reading the run's `output/` and `axolotl.yaml` where they are (two directories up); when the run directory on an SSH target no longer holds them, the local `output/` (never the checkpoints, but a stopped run's checkpoint when that is the model) and `axolotl.yaml` are uploaded first. On a Runpod target it runs on a new pod, with the target's GPU settings (see [Runpod](runpod.md#exports)). Ctrl-C cancels it: unlike a training job, an export is not left running.

```
export: export_20261001-121500: runs/rust_mentor_20261001-120000/output to GGUF Q8_0 on target `local`
export: GGUF in runs/rust_mentor_20261001-120000/output/gguf/rust_mentor_20261001-120000-Q8_0.gguf (Q8_0, 639.4 MB), Modelfile beside it
export: Ollama model mentor created: `ollama run mentor`
```

`[export]`:

| Key | Default | Meaning |
|---|---|---|
| `after_training` | `false` | Export at the end of each training job, after the merge. |
| `quantize` | `Q4_K_M` | `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, `Q3_K_M`, `Q2_K`, `F16` or `BF16`. `--quantize` overrides it. |
| `ollama_name` | none | Ollama model `ollama create` makes from the Modelfile, when `ollama` is on `PATH`. `--ollama` overrides it. |

### llama.cpp

The export uses one pinned llama.cpp release, `b11320`: the source tarball of the tag (for `convert_hf_to_gguf.py`, its `conversion/` package and `gguf-py/`) and the prebuilt binaries of the target's platform (for `llama-quantize` and its libraries): `ubuntu-x64` on Linux x86_64 (glibc 2.34 or newer), `ubuntu-arm64` on Linux aarch64 (glibc 2.38), `macos-arm64` on macOS with Apple silicon. Any other platform fails with a message naming it, unless the type is `F16` or `BF16`, which need no `llama-quantize`. The job downloads them, checks each against the SHA-256 overbrainer pins before extracting it, and caches them in `<workdir>/.cache/llama.cpp/b11320/` on the target (`runs/.cache/` for a local target), shared by the runs; on a Runpod network volume the cache outlives the pods. A container sees that cache at `/workspace/cache`. The conversion runs with `gguf-py` on `PYTHONPATH` and the target's own Python, which must have `torch`, `numpy`, `transformers` and `yaml` (the Axolotl images do); a missing one fails the export with its name.

Bumping llama.cpp, in `src/export/llama_cpp.rs`:

1. Set `TAG` to the new release tag.
2. Take the `digest` of `llama-<tag>-bin-ubuntu-x64.tar.gz`, `-ubuntu-arm64`, `-macos-arm64`, `-ubuntu-cuda-<cuda>-x64`, `-ubuntu-cuda-<cuda>-arm64` and `-ubuntu-rocm-<rocm>-x64`, and of `cudart-llama-<tag>-bin-ubuntu-cuda-<cuda>-x64.tar.gz` and `-arm64`, from `gh api repos/ggml-org/llama.cpp/releases/tags/<tag>`, and set the `*_SHA256` constants. Set `CUDA` to the CUDA version of those builds. The CUDA builds bundle their own runtime: the compare script puts the matching cudart archive's libraries first on `LD_LIBRARY_PATH` on every CUDA host, and falls back to the CPU build when the driver's CUDA version (from `nvidia-smi`) is below the major version of `CUDA`. Set `ROCM` to the ROCm label of the ROCm build. It does not bundle its runtime (`libamdhip64.so.7`, `librocblas.so.5`, `libhipblas.so.3` for the pinned release): the compare script uses it on a Linux x86_64 host with `/dev/kfd` and those libraries in `ldconfig -p`, and falls back to the CPU build, with a warning, when a library is missing or the user cannot read and write `/dev/kfd` (the `render` group). There is no ROCm build for arm64. When a bump changes the ROCm major version, update the library names in `pick_amd` in `src/compare/compare.sh` and in its warning.
3. Download `https://github.com/ggml-org/llama.cpp/archive/refs/tags/<tag>.tar.gz`, run `sha256sum` on it, and set `SOURCE_SHA256`.
4. Check that the archives still hold `llama.cpp-<tag>/convert_hf_to_gguf.py`, `llama-<tag>/llama-quantize` and `llama-<tag>/llama-server`, the cudart archives `libcudart.so.<major>`, and that `libggml-hip.so` of the ROCm archive still needs the libraries `pick_amd` checks (`readelf -d`), then export a small run on a GPU target, load it in Ollama, and compare it.

## Push to Hugging Face

`overbrainer push RUN_ID` uploads the model of a finished run to a Hugging Face model repo, in one commit, with a model card it generates. The run needs its `output/` in `runs/`: a `succeeded` run, or a `stopped` one whose partial model is in `output/` (the card does not say it is partial, so push it knowingly). A run with only a checkpoint is refused, with a hint to export or resume it, and so is one whose results were not retrieved (`train attach` gets them).

Everything in `runs/<run-id>/output/` goes, except `checkpoint-*` directories, `debug.log`, hidden files, Axolotl's own `README.md` and any other `README.md` that would land at the repo root, where the card goes. The layout in the repo:

- The adapter or model files (`adapter_config.json`, `adapter_model.safetensors`, the tokenizer files) sit at the root, as they do in `output/`.
- `merged/` keeps its name.
- The files of `output/gguf/` go to the root of the repo, not into a `gguf/` directory. That is where `ollama run hf.co/<repo>:<quant>` and `llama-cli -hf <repo>:<quant>` look for them. The `Modelfile` goes with them.
- `README.md` is the generated card.

The repo is `--repo NAMESPACE/NAME`, else `[hub] repo`, else `<you>/<project.name>` with `_` replaced by `-`, where `<you>` is the account of the token. It is created private unless `--public` or `[hub] private = false`. Visibility only applies when the repo is created: an existing repo keeps its own, and the command says so (`repo me/my-demo exists and stays private`). The one exception is a private push into an existing public repo, which is refused before anything is read or uploaded: pass `--public` or set `[hub] private = false` to push to it. Pushing again adds a commit to the same repo.

The card is replaced only when the repo has no `README.md`, or when its `README.md` carries the marker `<!-- overbrainer:card -->` that the generated card ends with. A card you wrote yourself is kept: the other files are pushed and the command says `kept the repo's own README.md (use --overwrite-card to replace it)`. `--overwrite-card` replaces it anyway.

`--dry-run` pushes nothing and needs no token. It lists the files with their sizes and writes the card to `runs/<run-id>/hub/README.md`, so you can read it first. Without a token the repo name shows `<you>` for the account.

```
push: <you>/my-demo, private when created; nothing is sent (--dry-run)
adapter_config.json  0.0 MB
adapter_model.safetensors  0.0 MB
push: 2 files, 0.0 MB; card written to runs/20260929-054448-62aa/hub/README.md
```

A real push ends with the commit URL, and writes `runs/<run-id>/hub/push.json` (`repo`, `commit`, `url`, `files`, `private`, `pushed`):

```
push: https://huggingface.co/me/my-demo/commit/abc123 (2 files, 0.0 MB)
```

Ctrl-C cancels a push before its commit: nothing is committed (`push cancelled before its commit finished; run it again to resume`), and running it again resumes, because Hugging Face skips the chunks it already stored. A token without write access to the namespace fails with `the Hugging Face token needs write access to <namespace>`; the token itself is never printed.

`[hub] after_training = true` pushes each run once it succeeded and came back, after the export when `[export] after_training` is on. It applies to `train` and `train attach`, to auto mode and to runs started from the TUI; a run that ends `stopped` is not pushed, nor one already pushed (it has `runs/<run-id>/hub/push.json`), so attaching it again makes no new commit: `overbrainer push RUN_ID` pushes it again. A failed push does not change the run, which stays `succeeded`: it warns, for example `push failed: ...; run it again with: overbrainer push RUN_ID`. Ctrl-C during that push stops it the same way (`push cancelled before its commit finished; run it again with: overbrainer push RUN_ID`). Without `OVERBRAINER_HF_TOKEN`, the training start already warns that the push will fail. In the TUI, `h` on a run in the Training view pushes it after a confirmation (see [the TUI page](tui.md)).

`[hub]`:

| Key | Default | Meaning |
|---|---|---|
| `repo` | none | `NAMESPACE/NAME`. `--repo` overrides it. |
| `private` | `true` | Create the repo private. `--public` overrides it. |
| `after_training` | `false` | Push each run when it succeeded and came back. |

`OVERBRAINER_HUB__BASE_URL` points the push at another Hub, over `https` (or `http` on a loopback address, for a test stub); it has no `overbrainer.toml` key.

Before you publish a repo, read its card. It names the topics and their descriptions and the parent model, which is why repos are private by default. Some providers' terms forbid training models on their outputs: check the terms of the parent's provider before you make a repo public.

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
# ssh_client = "openssh"       # openssh (default) | builtin, see "Built-in SSH client" below
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

#### Built-in SSH client

Some machines cannot run overbrainer's `ssh` the way it needs: a sandbox wrapper such as firejail kills the background master connection, or there is no `ssh` binary at all. The built-in client is a pure-Rust SSH client inside overbrainer that needs neither. Choose it with `ssh_client = "builtin"` on an `ssh` or `runpod` target, or for every target with `OVERBRAINER_SSH_CLIENT=builtin`, which wins over the field. `openssh` is the default.

Release binaries include it; a build from source needs `cargo install --locked overbrainer --features builtin-ssh`. Without the feature, `builtin` is refused with `ssh_client = "builtin" needs a build with the builtin-ssh feature (the release binaries have it)`, or with `OVERBRAINER_SSH_CLIENT=builtin needs ...` when the variable chose it. `overbrainer config check` shows the client of each target.

It reads `~/.ssh/config`, then `/etc/ssh/ssh_config`, with OpenSSH rules: the first value obtained wins (the values of `IdentityFile` add up), `Host` patterns (`*`, `?`, `!`) match, and `Include` globs are relative to `~/.ssh` in the user file and to `/etc/ssh` in the system file. A user file that its group or others can write is refused (`Bad owner or permissions on <file>`), as OpenSSH does. Directives fall in three classes:

- Applied: `HostName`, `User`, `Port`, `IdentityFile`, `IdentitiesOnly`, `IdentityAgent`, `UserKnownHostsFile`, `GlobalKnownHostsFile`, `HostKeyAlias`, `StrictHostKeyChecking`, `ProxyJump`, `ConnectTimeout`, `ServerAliveInterval`, `ServerAliveCountMax` and `Include`.
- Ignored, since they change neither the destination nor the authentication: `SendEnv`, `SetEnv`, `ForwardAgent`, `ForwardX11`, `ForwardX11Trusted`, every `GSSAPI*`, `Compression`, `LogLevel`, `HashKnownHosts`, `AddKeysToAgent`, `UseKeychain`, `ControlMaster`, `ControlPath`, `ControlPersist`, `LocalForward`, `RemoteForward`, `DynamicForward`, `VisualHostKey`, `UpdateHostKeys`, `BatchMode`, `PasswordAuthentication`, `KbdInteractiveAuthentication`, `ChallengeResponseAuthentication`, `TCPKeepAlive`, `CheckHostIP` and `PubkeyAuthentication yes` (the client never prompts, never uses a password and always refuses an unknown host).
- Refused: any other directive that applies to the host, such as `ProxyCommand`, `CertificateFile`, `PKCS11Provider`, `SecurityKeyProvider`, `Ciphers`, `KexAlgorithms`, `HostKeyAlgorithms`, `MACs` and `PubkeyAcceptedAlgorithms`, and `PubkeyAuthentication no`. So is any `Match` line, wherever it appears, except `Match all` and `Match final all`, which apply to every host.

The client negotiates only modern algorithms. The algorithm lists a system file sets (`Ciphers`, `KexAlgorithms`, `MACs`, `HostKeyAlgorithms`, `PubkeyAcceptedAlgorithms`, `CASignatureAlgorithms`, `GSSAPIKexAlgorithms`, `HostbasedAcceptedAlgorithms`, `RequiredRSASize`) are ignored in `/etc/ssh/ssh_config`, so a crypto policy such as Fedora's or RHEL's does not stop it. The same directives in a user file are refused, except `GSSAPIKexAlgorithms`, which is ignored like every `GSSAPI*`. A refusal names the file, the directive and the host:

```text
/home/me/.ssh/config: ProxyCommand for host gpu is not supported by the built-in SSH client: use ssh_client = "openssh", or a host entry without it
```

Authentication tries, in order, the agent's keys (`SSH_AUTH_SOCK`, or the socket `IdentityAgent` names; `IdentityAgent none` turns the agent off), then the `IdentityFile` keys, which default to `~/.ssh/id_ed25519` and `~/.ssh/id_ecdsa`. With `IdentitiesOnly yes`, as with OpenSSH, the agent offers only the keys of the `IdentityFile`s (it finds the public key in `<file>.pub`, or in the key file itself, even an encrypted one), so a passphrase key loaded in the agent still works. It never asks for a passphrase: an encrypted key file the agent does not hold is skipped, and so is an RSA key, which the client does not support. Like OpenSSH, it skips a key file that its group or others may read (`Permissions 0644 for '<path>' are too open`). The agent's RSA keys and certificates are passed over without a note. When no key is accepted, the error lists what was skipped, for example `key /home/me/.ssh/id_ed25519 is encrypted: add it to ssh-agent` or `key /home/me/.ssh/id_rsa is RSA, which the built-in SSH client does not support: use an ed25519 key, or ssh_client = "openssh"`.

The host key is checked against `UserKnownHostsFile` and `GlobalKnownHostsFile` (by default `~/.ssh/known_hosts`, `~/.ssh/known_hosts2` and `/etc/ssh/ssh_known_hosts`), with hashed entries, `[host]:port` entries and `HostKeyAlias`. A host that is not there is always refused, whatever `StrictHostKeyChecking` says: the message asks you to add it with `ssh-keyscan` or a first connection with `ssh`. A `@revoked` key is refused. A `@cert-authority` line is not supported, and refuses the connection when it is the only match.

`ProxyJump` works with a comma-separated chain: each hop is resolved through the same files and its host key is checked the same way. `ProxyJump none` is respected. The client keeps one session per target. For an `ssh` target, a keepalive goes every 15 seconds with 3 missed answers allowed, and the connect timeout is 30 seconds; `ServerAliveInterval`, `ServerAliveCountMax` and `ConnectTimeout` change them. A Runpod pod always uses these values.

## The Hugging Face token

`OVERBRAINER_HF_TOKEN` (a literal or a `vault:` reference) is needed for gated or private base models, for the deprecated `hub_model_id`, which pushes the adapter (not the merged model) to a private Hub repository, and for [`overbrainer push`](#push-to-hugging-face), which needs a token with `write` access. `push` resolves it only when it pushes (not for `--dry-run`), and never prints it.

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
| `hub_model_id` | none | Deprecated: use `[hub] repo`. Axolotl pushes the adapter to this private Hub repository. `overbrainer migrate` moves it. |

overbrainer also sets `attn_implementation: sdpa` (no extra package needed), `gradient_checkpointing: true`, `warmup_ratio: 0.1`, `logging_steps: 1` and `save_total_limit: 2`: Axolotl keeps the two newest checkpoints and removes older ones, so saves do not pile up on the target's disk. A snapshot is always the newest checkpoint, so it is kept. Set `save_total_limit` in `[training.axolotl_extra]` to keep more.

`[training.axolotl_extra]` is merged into the generated YAML last. Tables merge key by key, and any other value replaces the generated one, so it can change these too: for example `attn_implementation = "flash_attention_2"` on an image with flash-attn installed, or `chat_template = "qwen3"`. An `eval_steps` or `save_steps` there replaces `evals_per_epoch` or `saves_per_epoch`. It cannot set a key that has a typed setting above (use the setting), nor the keys overbrainer manages: `datasets`, `test_datasets`, `val_set_size`, `output_dir`, `dataset_prepared_path`, `plugins`, `resume_from_checkpoint`. `save_only_model = true` is refused too: a checkpoint without its optimizer state cannot be resumed after a snapshot.

Values set through the environment (`OVERBRAINER_TRAINING__AXOLOTL_EXTRA__WARMUP_STEPS=10`) arrive as text. overbrainer turns `true`, `false`, integers and decimal numbers into booleans and numbers, and leaves anything else as text. A number-like value that must stay text belongs in `overbrainer.toml`.
