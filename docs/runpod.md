# Training on Runpod

A `runpod` target creates a pod for each run on [Runpod](https://runpod.io?ref=ym24z23f) (referral link) Secure Cloud. It runs the job on the pod over SSH with the image's Axolotl, retrieves the results, then deletes the pod.

```toml
[targets.gpu_cloud]
kind = "runpod"
gpu_types = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA A40"]  # tried in order
max_hours = 6
```

It needs `OVERBRAINER_RUNPOD__API_KEY` (a literal or a `vault:` reference), resolved only when a Runpod command runs, and `ssh-keygen` next to `ssh` on this machine. A restricted API key needs read and write access to pods and to the account's secrets (see [SSH access](#ssh-access)); without the latter, `train` stops before creating any pod and says so.

| Key | Default | Meaning |
|---|---|---|
| `gpu_types` | required | Runpod GPU type IDs, tried in order until one can be placed, or `"auto"` to try every GPU type in stock, cheapest first, after up to 3 cheaper ones reported out of stock (see [`"auto"`](#auto) below). From the environment, one comma-separated value, or `auto`: `OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES="NVIDIA GeForce RTX 4090,NVIDIA A40"` or `OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES=auto`. |
| `min_vram_gb` | none | Least VRAM per GPU, in GB. Only with `gpu_types = "auto"`. At least 1. Unset, `auto` uses the [VRAM estimate](#vram-estimate) instead, when it can be made. |
| `max_price_per_hour` | none | Highest Secure Cloud list price of one GPU, in USD per hour. Only with `gpu_types = "auto"`. Greater than 0. |
| `max_hours` | required | The pod's watchdog deletes the pod this long after it was created, unless overbrainer is following a job that still makes progress ([the watchdog](#the-watchdog)). At most 720. |
| `max_cost_usd` | none | Most a run may spend on its pod, in USD, at the pod's hourly rate: at 95% the job is stopped with a snapshot, at 100% the watchdog deletes the pod ([automatic snapshots](#automatic-snapshots)). Greater than 0, and worth at least 30 minutes of the pod (a warning says so otherwise). Not applied with `--keep-pod`. |
| `gpu_count` | `1` | GPUs per pod. |
| `image` | `axolotlai/axolotl-cloud-term:0.19.0-py3.12-cu130-2.12.1`, pinned by digest | Pod image (CUDA 13, driver 580 or newer). |
| `venv` | `/workspace/axolotl-venv` | Virtual environment holding `bin/axolotl` on the pod. Absolute. |
| `container_disk_gb` | `50` | Container disk, at least 20. |
| `boot_grace_minutes` | `30` | The watchdog deletes a pod whose job never started after this. At least 5. |
| `retrieve_grace_minutes` | `60` | The watchdog deletes a pod whose ended job was not retrieved after this. |
| `data_center_ids` | any | Data centers the pod may be placed in, for example `["EU-RO-1"]`, or `"auto"` for those with a chosen GPU type in stock when the run starts (see [`"auto"`](#auto) below). Cannot be `"auto"` together with `network_volume_id`: list the volume's data center instead. |
| `network_volume_id` | none | Network volume mounted at `/workspace/data`; runs and the Hugging Face cache then live on it. Needs exactly one `data_center_ids` entry, the volume's data center, never `"auto"`. overbrainer never deletes anything on it. |
| `max_volume_gb` | none | Largest size, in GB, overbrainer may grow the network volume to when the run fills it ([disk space](#disk-space)). A grow is permanent: a volume cannot shrink, it is billed per GB each month after the pod is gone, and every pod sharing it sees the new size. At least 1, and only with `network_volume_id`. Without it, a full disk stops the job with a snapshot. |

`gpu_type` became `gpu_types`: a configuration with the old key is rejected as `unknown field`.

## The catalog

`overbrainer pod gpus`, `overbrainer pod datacenters`, `overbrainer pod volumes` and `overbrainer pod templates` read the Runpod catalog and account directly; none of them needs a `[training]` section, only `OVERBRAINER_RUNPOD__API_KEY`.

`overbrainer pod gpus` lists the Secure Cloud GPU types, cheapest first (ties by more VRAM, then by ID); a GPU type offered on the Community Cloud only is left out. The listing is for one GPU per pod, since the command takes no `gpu_count`.

```
$ overbrainer pod gpus
ID                       VRAM GB   $/H  MAX COUNT  STOCK
NVIDIA GeForce RTX 4090       24  0.34          8  HIGH
NVIDIA A40                    48  0.40          8  MEDIUM
NVIDIA H100 80GB HBM3         80     -          8  LOW
```

`--min-vram GB`, `--max-price PRICE` (USD/hour) and `--in-stock` narrow the list. `--data-center ID` keeps only the GPU types offered there and shows their stock there, in the STOCK column, instead of the overall band. `-` in `$/H` means Runpod lists no positive Secure Cloud price for that GPU type.

`overbrainer pod datacenters` lists every data center, by ID, with how many GPU types are in stock there overall:

```
$ overbrainer pod datacenters
ID       NAME          REGION         GPU TYPES IN STOCK
EU-RO-1  EU Romania 1  EUROPE         14
US-KS-2  US Kansas 2   NORTH_AMERICA  9
```

`overbrainer pod volumes` lists the account's network volumes, by name:

```
$ overbrainer pod volumes
ID        NAME           SIZE GB  DATA CENTER
nv1a2b3c  training-data      500  EU-RO-1
```

`overbrainer pod templates` lists the account's pod templates, by name (serverless templates are left out; a template ID the API lists twice shows once, its first occurrence):

```
$ overbrainer pod templates
ID        NAME            IMAGE
tp9z8y7x  axolotl-custom  myrepo/axolotl:0.20.0
```

An empty listing prints one line saying so instead of a table (`pod: no GPU type matches`, `pod: no data center found`, `pod: no network volume on this account`, `pod: no pod template on this account`).

The IDs, names, regions, data centers, images and CUDA versions of the catalog (GPU types, data centers, network volumes, templates) are cleaned as soon as they are read from the Runpod API: escape sequences are dropped, a newline, tab or other whitespace control character becomes a space, and any other control character (C0, DEL, C1) is dropped. Every later use (the `overbrainer pod` tables, the pickers, the start confirmation, the `"auto"` errors, a GPU type or data center written to `overbrainer.toml` from a picker) gets the clean text, so an odd name cannot move the cursor or change the terminal's state. `overbrainer pod ls` keeps only printable ASCII in the pod fields it prints.

## `"auto"`

`gpu_types = "auto"` and `data_center_ids = "auto"` are resolved from the same GPU listing (scoped to the target's `gpu_count`), read right before the run's first create call, never at `overbrainer pod gpus` time:

1. `gpu_types = "auto"` picks every Secure Cloud GPU type in stock for `gpu_count` GPUs, within `min_vram_gb` (or, when it is unset, the [VRAM estimate](#vram-estimate)) and `max_price_per_hour` when set, and in one of the listed `data_center_ids` when any are listed, cheapest first (ties by more VRAM, then by ID).
2. Runpod's stock reports change within seconds, and a type reported out of stock can still be placed. So up to 3 GPU types reported out of stock that are cheaper than the cheapest type in stock, and that meet every other limit (`gpu_count`, `min_vram_gb` or the estimate, `max_price_per_hour`, and the listed `data_center_ids`: a type that names none of them is left out, one that names no data center at all is kept), are tried first, cheapest first. A refusal costs one API call.
3. `data_center_ids = "auto"` then picks every data center with one of the chosen GPU types in stock for `gpu_count`, ordered by the cheapest such GPU type it has. Runpod names no data center for a type out of stock everywhere, so such a type is created without a data center list under `"auto"`; listed `data_center_ids` (and so a network volume's data center) still bind it.

With `gpu_types = "auto"`, each GPU type that cannot be placed makes overbrainer read the catalog again: the types not tried yet are sorted again by the same rules, so a cheaper type back in stock comes next, and `data_center_ids = "auto"` is picked again from the types left. A type is never tried twice in a run's walk, and the 3 types out of stock are counted over the whole walk. When that read fails, or reports nothing in stock left to try, the types left keep their order and a warning says so. The picks and each new order are logged at info level. A list of GPU types is tried as it is, without these reads.

When nothing in stock matches, the run fails before any pod is created, without trying a type out of stock, naming what was asked:

```
no GPU type in stock on Runpod's Secure Cloud for gpu_types = "auto" (gpu_count = 2, min_vram_gb = 48, max_price_per_hour = 1.5)
```

or, once GPU types are chosen but no data center has one of them in stock:

```
no data center has A or B in stock for data_center_ids = "auto" (gpu_count = 1)
```

## VRAM estimate

overbrainer estimates the GPU memory a training run needs on each GPU, from the child model's shape on Hugging Face (its parameter count from the model's `safetensors` metadata, its hidden size, layers and vocabulary from `config.json`, read with `OVERBRAINER_HF_TOKEN` when set, for gated models) and the `[training]` settings. `micro_batch_size`, `sequence_len`, `optimizer` and `gradient_checkpointing` are read from `training.axolotl_extra` first when it sets them, as Axolotl gets them. The estimate adds:

- the weights: 2 bytes per parameter (bf16), about 0.6 with `adapter = "qlora"` (4 bits);
- the gradients and optimizer state: 16 bytes per trained parameter (10 with an 8-bit `optimizer`, such as `paged_adamw_8bit`); every parameter with `adapter = "full"`, only the adapter's (about `18 x lora_r x hidden size` per layer) with LoRA and QLoRA;
- the activations kept for the backward pass, `micro_batch_size x sequence_len x hidden size x layers x 2` bytes with gradient checkpointing (on unless `axolotl_extra` sets `gradient_checkpointing = false`, then about 17 times that), and the fp32 logits, `micro_batch_size x sequence_len x vocabulary x 4` bytes;
- 2 GB of overhead, then 20% on top of it all.

Each GPU holds a whole copy of the model, so the estimate does not change with `gpu_count`. A run whose `axolotl_extra` shards the model instead (`deepspeed`, `fsdp` or `fsdp_config`) gets no estimate: its fit is `?` and `auto` has no floor. The estimate is rough and on the safe side, and it counts a GB as 10^9 bytes: the Runpod catalog gives the vendors' figures, and a "48 GB" A40 leaves about 45 GiB to CUDA, so a GPU is judged by the smaller unit.

`overbrainer train` estimates only when `gpu_types = "auto"` and `min_vram_gb` is unset; `auto` then keeps only the GPU types with at least that much VRAM (rounded up to a whole GB). An explicit `min_vram_gb` always wins, and a list of GPU types is tried as it is. When the shape cannot be read (a local path as `base_model`, a model without `safetensors` metadata, a gated model without a token, no network), the run starts as before, without that floor, and a warning says why. A run started from the TUI's confirmation uses the estimate that confirmation showed, without asking Hugging Face again; it also warns there about a listed GPU type with less VRAM than the estimate.

## The pickers

The [TUI](tui.md)'s Project view lets you set a Runpod target's `gpu_types`, `data_center_ids`, `network_volume_id` and `image` from the live catalog instead of typing them, and the confirmation before a run starts lets you re-pick the GPU types and data centers it will use, saving the choice to `overbrainer.toml` first. The GPU picker has a FIT column from the [VRAM estimate](#vram-estimate): `ok`, `tight` (less than 10% to spare), `small` (less VRAM than the estimate: shown dim and cannot be chosen) or `?` (no estimate, or a target other than `training.target`); its title says the estimate. `o` sorts the GPU picker by price, VRAM (most first) or the number of data centers with the type in stock (most first), and the data center picker by ID or region. With `network_volume_id` set to a volume the volume listing has, `data_center_ids` holds the volume's data center: changing it in the Project view is refused, pick another volume instead. See [Terminal UI](tui.md) for the keys.

`container_disk_gb` is checked against fixed bounds (at least 20): the Runpod v2 API does not say how much container disk a GPU type allows.

## What happens to a pod

### Creation

Before every create call, `runs/<run-id>/pod.json` records it, so a pod created by a client that dies right after is still found by `overbrainer pod ls` and `pod rm` (every pod carries its run ID in its environment). When Runpod reports a GPU type as a capacity failure, or refuses it with 403, overbrainer tries the next one. A 402 (no credits), a 422, and any other 400 (overbrainer's own request is wrong) stop at once instead of repeating for every type. A create that gets no clear answer is not sent again blindly: overbrainer first looks for the pod by its run ID.

### SSH access

Each run gets its own client key and its own pod host key in `runs/<run-id>/ssh/`. The pod's host key is generated here and pinned in `runs/<run-id>/ssh/known_hosts`; the image's own host keys are never trusted. Its private half goes to Runpod as an account secret named `overbrainer_host_key_<run-id>`, created before the first create call: the pod's environment only holds the reference `{{ RUNPOD_SECRET_overbrainer_host_key_<run-id> }}`, which Runpod replaces with the key when the container starts, so the pod details the API returns never show the key. The secret stays while a pod of the run may exist (a restarted container runs its bootstrap again and needs it), and is deleted with the run's last pod, stray pods included: while a stray of the run is left, the secret and the private client key stay until `pod rm` or the next `train` finds it gone. Otherwise they go at the end of the run, after a failed or interrupted start, at `max_hours` or `max_cost_usd` when overbrainer deletes the pod, by `pod rm`, and once a pod deleted by its watchdog is found gone. A secret left over by a crash is cleaned up by the next `train` (see [Stray pods and `pod rm`](#stray-pods-and-pod-rm)). overbrainer connects with `ssh -F runs/<run-id>/ssh/config`, which ignores `~/.ssh/config` and the agent. Neither `ssh/` nor `pod.json` is ever uploaded with the run directory. The pod is ready once SSH answers with that key, usually a few minutes after creation (the image is 8.5 GB). After 15 minutes it is deleted and the next GPU type tried.

### The watchdog

The pod's command starts a small shell watchdog as its first process. At startup it checks that the pod's own Runpod key can read the pod, and overbrainer refuses to train (and deletes the pod) when it cannot. The watchdog then deletes the pod:

- at `max_hours`, unless overbrainer is following the job (see below);
- at `max_cost_usd`, whether or not overbrainer follows the job;
- `boot_grace_minutes` after the pod started, if no job ever did;
- `retrieve_grace_minutes` after the job ended, if its results were not retrieved;
- at once, when overbrainer marks the results retrieved.

While overbrainer follows a job (`train`, `train attach` or the TUI stays open) and the job keeps making progress, `max_hours` deletes nothing: overbrainer renews a lease on the pod (`.pod/lease`) every 5 minutes, and the watchdog skips its deadline while that lease is less than 15 minutes old. The lease is only renewed while metrics keep arriving; a job with no new metric for 30 minutes stops renewing it. Once overbrainer stops following (Ctrl-C, closed, crashed, network lost) or the job stalls, the lease runs out after 15 minutes and a pod past its deadline is deleted.

A pod whose bootstrap failed (for example sshd could not start) is deleted at once too, once the watchdog's own proof runs. The watchdog writes its log to `.pod/watchdog.log` in the run directory on the pod, and to the pod's Runpod logs; the bootstrap writes its own to `.pod/bootstrap.log` and to the pod's Runpod logs.

### Pod logs

Runpod keeps a pod's logs (its container's output and its own system lines, such as the image pull) only while the pod exists. overbrainer keeps a copy in `runs/<run-id>/.pod/pod.log` (mode 600), one JSON object per line (`ts`, `source`, `line`):

- while it follows the run (`train`, `train attach`, the TUI), it reads the pod's log stream (`GET /v2/pods/{id}/logs`), starting with the last 5000 lines (which covers the image pull and the bootstrap), then resuming where it stopped (the last event ID is in `.pod/pod.log.cursor`), so a reconnect or a later `train attach` repeats no line; when Runpod no longer accepts that event ID, it starts again once from the event's time;
- before every delete of the run's pod (at the end of the run, at `max_hours`, after a failed start, `pod rm`), it reads what is left for up to 5 seconds;
- once the job ended, it copies `.pod/watchdog.log` and `.pod/bootstrap.log` from the pod next to it (up to 4 MiB each).

Nothing under `.pod/` is written or read through a symbolic link or a file with a second hard link: such a path is refused, whatever the pod sent back.

The copy stops at 20 MiB, with a last line saying so. Every line is cleaned before it is written or shown: the Runpod API key, anything shaped like a Runpod key (`rpa_`, `rps_`) or a Hugging Face token (`hf_`), PEM private keys, the job's own secrets (the Hugging Face token), the value of `NAME=value` when NAME contains KEY, TOKEN, SECRET or PASSWORD (and of `NAME: value`, quoted or not, when NAME ends with one of them), runs of 200 or more base64 characters, and lines made only of 40 or more base64 characters become `***`. Lines are printed without terminal control characters.

`overbrainer runs logs RUN_ID --pod` prints what is kept (the bootstrap's and watchdog's logs, then the pod's), then, while the pod exists, the lines Runpod has after the kept ones. `--source container` or `--source system` keeps one source, `--tail N` the last N lines of each log, and `--follow` keeps printing new lines until Ctrl-C. It only reads: the copy belongs to the command following the run. The TUI shows the same log with `s` in the Logs view.

### Automatic snapshots

Before a pod is lost to a limit, its job is stopped with a snapshot, as `overbrainer train stop` does (see [Training](training.md#stopping-with-a-snapshot)), so the steps done so far can be resumed with `overbrainer train --resume-from RUN_ID`:

- 15 minutes before `max_hours`, when nothing holds the lease: the watchdog asks for the snapshot (reason `deadline`) and holds the deadline off while the checkpoint is saved: 15 minutes in total from the request, or up to 15 minutes past the deadline when the request came later (the lease ran out after the deadline);
- when the run's disk is nearly full (reason `disk`, see [disk space](#disk-space));
- at 95% of `max_cost_usd`, and 15 minutes before 100% at the latest (with a small cap, 5% can be only minutes): once the pod exists, overbrainer writes on it when that is, from the pod's creation time and hourly rate (`.pod/snapshot_at`, and `.pod/cost_cap_at` for 100%). The watchdog asks for the snapshot then (reason `cost`), and so does overbrainer while it follows the job. A run whose cap cannot be handed to the pod (Runpod gave no hourly rate, or the files cannot be written) is refused before its job starts, and its pod deleted.

The log says which of the two applies, with the amounts: `the pod spent $19.00 of max_cost_usd $20.00 (95%)`, or `the pod reaches max_cost_usd $0.05 in 11 min ($0.01 spent so far)` when the snapshot is due 15 minutes before the cap. The watchdog's log says how many minutes are left before it deletes the pod.

Since the job stops 15 minutes before the cap, a cap that buys less than 30 minutes of the pod leaves 15 minutes of training or less, and none below 15 minutes: $0.05 at $0.24/h buys 12 minutes, so the run stops at its first step. Once Runpod gives the pod's hourly rate, overbrainer warns before the job starts, with the least cap that buys 30 minutes; the TUI start dialog does the same on its `max_cost` line at the highest listed rate of the GPU types. The run still starts: raise `max_cost_usd` to train longer.

When overbrainer follows the job, it retrieves the snapshot and deletes the pod as usual. When nothing follows it, the pod stays after the job ended, so `overbrainer train attach RUN_ID` can still collect the snapshot: until the retrieve grace ends, and at most `retrieve_grace_minutes` past the deadline. At 100% of `max_cost_usd` the pod is deleted whatever happens, by the watchdog and by overbrainer while it follows the job: a cost snapshot nobody collected by then is deleted with it. A run whose pod was deleted there fails with `max_cost_usd reached: the pod was deleted before the job ended`.

### Disk space

A run that fills its disk would fail and lose its steps, so while overbrainer follows the job it watches the disk of the run directory, every 10 seconds from the [system sample](training.md), and once a minute for what `du` reads:

- without a network volume, that disk is the container disk (`container_disk_gb`), as `df` reports it;
- on a network volume, `df` reports the whole shared cluster rather than the volume, so overbrainer reads what the volume holds with `du` and weighs it against the volume's size from the API. Runpod refuses writes a little before that size, so only 94% of it counts: a 100 GB volume is full at 94 GB. The system panel still shows `df`.

At 85% used overbrainer warns, then again at 90%, 95% and 100%. At 92% used, or once the free space is less than 1.5 times the size of the newest `checkpoint-*` (the next save would not fit), it acts:

- on a network volume with `max_volume_gb` set, it grows the volume through the API to half again its size, at least 50 GB more, never past `max_volume_gb`. Runpod grows a mounted volume without restarting the pod: writes work again within seconds. When the API does not report the new size within two minutes, the job is stopped with a snapshot. Once the volume is at `max_volume_gb`, or when Runpod refuses the grow, the next full disk stops the job. A grow cannot be undone: the volume keeps its new size, and its monthly bill, after the run, and other pods using it see it too;
- otherwise it stops the job with a snapshot (reason `disk`), as `overbrainer train stop` does. The container disk of a running pod cannot grow without restarting it. A snapshot already asked for (for example at the cost cap) keeps its reason.

A run started by overbrainer before 0.5.0 runs a metrics plugin that cannot save a snapshot, so for it a full disk only warns, once: the volume is still grown within `max_volume_gb`, but the job is never asked for a snapshot, and its pod keeps the watchdog it was created with.

overbrainer reads the volume's size before it asks for a pod, and refuses the run, with no pod created, when Runpod does not give it once its retries are spent. The volume's size is read again every minute, so a volume grown elsewhere (from another pod, or the Runpod console) is weighed against its new size.

The watchdog applies a backstop of its own at 97%, once a minute, so a full disk stops the job with a snapshot even when nothing follows it, while a grow overbrainer makes at 92% comes first. It reads `df` of the run directory, or on a network volume the last `du` of the volume, run in the background so a slow network mount never delays its other rules, against the volume's size: the one overbrainer writes in `.pod/volume_gb` while it follows the job and right after a grow, or else the size when the pod was created. A run started before overbrainer kept the volume's ID in `pod.json` is not measured at all on a network volume. Generated Axolotl configs keep only the two newest checkpoints (`save_total_limit: 2`, see [Training](training.md)), which keeps the disk use of a long run flat.

### Retrieval

The results are downloaded and every file checked against the SHA-256 the pod computed for it; a successful run must also have something in `output/`. Then the pod is deleted and overbrainer waits until Runpod no longer shows it. When the download fails or does not check out, the pod stays until the retrieve grace ends, and `overbrainer train attach RUN_ID` retries.

### Ctrl-C

Before the job exists, the pod is deleted and the run fails as interrupted. Once the job runs, Ctrl-C only stops following it, as on any target; the pod keeps running and the watchdog bounds its cost.

## Keeping a pod

`overbrainer train --keep-pod` keeps the pod with no time limit once its job exists. From then on neither overbrainer nor the watchdog deletes it for any reason, not even at `max_hours`. Until then it is guarded like any other pod: a pod whose bootstrap failed is deleted at once, and one whose job never started is deleted after `boot_grace_minutes`. Once the job starts, `train` warns with its hourly rate. Once the run ends, or is left running after Ctrl-C, it prints the `ssh -F ...` command that reaches it. Only `overbrainer pod rm RUN_ID` removes it.

## Stray pods and `pod rm`

Every `train` on a Runpod target first lists the account's pods and warns about overbrainer pods that nothing will delete: their run ended, is a stray left by an ambiguous create, is not in `runs/`, or has no run marker. It never deletes them itself. A pod named like overbrainer's but without a usable run marker is out of reach of `pod rm`: delete it from the Runpod console.

The same check looks at the account's `overbrainer_host_key_<run-id>` secrets. One whose run has no pod in the list and has ended in this project is deleted. One of a run still in progress is kept, as its pod may be on its way. One of a run that is not in this project's `runs/` is only warned about, like its pods would be: `overbrainer pod rm RUN_ID --force` deletes it if no other checkout owns the run.

`overbrainer pod rm RUN_ID` takes only a run ID, never a pod ID. Without `--force`, it refuses to delete anything for a run absent from this project's `runs/` (another checkout may own it) or a run still starting its pod. For a run in progress with a recorded pod, it deletes every other pod of the run (strays, and any extra pod left by an ambiguous create), keeps the training pod, and still fails, naming what it kept and deleted. `--force` also deletes the training pod and marks the run failed. A run in progress whose pod is not recorded needs `--force` too, since any pod of the run could be the one training.

`runs ls` shows each Runpod run's pod as `pod.json` last recorded it, with its rate or its estimated spend: rate times lifetime. Runpod bills per second, including the image pull.

## Exports

`overbrainer export RUN_ID` on a run of a Runpod target creates a new pod for the export, with the target's GPU settings: the merge of an adapter into its base model is much faster on a GPU, and the image needs CUDA. With `gpu_types = "auto"` and no `min_vram_gb`, the floor is what the merge needs: the base model in bf16, 2 bytes per parameter, plus 2 GB, raised by 20%. The run's `output/` (without its checkpoints, but a stopped run's checkpoint when that is the model) and `axolotl.yaml` are hard-linked into the export's directory, `runs/<run-id>/exports/<export-id>/`, and uploaded with the job to the pod's `<workdir>/<export-id>/`. Once the job ends, `output/gguf/` comes back, is checked against the pod's SHA-256 manifest, and the pod is deleted. The export pod follows the same rules as a training pod: lease, `max_hours`, `max_cost_usd`, `boot_grace_minutes`, `retrieve_grace_minutes`, and `--keep-pod`. Its `pod.json` and SSH keys live in the export's directory, and its pod is named after the export ID, as is the Runpod secret holding its host key (`overbrainer_host_key_<export-id>`): the run's own secret and the export's never share a lifetime. That secret goes with the export pod like a run's goes with its pod: once the export ends, its start fails, or `pod rm <export-id>` deletes the pod, and the startup sweep deletes the secret of an ended export with no pod left. `overbrainer pod ls` shows it under that ID, noted `export of run <run-id>`, and `overbrainer pod rm <export-id>` deletes it. Messages about an export pod never suggest `train attach`: an export is not followed again, it is run again with `overbrainer export <run-id>`. Ctrl-C cancels the export and ends its pod.

Without a network volume, a training run with `[export] after_training = true` and an export pod warn when `container_disk_gb` may be too small for what the export writes beside the run: the base model and the merged copy (bf16) unless the run already merged it, the 16-bit GGUF the quantization starts from, and the quantized GGUF, raised by 10%.

## Custom images

A custom `image` must keep an entrypoint that ends with `exec "$@"`, and provide `bash`, `sshd` (started with `service ssh`), `ssh-keygen`, `base64`, `curl`, `setsid`, `nohup`, `tar`, `find`, `sha256sum` and Axolotl in `venv`. Jobs on the pod start from `/etc/overbrainer/job.env`, which the pod writes with the image's `PATH`, its CUDA library path and `HF_HOME`, since an SSH session does not see the image's environment.

## Troubleshooting

### `the local ssh cannot keep its connection`

overbrainer opens one `ssh` master connection per pod and runs every command through it: `ssh -M -f` authenticates, then leaves a background process holding the connection. When the pod's sshd answers (its `SSH-2.0-` banner is readable) but that background process ends right after it started, twice in a row, the cause is on this machine: overbrainer deletes the pod at once and stops, without trying other GPU types, since every pod would fail the same way. The error ends with the last lines of ssh's own log (`ssh log: ...`), or `empty` when it wrote none.

The usual cause is a wrapper around `ssh` on `PATH` that kills background processes when the command in the foreground exits, for example a firejail symlink (`/usr/local/bin/ssh -> firejail`). Check with `command -v ssh`, then put the real `ssh` first on `PATH` for overbrainer, for example `PATH=/usr/bin:$PATH overbrainer train`.

### `max_hours reached: the pod was deleted before the job ended`

The job needed longer than `max_hours` while nothing followed it (overbrainer was closed or detached, or the job stopped making progress), and the pod was deleted at its deadline. The watchdog checks the deadline once a minute and deletes the pod itself; overbrainer's own guard deletes it 5 minutes later if it is still there and no lease holds it. Either way the run fails with this message and `pod.json` says who deleted the pod (`watchdog` or `client`). Runpod keeps nothing of a deleted pod, its logs included, so this message is the only trace of why it stopped.

While following a job, overbrainer says once when the training pace ends it after the deadline:

```
at this pace the job needs about 20.4h more, past max_hours (2026-09-29T19:54:50Z): the pod stays while overbrainer follows the job; if it stops following, the pod's watchdog deletes the pod
```

Keep overbrainer following the job until it ends, raise `max_hours` above the time the run needs (the TUI shows its ETA), or train with `--keep-pod` and remove the pod with `overbrainer pod rm RUN_ID` once done. When the watchdog's snapshot before the deadline was saved, the run is recorded `stopped` once `train attach` collects it, and `overbrainer train --resume-from RUN_ID` goes on from there; otherwise the run starts over from the beginning.

When the connection to the pod fails for another reason, the warnings name it, for example `cannot reach the job (1/5): ssh failed: the connection was terminated`.
