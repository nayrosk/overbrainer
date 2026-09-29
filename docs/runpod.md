# Training on Runpod

A `runpod` target creates a pod for each run on [Runpod](https://runpod.io?ref=ym24z23f) (referral link) Secure Cloud. It runs the job on the pod over SSH with the image's Axolotl, retrieves the results, then deletes the pod.

```toml
[targets.gpu_cloud]
kind = "runpod"
gpu_types = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA A40"]  # tried in order
max_hours = 6
```

It needs `OVERBRAINER_RUNPOD__API_KEY` (a literal or a `vault:` reference), resolved only when a Runpod command runs, and `ssh-keygen` next to `ssh` on this machine.

| Key | Default | Meaning |
|---|---|---|
| `gpu_types` | required | Runpod GPU type IDs, tried in order until one can be placed, or `"auto"` to try every GPU type in stock, cheapest first, when the run starts (see [`"auto"`](#auto) below). From the environment, one comma-separated value, or `auto`: `OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES="NVIDIA GeForce RTX 4090,NVIDIA A40"` or `OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES=auto`. |
| `min_vram_gb` | none | Least VRAM per GPU, in GB. Only with `gpu_types = "auto"`. At least 1. |
| `max_price_per_hour` | none | Highest Secure Cloud list price of one GPU, in USD per hour. Only with `gpu_types = "auto"`. Greater than 0. |
| `max_hours` | required | The pod's watchdog deletes the pod this long after it was created, whatever it is doing. At most 720. |
| `gpu_count` | `1` | GPUs per pod. |
| `image` | `axolotlai/axolotl-cloud-term:0.19.0-py3.12-cu130-2.12.1`, pinned by digest | Pod image (CUDA 13, driver 580 or newer). |
| `venv` | `/workspace/axolotl-venv` | Virtual environment holding `bin/axolotl` on the pod. Absolute. |
| `container_disk_gb` | `50` | Container disk, at least 20. |
| `boot_grace_minutes` | `30` | The watchdog deletes a pod whose job never started after this. At least 5. |
| `retrieve_grace_minutes` | `60` | The watchdog deletes a pod whose ended job was not retrieved after this. |
| `data_center_ids` | any | Data centers the pod may be placed in, for example `["EU-RO-1"]`, or `"auto"` for those with a chosen GPU type in stock when the run starts (see [`"auto"`](#auto) below). Cannot be `"auto"` together with `network_volume_id`: list the volume's data center instead. |
| `network_volume_id` | none | Network volume mounted at `/workspace/data`; runs and the Hugging Face cache then live on it. Needs exactly one `data_center_ids` entry, the volume's data center, never `"auto"`. overbrainer never deletes anything on it. |

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

`gpu_types = "auto"` and `data_center_ids = "auto"` are resolved once, from the same GPU listing (scoped to the target's `gpu_count`), right before the run's first create call, never at `overbrainer pod gpus` time and never again later in the same run even if several GPU types are tried:

1. `gpu_types = "auto"` picks every Secure Cloud GPU type in stock for `gpu_count` GPUs, within `min_vram_gb` and `max_price_per_hour` when set, and in one of the listed `data_center_ids` when any are listed, cheapest first (ties by more VRAM, then by ID).
2. `data_center_ids = "auto"` then picks every data center with one of the chosen GPU types in stock for `gpu_count`, ordered by the cheapest such GPU type it has.

When nothing in stock matches, the run fails before any pod is created, naming what was asked:

```
no GPU type in stock on Runpod's Secure Cloud for gpu_types = "auto" (gpu_count = 2, min_vram_gb = 48, max_price_per_hour = 1.5)
```

or, once GPU types are chosen but no data center has one of them in stock:

```
no data center has A or B in stock for data_center_ids = "auto" (gpu_count = 1)
```

## The pickers

The [TUI](tui.md)'s Project view lets you set a Runpod target's `gpu_types`, `data_center_ids`, `network_volume_id` and `image` from the live catalog instead of typing them, and the confirmation before a run starts lets you re-pick the GPU types and data centers it will use, saving the choice to `overbrainer.toml` first. `o` sorts the GPU picker by price, VRAM (most first) or the number of data centers with the type in stock (most first), and the data center picker by ID or region. With `network_volume_id` set to a volume the volume listing has, `data_center_ids` holds the volume's data center: changing it in the Project view is refused, pick another volume instead. See [Terminal UI](tui.md) for the keys.

`container_disk_gb` is checked against fixed bounds (at least 20): the Runpod v2 API does not say how much container disk a GPU type allows.

## What happens to a pod

### Creation

Before every create call, `runs/<run-id>/pod.json` records it, so a pod created by a client that dies right after is still found by `overbrainer pod ls` and `pod rm` (every pod carries its run ID in its environment). When Runpod reports a GPU type as a capacity failure, or refuses it with 403, overbrainer tries the next one. A 402 (no credits), a 422, and any other 400 (overbrainer's own request is wrong) stop at once instead of repeating for every type. A create that gets no clear answer is not sent again blindly: overbrainer first looks for the pod by its run ID.

### SSH access

Each run gets its own client key and its own pod host key in `runs/<run-id>/ssh/`. The pod's host key is generated here, sent in the create call's environment (which anyone holding the account API key can read) and pinned in `runs/<run-id>/ssh/known_hosts`; the image's own host keys are never trusted. overbrainer connects with `ssh -F runs/<run-id>/ssh/config`, which ignores `~/.ssh/config` and the agent. Neither `ssh/` nor `pod.json` is ever uploaded with the run directory. The pod is ready once SSH answers with that key, usually a few minutes after creation (the image is 8.5 GB). After 15 minutes it is deleted and the next GPU type tried.

### The watchdog

The pod's command starts a small shell watchdog as its first process. At startup it checks that the pod's own Runpod key can read the pod, and overbrainer refuses to train (and deletes the pod) when it cannot. The watchdog then deletes the pod:

- at `max_hours`;
- `boot_grace_minutes` after the pod started, if no job ever did;
- `retrieve_grace_minutes` after the job ended, if its results were not retrieved;
- at once, when overbrainer marks the results retrieved.

A pod whose bootstrap failed (for example sshd could not start) is deleted at once too, once the watchdog's own proof runs. The watchdog writes its log to `.pod/watchdog.log` in the run directory on the pod, and to the pod's Runpod logs.

### Retrieval

The results are downloaded and every file checked against the SHA-256 the pod computed for it; a successful run must also have something in `output/`. Then the pod is deleted and overbrainer waits until Runpod no longer shows it. When the download fails or does not check out, the pod stays until the retrieve grace ends, and `overbrainer train attach RUN_ID` retries.

### Ctrl-C

Before the job exists, the pod is deleted and the run fails as interrupted. Once the job runs, Ctrl-C only stops following it, as on any target; the pod keeps running and the watchdog bounds its cost.

## Keeping a pod

`overbrainer train --keep-pod` keeps the pod with no time limit once its job exists. From then on neither overbrainer nor the watchdog deletes it for any reason, not even at `max_hours`. Until then it is guarded like any other pod: a pod whose bootstrap failed is deleted at once, and one whose job never started is deleted after `boot_grace_minutes`. Once the job starts, `train` warns with its hourly rate. Once the run ends, or is left running after Ctrl-C, it prints the `ssh -F ...` command that reaches it. Only `overbrainer pod rm RUN_ID` removes it.

## Stray pods and `pod rm`

Every `train` on a Runpod target first lists the account's pods and warns about overbrainer pods that nothing will delete: their run ended, is a stray left by an ambiguous create, is not in `runs/`, or has no run marker. It never deletes them itself. A pod named like overbrainer's but without a usable run marker is out of reach of `pod rm`: delete it from the Runpod console.

`overbrainer pod rm RUN_ID` takes only a run ID, never a pod ID. Without `--force`, it refuses to delete anything for a run absent from this project's `runs/` (another checkout may own it) or a run still starting its pod. For a run in progress with a recorded pod, it deletes every other pod of the run (strays, and any extra pod left by an ambiguous create), keeps the training pod, and still fails, naming what it kept and deleted. `--force` also deletes the training pod and marks the run failed. A run in progress whose pod is not recorded needs `--force` too, since any pod of the run could be the one training.

`runs ls` shows each Runpod run's pod as `pod.json` last recorded it, with its rate or its estimated spend: rate times lifetime. Runpod bills per second, including the image pull.

## Custom images

A custom `image` must keep an entrypoint that ends with `exec "$@"`, and provide `bash`, `sshd` (started with `service ssh`), `ssh-keygen`, `base64`, `curl`, `setsid`, `nohup`, `tar`, `find`, `sha256sum` and Axolotl in `venv`. Jobs on the pod start from `/etc/overbrainer/job.env`, which the pod writes with the image's `PATH`, its CUDA library path and `HF_HOME`, since an SSH session does not see the image's environment.

## Troubleshooting

### `the local ssh cannot keep its connection`

overbrainer opens one `ssh` master connection per pod and runs every command through it: `ssh -M -f` authenticates, then leaves a background process holding the connection. When the pod's sshd answers (its `SSH-2.0-` banner is readable) but that background process ends right after it started, twice in a row, the cause is on this machine: overbrainer deletes the pod at once and stops, without trying other GPU types, since every pod would fail the same way. The error ends with the last lines of ssh's own log (`ssh log: ...`), or `empty` when it wrote none.

The usual cause is a wrapper around `ssh` on `PATH` that kills background processes when the command in the foreground exits, for example a firejail symlink (`/usr/local/bin/ssh -> firejail`). Check with `command -v ssh`, then put the real `ssh` first on `PATH` for overbrainer, for example `PATH=/usr/bin:$PATH overbrainer train`.
