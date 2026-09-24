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
| `gpu_types` | required | Runpod GPU type IDs, tried in order until one can be placed. From the environment, one comma-separated value: `OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES="NVIDIA GeForce RTX 4090,NVIDIA A40"`. |
| `max_hours` | required | The pod's watchdog deletes the pod this long after it was created, whatever it is doing. At most 720. |
| `gpu_count` | `1` | GPUs per pod. |
| `image` | `axolotlai/axolotl-cloud-term:0.19.0-py3.12-cu130-2.12.1`, pinned by digest | Pod image (CUDA 13, driver 580 or newer). |
| `venv` | `/workspace/axolotl-venv` | Virtual environment holding `bin/axolotl` on the pod. Absolute. |
| `container_disk_gb` | `50` | Container disk, at least 20. |
| `boot_grace_minutes` | `30` | The watchdog deletes a pod whose job never started after this. At least 5. |
| `retrieve_grace_minutes` | `60` | The watchdog deletes a pod whose ended job was not retrieved after this. |
| `data_center_ids` | any | Data centers the pod may be placed in, for example `["EU-RO-1"]`. |
| `network_volume_id` | none | Network volume mounted at `/workspace/data`; runs and the Hugging Face cache then live on it. Needs exactly one `data_center_ids` entry, the volume's data center. overbrainer never deletes anything on it. |

`gpu_type` became `gpu_types`: a configuration with the old key is rejected as `unknown field`.

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
