# Compare

`overbrainer compare` measures what the child is worth against the parent, on the questions of `data/eval.jsonl` that the child never trained on. It reports how often a judge model finds the child's answer as good as the parent's or better, how fast the child answers, and what 1,000 requests cost each.

## What it needs

- A run with a GGUF: `overbrainer export RUN_ID`, or `[export] after_training = true`.
- `data/eval.jsonl`, written by `overbrainer split`.
- A judge: `[roles] judge`, else the parent. A judge other than the parent avoids the parent favoring its own answers.
- For the cost rows, the prices of `[compare]` (see [configuration](configuration.md#compare)).
- The judge prompt `prompts/judge.txt`, which `init` writes (see [prompt templates](pipeline.md#prompt-templates)). A missing file falls back to the built-in default.

## Run it

```bash
overbrainer compare                       # the newest run with a GGUF, every eval question
overbrainer compare --run RUN_ID --limit 50
overbrainer compare --rejudge COMPARE_ID  # judge again, without asking the child again
```

| Option | Meaning |
|---|---|
| `--run RUN_ID` | The run to compare. Default: the newest run with a GGUF. |
| `--limit N` | Ask only the first N questions of `data/eval.jsonl` (N at least 1). Two compares with the same limit use the same questions. |
| `--keep-pod` | Runpod target only: keep the pod once the job ends, with no time limit. `overbrainer pod rm COMPARE_ID` deletes it. |
| `--rejudge COMPARE_ID` | Judge an earlier compare again. It cannot combine with `--limit` or `--keep-pod`. |

stdout is the path of the report, `runs/RUN_ID/compares/COMPARE_ID/compare.md`. Progress goes to stderr. It takes the project lock, like every command that writes.

## How it works

1. The compare runs on the target the run trained on: this machine, the SSH host, or a new Runpod pod. On a local or SSH target, overbrainer checks the run's GGUF there against the SHA-256 in `export.json`, and uploads it only when it is missing or different. On Runpod, the pod is new, so the GGUF is uploaded with the job and deleted, on the pod and here, once the job ends. The pod is deleted unless `--keep-pod` is set.
2. Before a Runpod pod starts, stderr shows its GPU price per hour (the cheapest in-stock price of the catalog, else the cap `max_price_per_hour`) and the number of questions.
3. A job downloads the pinned `llama-server` of llama.cpp, checks its SHA-256, caches it (the cache is shared with `export`) and serves the GGUF on `127.0.0.1`. Nothing listens on another address. Which build it gets depends on the machine:
   - NVIDIA (`nvidia-smi` works): the CUDA 13.4 build. Its CUDA runtime is a separate archive, downloaded and verified like the build, and put first on `LD_LIBRARY_PATH`. When the driver supports CUDA below 13, the CPU build runs instead, with a warning.
   - AMD (Linux with `/dev/kfd`): the ROCm build, which llama.cpp makes for x86-64 only. It needs the system ROCm 7 runtime (`libamdhip64.so.7`, `librocblas.so.5`, `libhipblas.so.3`; on Ubuntu, `apt install libamdhip64-7 librocblas5 libhipblas3`) and read and write access to `/dev/kfd` (your user in the `render` group). Otherwise the CPU build runs, with a warning that says why: the missing libraries and the Ubuntu packages that hold them, no access to `/dev/kfd` (the `render` group), or an arm64 host.
   - macOS on Apple silicon: the Metal build.
   - A Linux host without a usable GPU: the CPU build. Any other platform fails, naming it.
4. The job asks the child every question, one at a time, and records each answer with its time to first token, its total time and its tokens per second. The child gets the system message of the eval record, if it has one, and its user message.
5. Back here, the judge sees each question with both answers, labelled A and B, with the order drawn per question (about half the questions show the child first). It answers `A`, `B` or `tie` with a one-sentence reason. Reasoning blocks are stripped from both answers first. A reply that does not parse is asked again once, then counted as unparsed. A question the child gave no answer to is not judged and counts as a loss.
6. The report sums it up.

Ctrl-C while the job runs cancels it, and deletes the Runpod pod. Ctrl-C while the judge runs stops it and keeps the verdicts so far in `verdicts-<key>.jsonl`; `overbrainer compare --rejudge COMPARE_ID` resumes from there. A rejudge with another judge provider or model, or another `prompts/judge.txt`, starts a new verdicts file and judges every question again.

Questions go one at a time: the numbers describe one user. Throughput under load is not measured.

## The judge

`[roles] judge` takes a provider and a model like the other roles; the parent judges when it is unset. The judge goes through the same provider settings, retries and `[pipeline] concurrency` as the answer stage. It also appears in the role table of the Hugging Face model card (see [push to Hugging Face](training.md#push-to-hugging-face)).

`prompts/judge.txt` is a template with the variables `question`, `answer_a` and `answer_b`. It must ask for a JSON object, `{"verdict": "A" or "B" or "tie", "reason": "one sentence"}`.

## The report

`compare.md` has:

- what was compared: base model, GGUF type and SHA-256, llama.cpp release, hardware, judge, number of questions, date. The hardware is the GPUs of the build that served the child, each model counted (`1 x AMD Radeon RX 7800 XT + 1 x NVIDIA T4`): the NVIDIA GPUs for the CUDA build, the AMD GPUs for the ROCm build, the Apple chip for the Metal build (`Apple M2 Pro (Metal)`). A CPU build shows the CPU, even on a machine with a GPU it could not use. With the ROCm build, an integrated GPU named `AMD Radeon Graphics` is left out when a discrete GPU is listed;
- the summary: win or tie rate (unparsed verdicts left out), wins, ties and losses (child errors count as losses), unparsed verdicts, latency p50 and p95, time to first token p50, output tokens per second, child answers cut at the token limit (`[compare] max_tokens` or the context; the judge sees them as cut), cost per 1,000 requests for the parent and the child, and their ratio;
- the first five losses, then the first five wins, in eval order, with child errors first among the losses. Each shows the child's answer, then the parent's, and the judge's reason. The heading says how many there are (`Losses (first 5 of 18)`);
- the limits that apply: a judge that is the parent, fewer than 100 questions, requests sent one at a time (so the child's cost is an upper bound), a child cost that uses the Runpod pod's price, child answers cut at the token limit, a child that ran on a CPU.

`compare.json` holds the same, and every question with both answers, timings and verdict. Neither file holds a secret.

## Cost

- Parent: the mean tokens the parent was billed for when it answered the eval questions (from the dataset), at `parent_price_in` and `parent_price_out`. Both prices are needed.
- Child: the mean seconds per answer at `child_price_per_hour`. On Runpod, the pod's price is used when it is unset. Requests were sent one at a time, while a server answers several at once under load, so the child's cost is an upper bound.
- Without prices, the rows read `not computed: set [compare] prices`.

## Files

In `runs/RUN_ID/compares/COMPARE_ID/`:

| File | What |
|---|---|
| `setup.json` | What is compared: the GGUF, the questions with the parent's answers, the order seed. A rejudge reads it, not a later `data/eval.jsonl`. |
| `questions.jsonl` | What the child was asked. |
| `compare.sh`, `compare_client.py` | The job's scripts. |
| `child_answers.jsonl` | The child's answers and timings. |
| `server.log` | What `llama-server` printed. Read it when a compare fails. |
| `hardware.json` | The machine that served the child. |
| `verdicts-<key>.jsonl` | The verdicts of one judge and prompt. |
| `compare.json`, `compare.md` | The report. |
| `run.json`, `job.log`, `metrics.jsonl` | The job, as for any run. On Runpod, the pod's record too. |

## In the TUI

The Compare view (`6`) lists the compares, their summary, and each question with both answers and the judge's reason. `C` starts a compare, `J` judges one again, `c` cancels the running one. In the Training view, `C` compares the selected run. See [the TUI page](tui.md#compare-6).
