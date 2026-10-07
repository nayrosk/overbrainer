# Case study: a Rust compiler error explainer

Can a small open-weights model explain Rust compiler errors well enough to stand in for a large reasoning model, and what does it cost? This folder holds one full overbrainer run on that question, with three children measured against the same parent. Everything here was produced by overbrainer on 2026-10-07; the numbers are the reports as generated.

A 4B child trained on the parent's final answers wins or ties on about one question in five, answers in a few seconds on one 24 GB GPU, and costs about 6% of the parent per request. Use it as a cheap first line, and keep the parent for hard questions. Training the parent's reasoning into a small child made things worse.

## Setup

| | |
|---|---|
| Task | Explain a Rust compiler error (borrow checker, moves, lifetimes, traits, types, mutability, generics, async and Send, modules) and show the fix |
| Topics | 9 topics, 4 subtopics each, 55 questions per subtopic |
| Questions | 1,965 generated; 1,746 for training and 194 held out for eval after the split (24 truncated answers and 1 empty answer left out) |
| Generator | `deepseek/deepseek-v4-flash` |
| Parent | `deepseek/deepseek-v4-pro:thinking` (MIT license), with its raw reasoning |
| Children | `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` (Apache-2.0), QLoRA, 3 epochs |
| Hardware | Runpod, `gpu_types = "auto"`: an NVIDIA A40 for training, an A40 or RTX A5000 for the compares |
| Judge | `qwen3.5:9b` on a local Ollama, with `reasoning_effort = "none"`, pairwise, with a fixed order per question drawn from a seed (child first on about half) |

[`overbrainer.toml`](overbrainer.toml) is the config of the last run. Keys, URLs and hosts come from `OVERBRAINER_*` variables, as usual. The judge is a local Ollama, called `local_ollama` in the config and `xana` in the reports.

## Results

All three children answered the same 194 held-out questions with the same judge.

| Child | Trained on | Wins or ties | Cut at the token limit | Latency p50 / p95 | Child cost per 1,000 requests | Report |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | answers and reasoning | 16.0% | 93 of 194 | 16.4 s / 26.0 s | $2.28 (30% of the parent) | [compare-1.7b-with-reasoning.md](compare-1.7b-with-reasoning.md) |
| Qwen3-1.7B | answers only | 12.0% | 8 | 3.7 s / 10.5 s | $0.31 (4.1%) | [compare-1.7b-answers-only.md](compare-1.7b-answers-only.md) |
| Qwen3-4B | answers only | 20.2% | 2 | 5.6 s / 9.7 s | $0.46 (6.0%) | [compare-4b-answers-only.md](compare-4b-answers-only.md) |

The parent costs $7.62 per 1,000 requests at its list price, from the tokens it was billed for each answer. The child cost is the pod's hourly price times the mean time per answer, with one request at a time, so it is an upper bound.

The 4B model is on Hugging Face: [Nayrosk/rust-errors-qwen3-4b](https://huggingface.co/Nayrosk/rust-errors-qwen3-4b), with the GGUF for Ollama and llama.cpp.

## What we learned

- Do not train a small child on long reasoning. The parent thinks for about 2,750 tokens per answer, and up to 16,000. A 1.7B child trained on that learned to think at length without learning to conclude. That run used `[compare] max_tokens = 6144` and `sequence_len = 8192`: 89 of its 194 answers were reasoning cut at that limit, so the user saw an empty answer. It was also the slowest and the most expensive child.
- Training on final answers only fixed the format at once: 8 cut answers instead of 93, and a p50 latency of 3.7 s instead of 16.4 s. The config does this with `[training.axolotl_extra] chat_template = "chatml"`, a template that leaves the reasoning out.
- Size matters more than format for quality. On final answers, going from 1.7B to 4B raised the win or tie rate from 12% to 20%, and the final eval loss of training from 1.29 to 1.10, for about 50% more child cost.
- The questions are hard. The generator wrote comparison questions ("compare the diagnostics of `tokio::spawn` and `spawn_local`...") that a frontier model answers well and a small one gets subtly wrong, for example calling undefined behaviour a panic. A narrower question set (one error, one fix) would likely score higher; this run did not test that.

## Limits

- The judge is small. A 9B local judge is lenient compared with a large one. An earlier judging of the run with reasoning by Kimi K2.6 gave the child 0.5% of wins or ties, where the local judge gives 16%. Read the absolute rates with care, and compare the children with each other rather than with the parent.
- One run each. No seeds were repeated, and 194 questions give a margin of several points on each rate.
- Serial requests. Latency is that of one user, and the child cost is an upper bound: a server under load answers several requests at once.
- Pairwise against a strong parent. The rate measures how often the child is as good as the parent. A correct answer can still lose.

## Cost and time

| Stage | Time | Cost |
|---|---|---|
| Subtopics and questions | 6 min | $0.13 of API |
| Parent answers (1,965) | about 4 h at 2 requests at a time | $14.15 at list price, covered here by a NanoGPT subscription |
| Train Qwen3-1.7B with reasoning (A40) | 1 h 55 min | $0.95 |
| Train Qwen3-1.7B answers only (A40) | 53 min | $0.44 |
| Train Qwen3-4B answers only (A40) | 1 h 39 min | $0.81 |
| Three compares on Runpod (generation, pod time) | 17 to 58 min each | $0.66 in all |
| Judge | about 25 min for the three, on a local GPU | $0 |

Runpod in all, including a test run and two failed starts: about $3.10.

## Reproduce

The config needs overbrainer 0.8.0 or later (for `reasoning_effort = "none"` on the local judge) built with the `builtin-ssh` feature (it sets `ssh_client = "builtin"`). Until 0.8.0 is out, install from git:

```sh
cargo install --locked --git https://github.com/nayrosk/overbrainer --features builtin-ssh overbrainer
overbrainer init rust-errors && cd rust-errors   # writes overbrainer.toml, .env.example, .gitignore and the default prompts
cp /path/to/examples/rust-errors/overbrainer.toml .   # replaces the example config
cp .env.example .env
```

In `.env`, set `OVERBRAINER_PROVIDERS__NANOGPT__BASE_URL`, `OVERBRAINER_PROVIDERS__NANOGPT__API_KEY`, `OVERBRAINER_PROVIDERS__LOCAL_OLLAMA__BASE_URL` and `OVERBRAINER_RUNPOD__API_KEY`, then:

```sh
overbrainer run             # subtopics, questions, answers, split, then training with export
overbrainer compare         # measure the child against the parent
```

With a hosted judge instead, drop `reasoning_effort` and set `judge` to a model from a different family than the parent.
