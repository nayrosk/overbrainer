# The dataset pipeline

overbrainer builds the training set in four stages. Each one reads the previous stage's file in `data/` and can be run on its own:

| Command | Reads | Writes |
|---|---|---|
| `overbrainer subtopics` | `overbrainer.toml` | `data/subtopics.jsonl` |
| `overbrainer questions` | `data/subtopics.jsonl` | `data/questions.jsonl` |
| `overbrainer answers` | `data/questions.jsonl` | `data/answers.jsonl` |
| `overbrainer split` | `data/answers.jsonl` | `data/train.jsonl`, `data/eval.jsonl` |

`overbrainer run` chains the four, then trains when `overbrainer.toml` has a `[training]` section; without one it stops after `split`. `questions` generates the missing subtopics first.

Options:

- `--topic NAME` processes one topic only.
- `--force`, accepted by every stage command except `split`, throws away the stage's output for the selected topics and generates it again. `split` has no `--force`: it always rewrites its files.

## Resuming

Every stage resumes. Items already on disk are skipped, and each new item is appended as soon as it is ready, so an interruption only loses the requests in flight. A topic left with fewer subtopics than configured, for example after a crash mid-topic, is resumed too: `subtopics` only asks for the missing count. `answers` counts a question as done once it has any answer, including one excluded from training; only `--force` asks the parent again for it.

## Output and errors

Each stage prints one summary line on stdout, with token counts and the cost when the provider lists model prices (OpenRouter and NanoGPT do; otherwise the cost is shown as unknown). Prices come from the provider's `/models` listing, read once per provider per run and capped at 10 seconds. A slow, missing or unpriced listing leaves the cost unknown. Progress goes to stderr.

A stage that could not process some items exits with an error after writing everything else it produced; run it again to retry them. A fatal error, such as a rejected API key, stops the stage early. It still prints its summary line with the tokens and cost spent so far, keeps the answers already received, and exits non-zero.

## What each stage does

- `subtopics` asks the `generator` role for `subtopics` subtopic names per topic.
- `questions` asks the `generator` for questions in batches of `question_batch_size`, listing the questions already accepted for the subtopic. Near-duplicates are dropped, first by word overlap (`dedup_threshold`), then, when `roles.embedder` is set, by embedding similarity (`embedding_threshold`). A subtopic stops at `questions_per_subtopic`, or after `max_retries` batches that bring nothing new.
- `answers` sends each question to the `parent` role, `concurrency` requests at a time, with the answer system prompt. An answer that hit the token limit, was refused, is empty, or (with `reasoning = true`) has no raw reasoning is kept in `answers.jsonl` with `meta.excluded` set, and left out of training.
- `split` writes the usable answers of every topic to `train.jsonl` and `eval.jsonl`; `--topic` only limits what is counted in the printed report.

`split` uses only answers whose topic is still configured in `overbrainer.toml` and whose question is still in `questions.jsonl`. The others, left by a removed topic or by questions regenerated with `questions --force`, stay in `answers.jsonl` and are reported as orphaned:

```
split: 90 train, 10 eval, 3 excluded (truncated 3), 2 orphaned
```

The eval set holds `eval_ratio` of all usable answers, rounded. It gets at least one as soon as there are two, and never all of them, so train is never empty. It is spread in proportion to size, first over the topics, then over each topic's subtopics, so every topic gets its share even when its subtopics are small; which small subtopics contribute depends on `seed`. The split is deterministic for a given `seed`. When every usable answer is orphaned (for example with `questions.jsonl` missing), `split` also warns that train and eval are empty.

## Answer format

Answer lines use the Axolotl `chat_template` format, with the parent's reasoning in `reasoning_content`:

```json
{"id":"...","topic":"ownership","subtopic":"Borrowing",
 "messages":[{"role":"user","content":"..."},
             {"role":"assistant","content":"...","reasoning_content":"..."}],
 "meta":{"model":"deepseek/deepseek-r1","input_tokens":41,"output_tokens":812,
         "finish_reason":"stop","reasoning_kind":"raw","excluded":null}}
```

Set `include_system_prompt = true` to also store the system prompt as the first message.

## Reasoning

Only raw reasoning is used for training. A usable example keeps `reasoning_content` only when the parent returned the full reasoning trace; a summary or a redacted trace is dropped even on an otherwise usable example. An excluded example keeps whatever reasoning it got, for inspection.

With `reasoning = true` on the parent:

- `openai` protocol: overbrainer sends `reasoning: {"effort": ...}` (`medium` unless `reasoning_effort` is set) and reads the reasoning from `reasoning`, `reasoning_content`, `reasoning_details`, or a `<think>` block in the answer.
- `anthropic` protocol: overbrainer asks for adaptive thinking (`thinking: {"type": "adaptive"}`), unless the role sets `thinking_budget`.

Claude Sonnet 5, Opus 5, Opus 4.8, Opus 4.7 and Fable 5.x require adaptive thinking and reject a fixed budget, so leave `thinking_budget` unset for them. Claude Opus 4.5, Sonnet 4.5 and Haiku 4.5 do the opposite: they reject adaptive thinking (`400 adaptive thinking is not supported on this model`) and need a fixed budget. Set `roles.<role>.thinking_budget` to a value in `[1024, max_tokens)` to switch that role to `thinking: {"type": "enabled", "budget_tokens": ...}`, which also drops `output_config.effort` from the request (`thinking_budget` cannot combine with `reasoning_effort`).

Some parents never return their raw reasoning, whatever their responses claim: any model on the `anthropic` protocol, and Claude, Gemini and OpenAI models on the `openai` protocol (except the open-weight `gpt-oss` models, whose reasoning is raw). overbrainer stores their reasoning as a summary (`reasoning_kind = "summary"`). With `reasoning = true`, every answer from such a parent is therefore excluded as `no_raw_reasoning`, and overbrainer warns about it at startup. With `reasoning = false`, the answers stay usable, without their reasoning.

## Prompt templates

`init` writes the default prompts to `prompts/`. Edit them to change how questions are generated or how the parent is instructed. They are [minijinja](https://docs.rs/minijinja) templates, and using an undefined variable is an error. Variables:

| Template | Variables |
|---|---|
| `prompts/subtopics.txt` | `topic`, `description` (may be empty), `count` |
| `prompts/questions.txt` | `topic`, `description`, `subtopic`, `count`, `accepted` (list of questions) |
| `prompts/answer_system.txt` | `topic`, `description` |

A missing file falls back to the built-in default.

## Rejected items

A subtopic or question deleted in the [terminal UI](tui.md) is recorded in `data/rejected.jsonl`, so the stages do not generate it again: `subtopics` drops that name, and `questions` treats that text as already asked (its exact text, a case or spacing variant, or a near-duplicate). `--force` keeps these rejections. To allow one again, remove its line from `data/rejected.jsonl`.
