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

## Project state

overbrainer keeps its own state in `.overbrainer/`, next to `overbrainer.toml` (`init` adds it to `.gitignore`).

- `version`: the project format, one integer line (`1` since 0.4.0), written by `init`, the TUI init wizard and `migrate`. A project without it predates 0.4.0: see [migrating a project](#migrating-a-project).

- `history.jsonl`: one line per stage execution, appended when the stage ends, whether it succeeded (`ok`), had failed items or was stopped by a provider error (`failed`), or was stopped by Ctrl-C or the TUI (`interrupted`). Each line holds the stage, start and end times (UTC), provider and model, the counts, tokens and cost (`null` when the price is unknown). `split` lines hold the train, eval and orphaned counts instead of a model. The file is never rewritten; deleting it resets the totals. The counts of an interrupted line come from progress events: they can miss the last items, and for `answers` its `done` also counts answers excluded from training. Lines rebuilt by `migrate` carry `"backfilled": true`.
- `logs-<YYYYMMDDTHHMMSSZ>.log`: written by the [TUI](tui.md)'s `x` key in the Logs view, one such file per export; nothing else creates or reads these.
- the lock is held on the project directory itself: the commands that write to the project (`tui`, the stages, `run`, `train`, `pod rm`, `migrate`) hold it while they run, so a second one stops at once with `another overbrainer (pid N) is using this project`. Reading commands (`history`, `runs ls`, `runs logs`, `pod ls`, `config check`) never take it. `.overbrainer/lock` only holds the holding process's PID, for that message; removing it, or renaming `.overbrainer/`, changes nothing. The lock goes away with the process, even after a crash. It guards against a second overbrainer process, not against other programs changing the project. A project whose directory cannot be locked or whose `.overbrainer/` cannot be created (a read-only copy, a filesystem without file locks) refuses every command that writes, the TUI included.

`overbrainer history` prints the totals per stage and overall; `overbrainer history --all` prints every execution, a backfilled one ending with `(backfilled)`.

### Migrating a project

Every command on a project without `.overbrainer/version` (one made before 0.4.0) ends with one line on stderr, `this project predates overbrainer 0.4.0: run overbrainer migrate`; the TUI shows it on its status line on start. It never stops the command. `init`, `migrate` and `skill` never say it.

`overbrainer migrate` takes the project lock and:

- adds `/.overbrainer/` to `.gitignore` unless a line already ignores the directory (`.overbrainer`, `.overbrainer/`, `/.overbrainer` or `/.overbrainer/`), appending, never rewriting the file;
- rebuilds the `answers` history from `data/answers.jsonl`: one `ok` line per model, whose `done` counts its answers kept for training, `excluded` the others, and the tokens are summed; all of them are written at once; the cost and provider are unknown, and both times are the file's modification time. It is skipped, saying why, when `history.jsonl` already has an `answers` line recorded by a stage, so nothing is counted twice; after a migration that stopped part way, only the models not yet backfilled are. Subtopics and questions keep no tokens, so they are not rebuilt;
- writes `.overbrainer/version`, last.

It prints one line per change, or `nothing to migrate`: running it again changes nothing. `--dry-run` prints the same lines starting with `would`, and changes no project file; it still takes the lock, which creates `.overbrainer/lock`. A project whose format is newer than the one the running overbrainer knows is refused: upgrade overbrainer.

## What each stage does

- `subtopics` asks the `generator` role for `subtopics` subtopic names per topic.
- `questions` asks the `generator` for questions in batches of `question_batch_size`, listing the questions already accepted for the subtopic. Near-duplicates are dropped, first by word overlap (`dedup_threshold`), then, when `roles.embedder` is set, by embedding similarity (`embedding_threshold`). A subtopic stops at `questions_per_subtopic`, or after `max_retries` batches that bring nothing new. Up to `concurrency` subtopics are filled at a time, across topics; the batches of one subtopic run one after the other, and each topic's deduplicator admits one batch at a time, so parallel subtopics never keep the same question twice.
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
