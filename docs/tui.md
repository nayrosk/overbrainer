# The terminal UI

![A tour of overbrainer tui: the dataset tree and an answer's reasoning, the topic stats, a filter, the help, a training run's loss chart, the logs and a dialog](assets/tui-tour.gif)

`overbrainer tui` shows the project in four views, switched with `1` to `4` (or Tab, Shift-Tab): the dataset, the pipeline stages, the training runs, and the logs. It runs the same stages and training flows as the commands. It needs a terminal (it refuses to start when stdout is not one) and is laid out for at least 80x24 characters. While the TUI runs, logs go to its Logs view instead of stderr; `OVERBRAINER_LOG` still sets what is captured.

## Keys

Everywhere:

| Keys | Action |
|---|---|
| `1` `2` `3` `4`, Tab, Shift-Tab | Switch view. |
| `?` | List the keys of the current view. Esc, `?` or `q` closes it. |
| `q`, Ctrl-C | Quit. |
| `R` | Reload the data files and the runs list from disk. |
| `r` | Run a pipeline stage, or `run` (asks which). |
| `y`, then `n`, Esc or Enter | In a dialog: confirm, or cancel (the default, `n`). |

Dataset:

| Keys | Action |
|---|---|
| `k` `j`, Up, Down | Move. |
| `l` `h`, Right, Left, Enter | Expand, collapse. |
| PgUp, PgDn | Scroll the detail pane. |
| `/` | Filter the tree. Enter keeps the filter, Esc clears it. |
| `s` | Stats pane. |
| `e` | Edit the selected question, answer or subtopic in `$EDITOR`. |
| `d` | Delete the selected item, with what depends on it (asks first). |

Training:

| Keys | Action |
|---|---|
| `k` `j`, Up, Down | Select a run. |
| `a` | Attach: follow the selected run again. |
| `c` | Cancel the selected run's job (asks first). On a Runpod run still starting, abandon it instead (asks first). |
| `t` | Start a training run (asks first). |

Logs:

| Keys | Action |
|---|---|
| `k` `j`, Up, Down, PgUp, PgDn | Scroll. |
| `G`, End | Follow the newest lines. |
| `f` | Cycle the level shown: error, warn, info, debug, trace. |

## The views

### Dataset (`1`)

The topics, their subtopics, questions and answers as a tree, with a detail pane: an answer's reasoning and content, which PgUp and PgDn scroll. With `s`, the pane shows the stats of the selected topic and the train and eval sizes. `/` filters the tree by a case-insensitive substring of question texts and subtopic names. Enter keeps the typed filter applied; Esc clears the filter, whether it is being typed or already applied. Questions show `[a]` when answered and used for training, `[x]` when the answer is excluded, `[o]` when orphaned (its topic is no longer configured), and `[ ]` when unanswered. Topics no longer in `overbrainer.toml` and questions whose subtopic is gone are shown too, so they can be deleted.

### Pipeline (`2`)

`r` opens a menu of the stages and `run`, always on every topic and without `--force` (both stay command-line options). The view shows each stage's progress, requests in flight, retries, failures, tokens and cost, and the summary lines the command would print. `run` stops after `split` here: training starts only with `t`.

### Training (`3`)

The runs of `runs/`, and for the selected one its progress, ETA, pod and estimated spend, a loss chart and sparklines of the learning rate and gradient norm.

- `t` starts a run on `training.target` after a confirmation that shows the target, the model and the data. For Runpod it also shows each GPU type with its catalog list price times `gpu_count`, and the most `max_hours` can cost at the highest listed rate, marked "(some prices unknown)" when a price could not be read. `--target` and `--keep-pod` stay command-line options.
- A start holds the data lock until its job begins and is never interrupted before then. Quitting while a Runpod run is still provisioning offers to abandon it instead of waiting.
- `a` follows a run again, and `c` cancels its job after a confirmation. A Runpod run still starting has no job to cancel yet, so `c` offers to abandon that run instead, and the TUI stays open. If its pod is still being prepared, the pod is deleted and the run fails. Once its job is being sent, the run is detached, and `c` then cancels it.
- Leaving the view does not stop following a run: while the TUI is open, its results are still retrieved and its pod deleted on time.

### Logs (`4`)

The captured log lines, newest at the bottom. Scrolling back with `k`/`j`, the arrows or PgUp/PgDn pins the view on the line it reached; `G` or End follows the newest lines again. `f` cycles the level shown (error, warn, info, debug, trace) and resumes following.

## The footer and dialogs

The footer lists the keys of what the keys act on (the view, the filter being typed, a dialog or the help), dropping the last ones when the row is full. A new message replaces them for ten seconds, marked `✓` or `✗`. On the right, the footer shows the work running, each with a spinner, then `? help`. While `e`, `d`, `r` and `t` are refused (see below), they are drawn crossed out and `locked` joins the work.

A dialog highlights its default answer, `n`. `y` confirms; `n`, Esc and Enter cancel. A `y` that deletes, cancels or abandons something, or quits while a stage runs (its requests in flight are lost), is drawn as an error. The view under a dialog, the help or a menu goes dim.

## Editing and deleting

`e` edits the selected question, answer or subtopic name in `$VISUAL` or `$EDITOR` (`vi` by default; `code --wait` works). The text goes through a file in `data/` readable only by you. An answer shows its reasoning and content between two marker lines, which must stay as they are.

- Editing a question's text gives it a new ID and deletes its old answer, since it now answers another question: run `answers` to answer it again.
- Renaming a subtopic gives its questions new IDs and keeps their answers.
- An edit that collides with another question or subtopic is refused. An edit refused after you typed it keeps its file, whose path is shown.

`d` deletes the selected item after a confirmation that says what goes with it: a subtopic takes its questions and their answers, a question its answer, an answer only itself. After every edit or deletion, `split` runs again.

A deleted subtopic or question is recorded in `data/rejected.jsonl`, so the stages do not generate it again: `subtopics` drops that name, and `questions` treats that text as already asked (its exact text, a case or spacing variant, or a near-duplicate). `--force` keeps these rejections. To allow one again, remove its line from `data/rejected.jsonl`.

## The data lock

While a stage, an edit or a training start runs in the TUI, and once it is quitting, `e`, `d`, `r` and `t` are refused. An `overbrainer` command run in another terminal at the same time is not locked out. Avoid editing in the TUI while a stage runs elsewhere: a stage appending to a file the TUI rewrites loses those lines. An edit checks that what it changes is still on disk as shown, and refuses otherwise.

## Quitting

`q` quits at once when nothing runs. Otherwise it asks, saying what becomes of each piece of work:

- a stage stops and resumes on its next run (its requests in flight are lost, already paid);
- a followed run keeps running and can be attached again;
- a run still starting is left running once its job has started, never cut during its start; for a Runpod run still provisioning, a second question offers to delete its pod instead;
- an edit or a cancel in progress is waited for, including a cancel confirmed while quitting.

While the help overlay is open, `q` (like Esc or `?`) closes it instead of quitting, and a filter being typed takes `q` as a character. Ctrl-C quits the same way `q` does, but from any of those contexts too, closing or leaving them first. After the TUI exits, it prints what it left running, with the commands to follow it again.

A SIGTERM or SIGHUP from outside ends the TUI without asking, as Ctrl-C does on the command line. So does SIGINT, except while `$EDITOR` holds the terminal: then it is ignored until the TUI takes the terminal back, since it would come from a Ctrl-C typed in the editor. An editor still running when the TUI ends is sent SIGTERM, then SIGKILL two seconds later if it has not exited.

If the terminal fails (it cannot be drawn on or read, or a view panics), the TUI quits as a confirmed `q` does, without asking, and gives the terminal back. stderr says what it waits for, and prints the warnings and errors logged meanwhile. A run still starting is waited for until its job has started, never abandoned; Ctrl-C at that point abandons it, as on the command line.

## Color and motion

The TUI paints its own dark crimson background in 24-bit color (when `COLORTERM` is `truecolor` or `24bit`) and in 256 colors (when `TERM` contains `256color`, as `tmux-256color` does). Otherwise it uses the 16 named colors on the terminal's own background.

| Variable | Values |
|---|---|
| `OVERBRAINER_TUI_COLOR` | `truecolor`, `256` or `16` picks the color level. `16` gives the terminal's background and palette back on any terminal. |
| `OVERBRAINER_TUI_MOTION` | `on`, `reduced` (spinners and bars only) or `off`. It is `reduced` by default over SSH (`SSH_CONNECTION` or `SSH_TTY` set and not empty). |
| `NO_COLOR` | Switches to a monochrome theme and turns motion off. |

An unknown value of either `OVERBRAINER_TUI_*` variable is ignored, with a warning in the Logs view.

Motion stays small: spinners on running work, progress bars that close on their new value, and, in 24-bit color only, a short fade when a view, a dialog or a new message appears and a slow pulse on the `●` of the run followed. Nothing is redrawn while nothing moves.

## Recording the demo

The GIF above is recorded with [VHS](https://github.com/charmbracelet/vhs) from [`assets/tui-tour.tape`](assets/tui-tour.tape) by [`assets/record.sh`](assets/record.sh), which builds overbrainer and runs VHS and gifsicle in containers:

```bash
docs/assets/record.sh path/to/a/demo-project
```
