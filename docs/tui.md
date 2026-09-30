# The terminal UI

![A tour of overbrainer tui: the dataset tree and an answer's reasoning, the topic stats, a filter, the help, a training run's loss chart, the logs and a dialog](assets/tui-tour.gif)

`overbrainer tui` shows the project in five views, switched with `1` to `5` (or Tab, Shift-Tab): the project's configuration and stats, the dataset, the pipeline stages, the training runs, and the logs. It opens on the Project view; in a directory without `overbrainer.toml`, it opens the [init wizard](#the-init-wizard) first. It runs the same stages and training flows as the commands, and [auto mode](#auto-mode) runs them all in a row. It needs a terminal (it refuses to start when stdout is not one) and is laid out for at least 80x24 characters. While the TUI runs, logs go to its Logs view instead of stderr; `OVERBRAINER_LOG` still sets what is captured.

## Keys

Everywhere:

| Keys | Action |
|---|---|
| `1` `2` `3` `4` `5`, Tab, Shift-Tab | Switch view. |
| `?` | List the keys of the current view. Esc, `?` or `q` closes it. |
| `q`, Ctrl-C | Quit. |
| `R` | Reload the data files and the runs list from disk. |
| `g` | Open the overbrainer repository in a browser. |
| `r` | Run auto mode, a pipeline stage, or `run` (asks which). |
| `A`, in Project and Pipeline | Auto mode: every stage, then training (asks first). |
| `y`, then `n`, Esc or Enter | In a dialog: confirm, or cancel (the default, `n`). |

Project:

| Keys | Action |
|---|---|
| `k` `j`, PgUp, PgDn, Home, End | Select a field, by page, the first or the last. |
| Enter | Edit the field and save it to `overbrainer.toml`; toggles a bool, cycles a choice, opens the catalog picker on a Runpod target's `gpu_types`, `data_center_ids`, `network_volume_id` or `image`. |
| `t` `o`, in a Runpod picker | Type the value instead of picking it; sort the entries another way. |
| `a` | Add a topic, a provider or a target. |
| `d` | Delete the selected topic, provider or target (asks first). |
| `u` | Undo the last write to `overbrainer.toml`. |
| `E` | Open `overbrainer.toml` in `$EDITOR`. |

Runpod catalog picker (opened from the Project view, or with `g`/`c` before a run starts, see below):

| Keys | Action |
|---|---|
| `k` `j`, Up, Down, PgUp, PgDn, Home, End | Move. |
| Space | Toggle the entry under the cursor (GPU types, data centers: several may be chosen, in the order toggled). |
| `J` `K` | Move the toggled entry under the cursor later or earlier in that order. |
| `o` | Sort another way: GPU types by price, VRAM (most first), then the number of data centers with the type in stock (most first); data centers by ID, then region. The order shown is in the picker's title; toggled entries stay at the top. |
| `/` | Filter the entries. Enter keeps the filter, Esc clears it. |
| Enter | Keep what is toggled (GPU types, data centers), or pick the entry under the cursor (network volume, image). |
| `t` | Type the value instead of picking it. Not offered from the start confirmation's `g`/`c` pickers. |
| Esc | Cancel: nothing changes. |

The GPU type picker's FIT column says whether each type holds the training run, from the [VRAM estimate](runpod.md#vram-estimate) shown in its title: `ok`, `tight` (less than 10% to spare), `small` (dim, cannot be chosen; one already chosen can still be taken out) or `?` (no estimate, or a target other than `training.target`). Opened from the start confirmation, it uses the estimate the confirmation shows.

Pipeline:

| Keys | Action |
|---|---|
| `c`, while auto mode runs a stage | Cancel that stage and the rest of the chain (asks first). |

Dataset:

| Keys | Action |
|---|---|
| `k` `j`, Up, Down | Move. |
| `l` `h`, Right, Left | Expand, collapse. |
| Enter | Expand or collapse. |
| PgUp, PgDn | Scroll the detail pane. |
| `[` `]` | Jump the detail pane from part to part (the question, its reasoning, its answer). |
| `/` | Filter the tree. Enter keeps the filter, Esc clears it. |
| `s` | Stats pane. |
| `e` | Edit the selected question's text or a subtopic's name in `$EDITOR`. |
| `E` | Edit the selected question's answer in `$EDITOR`. |
| `d` | Delete the selected item, with what depends on it (asks first). |
| `D` | Delete the selected question's answer only (asks first). |

Training:

| Keys | Action |
|---|---|
| `k` `j`, Up, Down | Select a run. |
| `a` | Attach: follow the selected run again. |
| `c` | Cancel the selected run's job (asks first). On a Runpod run still starting, abandon it instead (asks first). |
| `t` | Start a training run (asks first). |
| `s` | Stop the selected run's job with a snapshot (asks first). |
| `T` | Start a training run resuming the selected stopped run (asks first). |
| `x` | Hide the failed runs from the list until the TUI restarts (asks first). Nothing is deleted. |
| `p` | Dismiss the selected run's pod from the view, or show it again. Refused while the run is followed. |
| `g` `c`, in the start confirmation for a Runpod target | Choose the GPU types or the data centers from the catalog instead of what `overbrainer.toml` has; the choice is saved to `overbrainer.toml` when the run starts. |

Logs:

| Keys | Action |
|---|---|
| `k` `j`, Up, Down, PgUp, PgDn | Scroll. |
| `G`, End | Follow the newest lines. |
| `f` | Cycle the level shown: error, warn, info, debug, trace. |
| `s` | Show the pod log of the run selected in the Training view, or overbrainer's own log again. |
| `x` | Export the lines shown to `.overbrainer/logs-<timestamp>.log` (`logs-pod-<run-id>-<timestamp>.log` for a pod log). |

## The views

### Project (`1`)

The left pane lists the effective configuration section by section: `project`, each topic, each provider, the roles, `pipeline`, `training`, each target, `metrics`, then the env-only `runpod` and `other` tables. A value comes from `overbrainer.toml`, else from the environment, else from its default, shown dim; a field with no value shows `unset`. A value the environment sets is marked `(env)` and is read-only here: change it in `.env` instead (a field only the environment can ever set, such as a provider's `base_url`, is read-only the same way). A secret never shows its value, only `set`, `unset` or `vault ref` for a `vault:` reference.

`k`/`j`, PgUp/PgDn, Home/End move the selection. Enter opens the field for editing: a bool toggles at once, a choice cycles through its values (and to unset, when the field may be left out), and anything else opens a one-line form under the list, shown with what it accepts and any error from the last Enter. A number is checked when the form is submitted with Enter, not as it is typed. `a` asks what to add, a topic, a provider or a target, then its name (must match `^[a-z0-9_]+$` and be new) and, for a provider or a target, its protocol or kind. `d` deletes the selected topic, provider or target after a confirmation; it is refused while something uses it. Each change is saved to `overbrainer.toml` at once: the value as typed with Enter in the form, a toggled bool or a cycled choice, a pick, a table added or a confirmed deletion. A new Runpod target is written with `gpu_types = "auto"`, so it is valid as added.

On a Runpod target, Enter on `gpu_types` or `data_center_ids` instead opens a picker reading the Runpod catalog live, its top row `auto` (cheapest GPU types in stock, or the data centers with a chosen GPU type in stock); Space toggles an entry, `J`/`K` move it in the chosen order, `o` sorts the entries another way, and Enter saves the choice, `auto` included. Enter on `network_volume_id` opens a picker of the account's network volumes, its top row `none` (unsets it); picking a volume, or typing one the volume listing read last has, also sets `data_center_ids` to the volume's own data center. Once a volume listing is read, a target whose volume it lists in another data center gets that data center, saved in a write of its own, and the status line says so; while such a volume is set, changing `data_center_ids` to anything else (in its picker or typed) is refused: the status line says to pick another `network_volume_id` instead. A volume the listing does not have, or no listing yet, leaves `data_center_ids` free; since a volume needs `data_center_ids` with exactly one entry, the volume's, set it first for such a volume. Enter on `image` opens a picker of the account's pod templates, its top row `default` (unsets `image`, so the pinned Axolotl image applies); a template with no image cannot be picked. In any of these pickers, `t` types the value instead, and Esc cancels without changing anything. Selecting `gpu_count` or `max_hours` reads the GPU catalog in the background and adds a hint: the most GPUs a pod can have with the chosen types (or on Runpod's Secure Cloud with `auto`), and the most a run can cost at `max_hours`. Picking or typing a list of GPU types for `gpu_types` (instead of `auto`) also unsets `min_vram_gb` and `max_price_per_hour`, since those narrow `gpu_types = "auto"` only, and the status line says so.

Each change is validated against the whole configuration before anything is written: on error, nothing is written, the form stays open with the problem (or the status line says it, for a change made without the form), and the field keeps its value. A change that sets several fields (a volume with its data center, a list of GPU types with the `auto` limits it unsets, a renamed topic) is one write. The file is written atomically, keeping its permissions and its comments; the write is refused if the file changed on disk since it was read or if it is a symlink. No other change starts until a write ends. `u` undoes the last write from the TUI: it writes back the file as it was before it, unless the file changed on disk since, or a stage or a training run uses a field it would change (it can be undone once that ends). There is one level of undo: an undo cannot be undone, and a reload of the file or `E` leaves nothing to undo. `E` opens `overbrainer.toml` whole in `$EDITOR`, for what the form does not cover, such as `training.axolotl_extra`: it is refused while a stage, an edit or a write runs, or while a training run is starting or being followed (since it would change the training table).

`overbrainer.toml` and `.env` are read again when they change on disk, every 2 seconds at most, except while a save runs or `E` has the file: a valid configuration replaces the one shown, as a save does, and the next stages and runs use it; an invalid one leaves the view as it was and marks the fields its problems name, and the stages, runs, edits and catalog pickers started meanwhile keep using the configuration the view shows. A write from the view is not read again.

A field a running pipeline stage uses is read-only until the stage ends: the roles it sends requests to (`subtopics` and `questions` use the generator and the embedder, `answers` uses the parent, `run` uses all three) and the providers those roles use. A field a training run uses is read-only from the moment it starts to when it ends: the whole `training` table and the target it trains on. The row shows `(used by …)` (or `(used)` when there is no room) and the detail line under the list says which; Enter and `d` refuse a change or a deletion that touches a locked field or table the same way.

The right pane shows the project's stats: cost and tokens (in/out) per pipeline stage and in total, the dataset's topic, subtopic, question and answer counts and the train/eval split, the training runs by state, the estimated Runpod spend of their pods, and the cost per model used. A group not yet known says so (`nothing spent yet`, `not read yet`, `no runs yet`) instead of showing zeros.

### Dataset (`2`)

The topics, their subtopics and questions as a tree, with a detail pane for the selected item. A question's detail shows the question's text and a line with its ID and status, then, when it has an answer, the answer metadata (model, tokens, finish reason), `── reasoning ──` (when the answer has one) and `── answer ──` with the answer's text; there is no separate node for the answer in the tree. PgUp and PgDn scroll the detail pane; `[` and `]` jump it from part to part (the question, its reasoning, its answer). With `s`, the pane shows the stats of the selected topic and the train and eval sizes instead. `/` filters the tree by a case-insensitive substring of question texts and subtopic names. Enter keeps the typed filter applied; Esc clears the filter, whether it is being typed or already applied. Questions show `[a]` when answered and used for training, `[x]` when the answer is excluded, `[o]` when orphaned (its topic is no longer configured), and `[ ]` when unanswered. Topics no longer in `overbrainer.toml` and questions whose subtopic is gone are shown too, so they can be deleted. While a stage started from the TUI runs, the view reads the data files again every 2 seconds and whenever it is shown, so its counts follow what the stage writes; the selection and open nodes stay.

### Pipeline (`3`)

`r` opens a menu of `auto`, the stages and `run`, always on every topic and without `--force` (both stay command-line options). The view shows each stage's progress, requests in flight, items being retried, failures, tokens and cost, and the summary lines the command would print. The cost updates live as the running stage's items finish, before the stage itself reports its total. `run` stops after `split` here: training starts only with `t`, or with auto mode, whose chain shows under the stages (see [auto mode](#auto-mode)).

### Training (`4`)

The runs of `runs/` with the system panel of the selected run, and for the selected one its progress, ETA, pod and estimated spend, a loss chart and sparklines of the learning rate and gradient norm. Once the metrics carry the run's `max_steps`, the chart's x axis spans 0 to `max_steps` from the first step on, so the curves fill it as the run goes; before that, it spans the steps seen so far.

- `t` starts a run on `training.target` after a confirmation that shows the target, the model and the data. For Runpod it also shows the VRAM the run needs per GPU (the [estimate](runpod.md#vram-estimate), or why it is unknown, and the floor it gives `auto`), each GPU type with its catalog list price times `gpu_count`, VRAM, stock and fit (`ok`, `tight`, `small` or `?`, with a warning when a listed type is `small`), and the most `max_hours` can cost at the highest listed rate, marked "(some prices unknown)" when a price could not be read; with `gpu_types = "auto"` or `data_center_ids = "auto"`, the catalog read shows what `auto` would pick right now instead. `--target` and `--keep-pod` stay command-line options.
- `g` and `c` in that confirmation open the catalog picker (no `t`: only picking, never typing) to choose the GPU types or the data centers the run will use instead of what `overbrainer.toml` has; the dialog shows again once the picker closes, with what changed on a line of its own. `y` then saves that choice to `overbrainer.toml` first (validated and written atomically, like a change in the Project view, which `u` there undoes) and only starts the run once the save succeeds; a refused save (validation, the file changed on disk, a lock) starts nothing: the confirmation closes, its choices are dropped, the status line says why, and `t` opens it again. `g` and `c` are themselves refused while `overbrainer.toml` is being saved, or on a field the environment sets or a running task locks.
- A start holds the data lock until its job begins and is never interrupted before then. Quitting while a Runpod run is still provisioning offers to abandon it instead of waiting.
- `a` follows a run again, and `c` cancels its job after a confirmation. A Runpod run still starting has no job to cancel yet, so `c` offers to abandon that run instead, and the TUI stays open. If its pod is still being prepared, the pod is deleted and the run fails. Once its job is being sent, the run is detached, and `c` then cancels it.
- `s` stops a running run with a snapshot after a confirmation, as `overbrainer train stop` does (see [Training](training.md#stopping-with-a-snapshot)): a task following the run is detached first, then the stop follows it until it ends, retrieves the checkpoint and, for Runpod, deletes the pod. The run shows `stopping` meanwhile, then `stopped`, with `snapshot at step N (reason), output/ partial: T resumes it` under its ID: the model in its `output/` is the partial one from that step, not a finished model.
- `T` on a stopped run opens the start confirmation of a new run resuming it, titled "Resume a training run?", with the snapshot and the stopped run's data; `y` runs `overbrainer train --resume-from RUN_ID`. A stopped run whose snapshot is missing locally, or whose training settings changed since, is refused with the reason. A resumed run shows the run it resumed from under its ID.
- With `max_cost_usd` set on a Runpod target, the start confirmation also says what the run may spend: the snapshot at 95%, the pod deleted at 100%.
- The pod line shows the state `pod.json` records (`running` once the job started), the rate, the uptime and spend so far, and when the watchdog deletes the pod at the latest. A deleted pod shows when it was deleted, how long it existed and what it cost. The TUI only learns that a pod is gone from `pod.json`: a pod its watchdog deleted while nothing followed the run still shows as it was last recorded, until `overbrainer train attach` or `overbrainer pod ls` finds it gone.
- `x` hides the failed runs that no task follows, after a confirmation. They stay hidden for as long as the TUI runs, even when `runs/` is read again; their files are kept and `overbrainer runs ls` still lists them. A run that fails later shows until `x` is pressed again.
- `p` dismisses the selected run's pod: its pod line and its pod column show nothing, until `p` shows them again or the run is followed again. It is refused while a task follows the run, since that pod is live.
- On a terminal 120 columns wide or more, when the rows under it still leave 14 for the selected run, a system panel sits right of the runs: the machine of the selected run, sampled every 10 seconds while a task follows it (see [Training](training.md)). One row per disk (`disk` for the file system of the run directory, `root` for `/` when it is another), then `cpu`, `mem` and one per GPU, each with a gauge, its percentage, a sparkline and the figures behind it: space used and size, CPU use, memory used and its limit, and for a GPU its memory, temperature and power (the temperature and power are left out when they do not fit). A GPU's gauge is its utilisation. On a host the CPU figures are the load average over the CPU count; inside a container, whose load average is the host's, they are the cores busy over the container's CPUs (`4.0/8.0 cores`). Gauges turn yellow at 85% and red at 95%; a GPU's figures do the same with its memory. A network file system, such as a Runpod network volume, is marked `shared` and stays dim: `df` reports the whole shared cluster there, not what the volume holds or allows, so its figures never warn (the volume's own usage is measured separately). With more than four GPUs, the first three show and a `rest` row averages the others. The last 60 samples of each run (ten minutes) are kept while the TUI runs; each sparkline cell shows the peak of its share of them, and a cell without the figure stays blank. A run no longer followed shows its last sample. Once the last sample is more than 20 seconds old, followed or not, the panel's title shows its age. Before the first sample of a followed run the panel says so.
- Leaving the view does not stop following a run: while the TUI is open, its results are still retrieved and its pod deleted on time.

### Logs (`5`)

The captured log lines, newest at the bottom. Scrolling back with `k`/`j`, the arrows or PgUp/PgDn pins the view on the line it reached; `G` or End follows the newest lines again. `f` cycles the level shown (error, warn, info, debug, trace) and resumes following. `x` exports every retained line at the shown level or more severe, oldest first, to a new file `.overbrainer/logs-<YYYYMMDDTHHMMSSZ>.log`; it never overwrites an existing file, writes none when no line is at the shown level, and the footer says how many lines went where, or why it wrote none.

`s` switches to the Runpod pod log of the run selected in the Training view, as overbrainer keeps it in `runs/<run-id>/.pod/pod.log` (see [pod logs](runpod.md#pod-logs)), read again every 2 seconds. Each line shows its time, its source (`sys` for Runpod's own lines, such as the image pull, `ctr` for the container's output) and the line, secrets masked. `f` does not apply there; `x` exports that log to `.overbrainer/logs-pod-<run-id>-<YYYYMMDDTHHMMSSZ>.log`. `s` again shows overbrainer's own log.

## The footer and dialogs

The footer lists the keys of what the keys act on (the view, the filter or a form being typed, a dialog or the help), dropping the last ones when the row is full. A new message replaces them for ten seconds, marked `✓` or `✗`: `✓ config reloaded` when `overbrainer.toml` or `.env` changed on disk and was applied, `✗ overbrainer.toml: <problem>` or `✗ cannot parse .env (syntax error at line N)` when the change is invalid and the previous settings stay (see [reloading](configuration.md#reloading-while-overbrainer-runs)). On the right, the footer shows the work running, each with a spinner, `locked` while the data lock holds (see below), the project's cost so far, the version, then `? help`. The cost is `$` with two decimals, a trailing `+` when part of it is unknown, and it is left out while nothing has been spent. It includes the estimated spend of the Runpod pods, so it can differ from `overbrainer history`, which counts the stages only. When a newer overbrainer release is available, `↑ X.Y.Z` follows the version, naming that release. The cost, the version and that marker are left out while a dialog, the help overlay or the `r` menu is open, so the hints have the room. While `e`, `d`, `r`, `A` and `t` are refused (see below), they are drawn crossed out and `locked` joins the work.

A dialog highlights its default answer, `n`. `y` confirms; `n`, Esc and Enter cancel. A `y` that deletes, cancels or abandons something, or quits while a stage runs (its requests in flight are lost), is drawn as an error. The view under a dialog, the help or a menu goes dim.

## The init wizard

`overbrainer tui` in a directory without `overbrainer.toml` asks what the project needs, one screen at a time, then writes it:

1. The project name (the directory's name until changed).
2. The provider: OpenRouter, NanoGPT, OpenAI or Anthropic fill in the protocol and the base URL; `custom` asks for a name (`^[a-z0-9_]+$`), the protocol and the base URL.
3. Its API key: typed (shown as `•`, never shown again), a `vault:<mount>/<path>#<field>` reference, or empty to fill `.env` later.
4. The models: the generator, the parent (with reasoning on or off) and an optional embedder, as the provider names them.
5. The topics: `a` adds one (name, description, subtopics, questions per subtopic), `e` edits the selected one, `d` deletes it. At least one is needed.
6. Training: `skip` (auto mode then stops after split), `local` (native or docker), `ssh` (the host, native or docker) or `runpod` (its API key, and `auto` or a list of GPU types); then the base model and the adapter.
7. A summary, then Enter writes the files.

Enter or Tab goes to the next screen, Shift-Tab or Esc to the one before, every value kept; ↑ and ↓ move between the fields of a screen, ← and → change a choice, and a paste goes to the field being typed. A screen refuses Next while a required field is empty or invalid, and says why under the field. Ctrl-C asks, then quits without writing anything. The wizard runs before anything else and catches no signal: a SIGTERM or SIGHUP while it is open may leave the terminal in raw mode, which `reset` fixes.

The wizard writes `overbrainer.toml` (the template `overbrainer init` writes, with the answers set and its comments kept, validated before anything is written), `.env` with the `OVERBRAINER_*` variables the choices need (the base URL, the keys, the SSH host; a key left empty is an empty line to fill), created readable by you only (mode 600), `.env.example` with the same variables and no secret, the prompt templates in `prompts/`, and the entries `.gitignore` lacks. Like `init`, it never overwrites a file: when one of them exists, `overbrainer tui` names it and exits before the first screen, and a file that appears meanwhile is named at the summary, with nothing written. A write that fails partway removes what it created, so writing again works. A project directory given with `-C` that does not exist is never created: `overbrainer tui` fails as before. The last screen asks "Start auto now?": yes opens the TUI on auto mode's confirmation, no on the Project view. `overbrainer init` stays non-interactive.

## Auto mode

`A` in the Project or Pipeline view, or `auto` at the top of the `r` menu, runs every stage, then training. One confirmation comes first: the stages that will run, then what `t` would show (the target, the model and, for Runpod, each GPU type's list price and the most `max_hours` can cost). Without `[training]`, it says the chain stops after split.

The chain runs subtopics, questions, answers and split, each once the one before ended without failed items, then starts a run on `training.target` as `t` does, switches to the Training view and follows it. The Pipeline view shows the chain under the stages, for example `auto  subtopics ✓ → questions ● → answers → split → train`. At the end, the status line says where the model is: `✓ auto done: runs/<run-id>/output`, and the merged model in `runs/<run-id>/output/merged` with `merge = true`.

A failure stops the chain at its stage, with the error; running auto mode again resumes it, since the stages skip what is done. `c` in the Pipeline view, while the chain runs a stage, cancels that stage and the rest of the chain after a confirmation. Once training has started, the run is a run like any other: `c` in the Training view cancels it as it does today.

## Editing and deleting

`e` edits the selected question's text or a subtopic's name in `$VISUAL` or `$EDITOR` (`vi` by default; `code --wait` works); `E` edits the selected question's answer instead. The text goes through a file in `data/` readable only by you. An answer shows its reasoning and content between two marker lines, which must stay as they are.

- Editing a question's text gives it a new ID and deletes its old answer, since it now answers another question: run `answers` to answer it again.
- Renaming a subtopic gives its questions new IDs and keeps their answers.
- An edit that collides with another question or subtopic is refused. An edit refused after you typed it keeps its file, whose path is shown.

`d` deletes the selected item after a confirmation that says what goes with it: a subtopic takes its questions and their answers, a question its answer; `D` deletes only the selected question's answer, leaving the question. After every edit or deletion, `split` runs again.

A deleted subtopic or question is recorded in `data/rejected.jsonl`, so the stages do not generate it again: `subtopics` drops that name, and `questions` treats that text as already asked (its exact text, a case or spacing variant, or a near-duplicate). `--force` keeps these rejections. To allow one again, remove its line from `data/rejected.jsonl`.

## The data lock

While a stage, an edit or a training start runs in the TUI, and once it is quitting, `e`, `d`, `r`, `A` and `t` are refused. `overbrainer tui` also holds the project's [lock](pipeline.md#project-state) for as long as it runs, so no other overbrainer command can write to the same project at the same time. An edit checks that what it changes is still on disk as shown, and refuses otherwise.

## Quitting

`q` quits at once when nothing runs. Otherwise it asks, saying what becomes of each piece of work:

- a stage stops and resumes on its next run (its requests in flight are lost, already paid);
- a followed run keeps running and can be attached again;
- a run still starting is left running once its job has started, never cut during its start; for a Runpod run still provisioning, a second question offers to delete its pod instead;
- an edit or a cancel in progress is waited for, including a cancel confirmed while quitting;
- a write of `overbrainer.toml` in progress is waited for.

While the help overlay is open, `q` (like Esc or `?`) closes it instead of quitting, and a filter or a Project form being typed takes `q` as a character. Ctrl-C quits the same way `q` does, but from any of those contexts too, closing or leaving them first. After the TUI exits, it prints what it left running, with the commands to follow it again.

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
