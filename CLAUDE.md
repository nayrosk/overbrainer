# overbrainer

Instructions for coding agents working on this repository. `AGENTS.md` points here.

overbrainer is a Rust CLI and TUI that distills a large "parent" LLM into a smaller open-weights "child" model. It generates subtopics, questions and answers from the parent, deduplicates and splits them into a dataset, fine-tunes the child with Axolotl (locally, over SSH or on a Runpod GPU pod), and exports the result to GGUF and an Ollama Modelfile. User docs live in `README.md` and `docs/`.

## Layout

- `src/main.rs`: entry point. `src/lib.rs`: the library the binary and tests use.
- `src/cli/`: clap commands, one file per command group.
- `src/config/`: `overbrainer.toml` loading, validation and editing.
- `src/pipeline/`: subtopics, questions, answers and split stages. `src/dedup/`: lexical and embedding dedup. `src/dataset/`: JSONL records.
- `src/llm/`: parent model clients (Anthropic, OpenAI compatible) behind `LlmClient`.
- `src/train/`: Axolotl trainer and its metrics. `src/exec/`: local and SSH executors. `src/runpod/`: pod catalog, provisioning, watchdog, logs.
- `src/runs/`, `src/history.rs`, `src/project_lock.rs`: run records, state and locking.
- `src/secrets/`: `.env`, Vault and redaction. `src/export/`: GGUF export.
- `src/tui/`: ratatui front end, views, widgets and the init wizard.
- `tests/it/`: all integration tests, built as a single binary (`tests/it/main.rs` lists the modules). Add a new file there and register it in `main.rs`; do not add top-level files under `tests/`.
- `tests/snapshots/`: insta snapshots. Review changes with `cargo insta review`; never edit `.snap` files by hand.
- `templates/`: files embedded with `include_str!`: what `overbrainer init` writes (config, `.env` example, `.gitignore`) and the default prompts.
- `skills/overbrainer/`: the user-facing agent skill shipped with the binary, plus `.claude-plugin/`. It is product content, not instructions for working on this repo.
- `docs/design/`: specs and plans. Gitignored, never committed.

## Build and test

Limit parallelism and share one target directory across worktrees. Parallel test linking overheats small machines and each worktree's `target/` grows past 15 GB.

```sh
export CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR="$HOME/.cache/overbrainer-target"
cargo fmt --all --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked -- --test-threads=4
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
```

- One test: `cargo test --test it -- runpod_flow::` or `cargo test --lib tui::`.
- Never run `cargo clean` on the shared target directory while another worktree builds.
- CI uses the latest stable toolchain: run `rustup update stable` before checking clippy locally.
- SSH integration tests (`exec_ssh::`, `runpod_ssh::`) run only when `OVERBRAINER_TEST_SSH_HOST` (a host alias) and `OVERBRAINER_TEST_SSH_CONFIG` (the ssh config file defining it) are set; otherwise they skip. The `ssh` job in `.github/workflows/ci.yml` shows the setup.
- If `ssh` on `PATH` is a sandbox wrapper such as firejail, put the real ssh first for those tests: `PATH=/usr/bin:$PATH cargo test ...`. The wrapper kills the background OpenSSH master connection.

## Code rules

The source of truth is the `path_instructions` in `.coderabbit.yaml`; CodeRabbit reviews every PR against it. In short:

- No `unwrap()`, `expect()`, `panic!`, `todo!`, `unimplemented!` or `dbg!`, in tests too. Tests return `Result` and use `?`.
- No `#[allow(...)]`. Fix the lint at its cause. Clippy pedantic is on.
- Errors: `thiserror` enums in library modules; `anyhow` only in `main.rs` and `cli/` (and the existing TUI glue).
- At most 5 parameters per function. Group them in a struct when more are needed.
- A doc comment on every item you add or touch, tests and private helpers included. Public functions returning `Result` get an `# Errors` section.
- Traits only at swap points (`LlmClient`, `SecretSource`, `Deduplicator`, `Trainer`, `Executor`, `Observer`). No speculative abstraction.
- stdout carries command output only; logs and errors go to stderr.
- Tests never call `std::env::set_var` and never depend on the developer's environment.
- Code, comments and docs in English. No emoji, no em dashes.
- Dependencies only through `cargo add`. Never edit `Cargo.lock` by hand.
- Workflow files: actions pinned by full commit SHA with a version comment, minimal permissions, `persist-credentials: false`, untrusted values through `env`. Validate edits with `actionlint`.

## Secrets

- Tokens and keys are `secrecy::SecretString`. They never reach stdout, stderr, logs, `Debug` output or error messages.
- Never format a secret, or a value derived from one, into an assert message. CodeQL `rust/cleartext-logging` flags it. Use fixed assert messages.
- Never read or print `.env`, `.vault-token` or other secret files. Test for presence only.

## Workflow

Follow these steps for every change. The `feature-flow` skill walks through them.

1. **Issue.** Search for duplicates (`gh issue list --search`). Open the issue with the right labels, a milestone (create one if none fits) and assign it to yourself. Anything beyond a small fix needs a design approved by a maintainer before code: write the spec and plan in `docs/design/` and wait for an explicit yes.
2. **Branch.** From an up-to-date `main`: `gh issue develop <n> --name <type>/<n>-slug --base main`, `<type>` one of `feat`, `fix` or `chore` (CI also accepts `feature`, `bugfix`, `hotfix`; no `docs/`, `perf/` or `test/`). Work in a git worktree for that branch. Pick the prefix right away: renaming a PR's head branch closes the PR.
3. **Develop**, in English, strictly what the issue asks:
   1. modular code following the code rules above;
   2. unit and integration tests (TDD where it fits);
   3. a doc comment on every item added or touched, tests included;
   4. the user docs (`README.md`, `docs/`, the `overbrainer` skill) when behaviour changes. Do not edit `CHANGELOG.md`: git-cliff writes it.
4. **Test**, with the build limits:
   1. unit tests;
   2. integration and end-to-end tests (`tests/it`; SSH suites when the change touches SSH);
   3. `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`, `cargo deny check`, `actionlint` when a workflow changes;
   4. no warning, no unused code, no `unwrap`, no `#[allow]`; snapshots accepted with `cargo insta review` after reading every changed line;
   5. multi-target: the default build and every cargo feature the change touches, plus the MSRV (`cargo +<rust-version> check --all-targets`). CI covers musl and macOS.
   Paid live tests (Runpod, Hugging Face) only with a maintainer's approval, and only delete resources you created.
5. **Commit.** Conventional Commits, signed with GPG (`git commit -S`) only. No AI attribution, no `Co-Authored-By` trailers, no session links: check `git log --format=%B origin/main..HEAD` before each push.
6. **Pull request.**
   - Sync from `main` by rebase (history is linear), never by merge.
   - Batch: push once per review round, since each push costs a rate-limited CodeRabbit review.
   - Title in Conventional Commits form (it becomes the squash commit subject), `Closes #<n>` in the body, the issue's labels and milestone.
   - Written in English, no transcript, and say which tests and checks ran.
   - Keep the diff in scope: no unrelated edits, not even a stray `.gitignore` line.
7. **Review and CI.** Wait for CodeRabbit and CI, then fix:
   - Count the checks (17 on a normal PR; the `main` ruleset requires 15: `fmt`, `clippy`, `test`, `ssh`, `doc`, `msrv`, `macos`, `deny`, `crates-metadata`, `commits`, `branch`, `title`, `issue`, `analyze (rust)`, `analyze (actions)`). A missing check is not a pass.
   - Gather every finding of a round, verify each one, fix them all, rerun the checks, push once, reply in each thread and resolve it.
   - CodeRabbit's pre-merge checks must pass: docstring coverage over touched functions (80% minimum, aim for 100%), no out-of-scope changes, linked issue covered. It skips `docs:` titles and the `release` label.
   - During its rate limit, do not repeat `@coderabbitai review`: wait for the window it names.
8. **Hand over.** Once everything is green and CodeRabbit approved, tag the maintainer (`@nayrosk`) on the PR. Agents do not merge, tag releases or publish unless a maintainer asks. After the merge: remove the worktree and its `target/`, then pull `main`.

## Release

Done by the maintainer, or on their explicit request. Details in `README.md#releasing`.

1. Pick the version with `cargo semver-checks check-release`: under 0.x any API break (even a new pub field) forces the minor slot.
2. Branch `chore/release-vX.Y.Z`, PR labelled `release`, no issue needed.
3. Bump `Cargo.toml`, `.claude-plugin/plugin.json`, the `blob/vX.Y.Z` links in `skills/overbrainer/SKILL.md` and the `version=vX.Y.Z` install example in `README.md`. Tests fail until they agree.
4. `git cliff --tag vX.Y.Z -o CHANGELOG.md` (format in `cliff.toml`).
5. After the squash merge, tag the merge commit: `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`.
6. The tag runs `release.yml`: verify, static binaries, crates.io through trusted publishing, GitHub release.

## Writing

Docs, comments, commit messages and PR text: plain English, short sentences, imperative where it fits. No em dashes, no marketing words, no filler openers or closing summaries.
