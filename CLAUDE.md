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

1. Open an issue first. Every change goes issue, branch, PR. Nobody pushes to `main`.
2. Branch from the issue: `gh issue develop <n> --name <type>/<n>-slug --base main`, with `<type>` one of `feat`, `fix` or `chore` (CI also accepts `feature`, `bugfix`, `hotfix`). No `docs/`, `perf/` or `test/` prefixes. Pick the prefix right away: renaming a PR's head branch closes the PR.
3. Work in a git worktree for that branch.
4. Commits: Conventional Commits, GPG-signed (`git commit -S`). No AI attribution, no `Co-Authored-By` trailers, no session links or transcripts in commits or PRs.
5. PR to `main`: Conventional title (it becomes the squash commit subject) and `Closes #<n>` in the body. `pr-policy.yml` enforces both.
6. Keep the diff in scope. No unrelated edits, not even a stray `.gitignore` line.
7. Review: CodeRabbit reviews the PR (it skips `docs:` titles and the `release` label). Its pre-merge checks need docstring coverage of at least 80% over touched functions (aim for 100%) and no out-of-scope changes.
8. Fix every finding of a review round locally, run the checks, push once, then reply in each thread. Each push costs a rate-limited review.
9. CI: count the checks, do not only look for failures. The `main` ruleset requires 15: `fmt`, `clippy`, `test`, `ssh`, `doc`, `msrv`, `macos`, `deny`, `crates-metadata`, `commits`, `branch`, `title`, `issue`, `analyze (rust)`, `analyze (actions)`.
10. Merge: squash only. The branch is deleted on merge.

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
