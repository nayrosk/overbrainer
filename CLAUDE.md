# overbrainer

Instructions for coding agents on this repo. `AGENTS.md` points here.

overbrainer = Rust CLI + TUI. Distills large "parent" LLM into smaller open-weights "child" model. Generates subtopics, questions, answers from parent, dedups, splits into dataset, fine-tunes child with Axolotl (local, SSH, or Runpod GPU pod), exports to GGUF + Ollama Modelfile. User docs in `README.md` and `docs/`.

## Layout

- `src/main.rs`: entry point. `src/lib.rs`: library used by binary + tests.
- `src/cli/`: clap commands, one file per command group.
- `src/config/`: `overbrainer.toml` load, validate, edit.
- `src/pipeline/`: subtopics, questions, answers, split stages. `src/dedup/`: lexical + embedding dedup. `src/dataset/`: JSONL records.
- `src/llm/`: parent model clients (Anthropic, OpenAI compatible) behind `LlmClient`.
- `src/train/`: Axolotl trainer + metrics. `src/exec/`: local + SSH executors. `src/runpod/`: pod catalog, provisioning, watchdog, logs.
- `src/runs/`, `src/history.rs`, `src/project_lock.rs`: run records, state, locking.
- `src/secrets/`: `.env`, Vault, redaction. `src/export/`: GGUF export.
- `src/tui/`: ratatui front end, views, widgets, init wizard.
- `tests/it/`: all integration tests, one binary (`tests/it/main.rs` lists modules). Add new file there, register in `main.rs`; no top-level files under `tests/`.
- `tests/snapshots/`: insta snapshots. Review with `cargo insta review`; never hand-edit `.snap` files.
- `templates/`: files embedded via `include_str!`: what `overbrainer init` writes (config, `.env` example, `.gitignore`) + default prompts.
- `skills/overbrainer/`: user-facing agent skill shipped with binary, plus `.claude-plugin/`. Product content, not instructions for this repo.
- `docs/design/`: specs + plans. Gitignored, never committed.

## Build and test

Limit parallelism, share one target dir across worktrees. Parallel test linking overheats small machines; each worktree `target/` grows past 15 GB.

```sh
export CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR="$HOME/.cache/overbrainer-target"
cargo fmt --all --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked -- --test-threads=4
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
```

- Run clippy, test and doc again with `--features builtin-ssh` (the built-in SSH client; release binaries ship it):

```sh
cargo clippy --all-targets --locked --features builtin-ssh -- -D warnings
cargo test --all-targets --locked --features builtin-ssh -- --test-threads=4
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked --features builtin-ssh
```

- One test: `cargo test --test it -- runpod_flow::` or `cargo test --lib tui::`.
- Never `cargo clean` shared target dir while other worktree builds.
- CI uses latest stable toolchain: run `rustup update stable` before local clippy.
- SSH integration tests (`exec_ssh::`, `runpod_ssh::`) run only when `OVERBRAINER_TEST_SSH_HOST` (host alias) and `OVERBRAINER_TEST_SSH_CONFIG` (ssh config file defining it) set; else skip. `OVERBRAINER_TEST_SSH_JUMP_HOST` (ProxyJump bastion) enables the jump test. With `--features builtin-ssh` suites run under both clients; `OVERBRAINER_TEST_SSH_CLIENT` (`openssh` or `builtin`) limits them to one. `ssh` job in `.github/workflows/ci.yml` shows setup.
- If `ssh` on `PATH` is sandbox wrapper (e.g. firejail), put real ssh first for those tests: `PATH=/usr/bin:$PATH cargo test ...`, or build with `--features builtin-ssh`, whose suite run uses the built-in client. Wrapper kills background OpenSSH master connection.

## Code rules

Source of truth: `path_instructions` in `.coderabbit.yaml`; CodeRabbit reviews every PR against it. Short:

- No `unwrap()`, `expect()`, `panic!`, `todo!`, `unimplemented!`, `dbg!`, tests too. Tests return `Result`, use `?`.
- No `#[allow(...)]`. Fix lint at cause. Clippy pedantic on.
- Errors: `thiserror` enums in library modules; `anyhow` only in `main.rs` and `cli/` (+ existing TUI glue).
- Max 5 params per function. More → group in struct.
- Doc comment on every item added or touched, tests + private helpers included. Public fns returning `Result` get `# Errors` section.
- Traits only at swap points (`LlmClient`, `SecretSource`, `Deduplicator`, `Trainer`, `Executor`, `Observer`). No speculative abstraction.
- stdout = command output only; logs + errors to stderr.
- Tests never call `std::env::set_var`, never depend on dev environment.
- Code, comments, docs in English. No emoji, no em dashes.
- Deps only via `cargo add`. Never hand-edit `Cargo.lock`.
- Workflow files: actions pinned by full commit SHA + version comment, minimal permissions, `persist-credentials: false`, untrusted values via `env`. Validate with `actionlint`.

## Secrets

- Tokens + keys are `secrecy::SecretString`. Never reach stdout, stderr, logs, `Debug` output, error messages.
- Never format secret, or value derived from one, into assert message. CodeQL `rust/cleartext-logging` flags it. Use fixed assert messages.
- Never read or print `.env`, `.vault-token`, other secret files. Test presence only.

## Workflow

Every change follows these steps. `feature-flow` skill walks through them.

1. **Issue.** Search dupes (`gh issue list --search`). Open issue with right labels, milestone (create if none fits), assign self. Beyond small fix: need maintainer-approved design before code. Write spec + plan in `docs/design/`, wait for explicit yes.
2. **Branch.** From up-to-date `main`: `gh issue develop <n> --name <type>/<n>-slug --base main`, `<type>` one of `feat`, `fix`, `chore` (CI also accepts `feature`, `bugfix`, `hotfix`; no `docs/`, `perf/`, `test/`). Work in git worktree for that branch. Pick prefix right away: renaming PR head branch closes PR.
3. **Develop**, in English, strictly what issue asks:
   1. modular code per code rules above;
   2. unit + integration tests (TDD where fits);
   3. doc comment on every item added or touched, tests included;
   4. user docs (`README.md`, `docs/`, `overbrainer` skill) when behaviour changes. Don't edit `CHANGELOG.md`: git-cliff writes it.
4. **Test**, with build limits:
   1. unit tests;
   2. integration + end-to-end tests (`tests/it`; SSH suites when change touches SSH);
   3. `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`, `cargo deny check`, `actionlint` when workflow changes;
   4. no warning, no unused code, no `unwrap`, no `#[allow]`; snapshots accepted via `cargo insta review` after reading every changed line;
   5. multi-target: default build + every cargo feature change touches, plus MSRV (`cargo +<rust-version> check --all-targets`). CI covers musl + macOS.
   Paid live tests (Runpod, Hugging Face) only with maintainer approval; only delete resources you created.
5. **Commit.** Conventional Commits, GPG-signed (`git commit -S`) only. No AI attribution, no `Co-Authored-By` trailers, no session links: check `git log --format=%B origin/main..HEAD` before each push.
6. **Pull request.**
   - Sync from `main` by rebase (linear history), never merge.
   - Batch: push once per review round; each push costs rate-limited CodeRabbit review.
   - Title in Conventional Commits form (becomes squash commit subject), `Closes #<n>` in body, issue labels + milestone.
   - English, no transcript, state which tests + checks ran.
   - Diff in scope: no unrelated edits, not even stray `.gitignore` line.
7. **Review and CI.** Wait for CodeRabbit + CI, then fix:
   - Count checks (17 on normal PR; `main` ruleset requires 15: `fmt`, `clippy`, `test`, `ssh`, `doc`, `msrv`, `macos`, `deny`, `crates-metadata`, `commits`, `branch`, `title`, `issue`, `analyze (rust)`, `analyze (actions)`). Missing check ≠ pass.
   - Gather all findings of round, verify each, fix all, rerun checks, push once, reply in each thread + resolve.
   - CodeRabbit pre-merge checks must pass: docstring coverage over touched fns (80% min, aim 100%), no out-of-scope changes, linked issue covered. Skips `docs:` titles and `release` label.
   - During rate limit, don't repeat `@coderabbitai review`: wait for window it names.
8. **Hand over.** All green + CodeRabbit approved → tag maintainer (`@nayrosk`) on PR. Agents don't merge, tag releases, or publish unless maintainer asks. After merge: remove worktree + its `target/`, pull `main`.

## Release

Maintainer does it, or on their explicit request. Details in `README.md#releasing`.

1. Pick version via `cargo semver-checks check-release`: under 0.x any API break (even new pub field) forces minor slot.
2. Branch `chore/release-vX.Y.Z`, PR labelled `release`, no issue needed.
3. Bump `Cargo.toml`, `.claude-plugin/plugin.json`, `blob/vX.Y.Z` links in `skills/overbrainer/SKILL.md`, `version=vX.Y.Z` install example in `README.md`. Tests fail until they agree.
4. `git cliff --tag vX.Y.Z -o CHANGELOG.md` (format in `cliff.toml`).
5. After squash merge, tag merge commit: `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`.
6. Tag runs `release.yml`: verify, static binaries, crates.io via trusted publishing, GitHub release.

## Recommended plugins

`.claude/settings.json` registers marketplaces + enables plugins; Claude Code offers install when repo trusted.

- `superpowers` (`claude-plugins-official`): brainstorming, writing-plans, subagent-driven-development, TDD, systematic-debugging, verification. Use for every non-trivial change: design approval, plan, task-by-task review.
- `caveman`: terse output + `caveman-compress` for agent `.md` files. Keep agent files compressed.
- `brag`: launch video + share copy from project code. Use for release announcements.
- VoltAgent subagents (`voltagent-subagents`): `voltagent-lang` (`rust-engineer`), `voltagent-qa-sec` (code, security, test review), `voltagent-dev-exp` (CLI, docs, build), `voltagent-infra` (CI, release, Docker).

Repo subagents `rust-reviewer` + `docs-checker` (`.claude/agents/`) apply repo rules; prefer them for review of this repo.

## Writing

Docs, comments, commit messages, PR text: plain English, short sentences, imperative where fits. No em dashes, no marketing words, no filler openers or closing summaries.