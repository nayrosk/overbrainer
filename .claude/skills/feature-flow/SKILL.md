---
name: feature-flow
description: Take an overbrainer change from issue to a PR ready for the maintainer (issue, design approval, branch, worktree, code, tests, signed commits, PR, CodeRabbit rounds, CI, hand-over). Use when starting work on a feature, fix or chore in this repo.
---
# Feature flow

Follow Workflow section of `CLAUDE.md`. Ask maintainer before any decision issue not settle.

1. **Issue.** `gh issue list --search "<words>"` for duplicates, then `gh issue create` with labels, milestone, `--assignee @me`. Beyond small fix: write spec + plan in `docs/design/`, wait maintainer explicit approval.
2. **Branch and worktree.** `git fetch origin`, `gh issue develop <n> --name <type>/<n>-slug --base main` (`feat`, `fix` or `chore`), then `git worktree add ../overbrainer-wt/<n> <type>/<n>-slug`.
3. **Develop** only what issue asks: modular code, tests, doc comment on every item added/touched (tests too), user docs when behaviour change. Deps via `cargo add`. Never edit `CHANGELOG.md`.
4. **Test** with build limits:

   ```sh
   export CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="$HOME/.cache/overbrainer-target"
   cargo fmt --all --check
   cargo clippy --all-targets --locked -- -D warnings
   cargo test --all-targets --locked -- --test-threads=4
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
   cargo deny check
   cargo +"$(grep -m1 rust-version Cargo.toml | cut -d'"' -f2)" check --all-targets --locked
   ```

   Repeat clippy, test, MSRV check per cargo feature change touches. `actionlint` after workflow edit; SSH suites when SSH code change. Run `rust-reviewer` + `docs-checker` subagents on diff. Paid live tests only with maintainer approval.
5. **Commit.** `git commit -S`, Conventional Commit subject, no AI attribution, co-author trailers, session links. Check via `git log --format=%B origin/main..HEAD`.
6. **PR.** `git rebase origin/main`, push once, `gh pr create --base main` with Conventional title, `Closes #<n>`, issue labels + milestone, tests/checks run. English, no transcript, diff in scope.
7. **Review round.** Wait CI + CodeRabbit. Count checks (17 on normal PR). Gather all inline + outside-diff findings, verify each, fix all, rerun checks, push once, reply each thread + resolve. Docstring coverage on touched functions: aim 100%, floor 80%. CodeRabbit rate limit → wait window it names.
8. **Hand over.** All green + CodeRabbit approved: comment on PR tagging `@nayrosk`. No merge, tag, publish unless asked. After merge: remove worktree + its `target/`, pull `main`.