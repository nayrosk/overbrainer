---
name: feature-flow
description: Take an overbrainer change from issue to a PR ready for the maintainer (issue, design approval, branch, worktree, code, tests, signed commits, PR, CodeRabbit rounds, CI, hand-over). Use when starting work on a feature, fix or chore in this repo.
---

# Feature flow

Follow the Workflow section of `CLAUDE.md`. Ask the maintainer before any decision the issue does not settle.

1. **Issue.** `gh issue list --search "<words>"` for duplicates, then `gh issue create` with labels, a milestone and `--assignee @me`. For anything beyond a small fix, write the spec and plan in `docs/design/` and wait for the maintainer's explicit approval.
2. **Branch and worktree.** `git fetch origin`, `gh issue develop <n> --name <type>/<n>-slug --base main` (`feat`, `fix` or `chore`), then `git worktree add ../overbrainer-wt/<n> <type>/<n>-slug`.
3. **Develop** strictly what the issue asks: modular code, tests, a doc comment on every item added or touched (tests included), user docs when behaviour changes. Dependencies through `cargo add`. Never edit `CHANGELOG.md`.
4. **Test** with the build limits:

   ```sh
   export CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="$HOME/.cache/overbrainer-target"
   cargo fmt --all --check
   cargo clippy --all-targets --locked -- -D warnings
   cargo test --all-targets --locked -- --test-threads=4
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
   cargo deny check
   cargo +"$(grep -m1 rust-version Cargo.toml | cut -d'"' -f2)" check --all-targets --locked
   ```

   Repeat clippy, test and the MSRV check with each cargo feature the change touches. Run `actionlint` after editing a workflow, and the SSH suites when SSH code changes. Run the `rust-reviewer` and `docs-checker` subagents on the diff. Paid live tests only with the maintainer's approval.
5. **Commit.** `git commit -S`, Conventional Commit subject, no AI attribution, co-author trailers or session links. Check with `git log --format=%B origin/main..HEAD`.
6. **PR.** `git rebase origin/main`, push once, `gh pr create --base main` with a Conventional title, `Closes #<n>`, the issue's labels and milestone, and the tests and checks that ran. English, no transcript, diff in scope.
7. **Review round.** Wait for CI and CodeRabbit. Count the checks (17 on a normal PR). Gather every inline and outside-diff finding, verify each, fix them all, rerun the checks, push once, reply in each thread and resolve it. Docstring coverage over touched functions: aim for 100%, 80% is the floor. During CodeRabbit's rate limit, wait for the window it names.
8. **Hand over.** Everything green and CodeRabbit approved: comment on the PR tagging `@nayrosk`. Do not merge, tag or publish unless asked. After the merge, remove the worktree and its `target/`, then pull `main`.
