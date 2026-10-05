---
name: feature-flow
description: Take an overbrainer change from issue to merged PR (issue, branch, worktree, signed commits, PR, CodeRabbit rounds, CI, squash merge). Use when starting work on a feature, fix or chore in this repo.
---

# Feature flow

Follow `CLAUDE.md`. Ask the maintainer before any decision the issue does not settle.

1. **Issue.** Find or open the issue (`gh issue view <n>` / `gh issue create`). Agree on the approach before large changes.
2. **Branch.** `gh issue develop <n> --name <type>/<n>-slug --base main`, `<type>` in `feat`, `fix`, `chore`. Get the prefix right now: renaming a PR's head branch closes the PR.
3. **Worktree.** `git fetch origin && git worktree add ../overbrainer-wt/<n> <type>/<n>-slug`.
4. **Code.** Follow the code rules. Doc comment on every item touched, tests included. Dependencies through `cargo add`.
5. **Check** with the build limits:

   ```sh
   export CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="$HOME/.cache/overbrainer-target"
   cargo fmt --all --check
   cargo clippy --all-targets --locked -- -D warnings
   cargo test --all-targets --locked -- --test-threads=4
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
   ```

   Run `actionlint` after editing a workflow. Optionally run the `rust-reviewer` and `docs-checker` subagents on the diff.
6. **Commit.** `git commit -S` with a Conventional Commit subject. No AI attribution, co-author trailers or session links.
7. **PR.** Push, then `gh pr create --base main` with a Conventional title and `Closes #<n>` in the body. Keep the diff in scope.
8. **Review round.** Wait for CodeRabbit. Gather every inline and outside-diff finding, verify each one, fix them all locally, rerun the checks, push once, reply in each thread. Make sure docstring coverage on touched functions reaches 100% (80% is the hard floor).
9. **CI.** Count the checks against the 15 required ones listed in `CLAUDE.md`; a missing check is not a pass.
10. **Merge.** Squash merge once checks pass and conversations are resolved, if the maintainer asked for it. Then remove the worktree.
