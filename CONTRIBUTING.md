# Contributing

Thanks for helping. This guide covers how work flows from an idea to `main`.

By taking part you agree to the [code of conduct](CODE_OF_CONDUCT.md). Report vulnerabilities privately as described in the [security policy](SECURITY.md).

## Workflow

Every change goes issue, branch, pull request. Nobody pushes to `main` directly, the maintainer included.

1. **Issue.** Open one with the [issue forms](https://github.com/nayrosk/overbrainer/issues/new/choose), or comment on an existing one to say you are taking it. Wait for agreement on the approach before large changes.
2. **Branch.** Create the branch from the issue so GitHub links them:

   ```sh
   gh issue develop 42 --name feature/42-short-description --base main --checkout
   ```

   Branch names follow [Conventional Branch](https://conventional-branch.github.io): `feature/`, `bugfix/`, `hotfix/` or `chore/`, then lowercase words joined by hyphens. Starting with the issue number is recommended. CI rejects other names.
3. **Pull request.** Open it against `main` with the template. The description must close the issue (`Closes #42`), and the title must be a Conventional Commit because it becomes the commit subject on `main`.
4. **Review.** CI, CodeQL and CodeRabbit run on the pull request. Every check must pass and every review conversation must be resolved before merging.
5. **Merge.** Pull requests are squash merged, which closes the issue and deletes the branch.

Dependabot pull requests and release pull requests (`chore/release-vX.Y.Z`) do not need an issue.

## Commits

- Commits follow [Conventional Commits](https://www.conventionalcommits.org): `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore` or `revert`, with an optional scope, for example `fix(runpod): stop the pod when the watchdog fires`. CI checks every commit subject.
- Every commit must be signed, on every branch. Set up [GPG, SSH or S/MIME signing](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification) and turn it on with `git config commit.gpgsign true`. Pushes with unsigned commits are rejected.
- Rebase on `main` rather than merging it into your branch.

## Code

- Rust stable, with the minimum supported version from `rust-version` in `Cargo.toml`.
- No `unwrap`, `expect`, `panic!`, `todo!` or `dbg!`; return errors instead. No `#[allow(...)]`; fix the lint.
- Secrets stay in `secrecy::SecretString` and never reach output, logs, `Debug` or error messages.
- Add dependencies with `cargo add`; never edit `Cargo.lock` by hand.
- Code, comments and documentation are in English.

Before pushing:

```sh
cargo fmt --all --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
```

## Labels and milestones

Issues get a type label (`bug`, `enhancement`, `documentation`, ...) and, when relevant, an `area:` label. New issues start with `triage`. Labels are defined in [.github/labels.yml](.github/labels.yml); change that file to add or rename one. Accepted issues go into the milestone of the release that should ship them.

## Releases

Releases are cut by the maintainer; see [Releasing](README.md#releasing).
