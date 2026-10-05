---
name: release
description: Cut an overbrainer release (release PR with version bumps and changelog, signed tag, release workflow with trusted publishing). Use only when the maintainer asks for a release.
---

# Release

Publishing to crates.io cannot be undone. Get the maintainer's go for the merge and for the tag.

1. **Version.** On an up to date `main`, run `cargo semver-checks check-release`. Under 0.x any API break, even a new pub field, forces the minor slot. A wrong slot fails the `verify` job after the tag is pushed.
2. **Branch.** `git switch -c chore/release-vX.Y.Z origin/main`. No issue needed.
3. **Bump** the version in:
   - `Cargo.toml` (then build once so `Cargo.lock` follows);
   - `.claude-plugin/plugin.json`;
   - the `blob/vX.Y.Z` links in `skills/overbrainer/SKILL.md`;
   - the `version=vX.Y.Z` binary install example in `README.md`.
4. **Changelog.** `git cliff --tag vX.Y.Z -o CHANGELOG.md` (or `uvx git-cliff ...`). Check the new section.
5. **Check.** Run the checks from `CLAUDE.md`; tests fail until versions and links agree.
6. **PR.** Signed commit `chore: release vX.Y.Z`, push, `gh pr create --base main --title "chore: release vX.Y.Z" --label release`. The label makes CodeRabbit skip it.
7. **Merge.** Squash merge once the 15 required checks pass.
8. **Tag.** On the merge commit: `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`. Confirm with `git ls-remote --tags origin vX.Y.Z`.
9. **Watch** `release.yml`: verify, build, publish (trusted publishing, `production` environment), github-release. Then check the crate version on crates.io, the release archives, and close the milestone.
