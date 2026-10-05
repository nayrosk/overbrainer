---
name: release
description: Cut an overbrainer release (release PR with version bumps and changelog, signed tag, release workflow with trusted publishing). Use only when the maintainer asks for a release.
---
# Release

Publishing to crates.io irreversible. Get maintainer go for merge and tag.

1. **Version.** On up-to-date `main`, run `cargo semver-checks check-release`. Under 0.x any API break, even new pub field, forces minor slot. Wrong slot fails `verify` job after tag pushed.
2. **Branch.** `git switch -c chore/release-vX.Y.Z origin/main`. No issue needed.
3. **Bump** version in:
   - `Cargo.toml` (then build once so `Cargo.lock` follows);
   - `.claude-plugin/plugin.json`;
   - `blob/vX.Y.Z` links in `skills/overbrainer/SKILL.md`;
   - `version=vX.Y.Z` binary install example in `README.md`.
4. **Changelog.** `git cliff --tag vX.Y.Z -o CHANGELOG.md` (or `uvx git-cliff ...`). Check new section.
5. **Check.** Run checks from `CLAUDE.md`; tests fail until versions and links agree.
6. **PR.** Signed commit `chore: release vX.Y.Z`, push, `gh pr create --base main --title "chore: release vX.Y.Z" --label release`. Label makes CodeRabbit skip.
7. **Merge.** Squash merge once 15 required checks pass.
8. **Tag.** On merge commit: `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`. Confirm with `git ls-remote --tags origin vX.Y.Z`.
9. **Watch** `release.yml`: verify, build, publish (trusted publishing, `production` environment), github-release. Then check crate version on crates.io, release archives, close milestone.