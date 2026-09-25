<!-- The title becomes the commit on main: use a Conventional Commit, e.g. `feat(tui): show the run cost`. -->

Closes #

## Summary

<!-- What changes and why. -->

## Testing

<!-- Commands you ran and what they showed. -->

## Checklist

- [ ] The branch was created from the issue (`gh issue develop <number>`) and follows Conventional Branch
- [ ] Commits are signed and follow Conventional Commits
- [ ] `cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings` and `cargo test --all-targets --locked` pass
- [ ] No `unwrap`, `expect` or `panic!`; secrets never reach output, logs or errors
- [ ] Documentation and README updated when behavior changes
