# The init wizard: `overbrainer tui` in a directory without overbrainer.toml
# runs it before opening. Render with docs/assets/record.sh; this tape needs
# no demo project, just an empty directory made in the hidden setup below. It
# starts no paid work: the key typed is a placeholder, training is skipped,
# and the wizard's own "Start auto now?" is answered n, so it opens the
# Project view instead of a paid run.
#
# NOTE for whoever records this once #63 (the init wizard) lands: the exact
# key presses within each screen (field order, which key adds a topic) are
# reasoned from docs/tui.md on that branch, not verified against the running
# wizard. Watch the first recording and adjust the timings or keys that miss.

Output wizard.gif

Set Shell "bash"
Set FontSize 16
Set Width 1264
Set Height 720
Set Padding 16
Set Theme "Catppuccin Mocha"
Set CursorBlink false
Set TypingSpeed 60ms

Hide
Type "mkdir -p /work/wizard-demo && cd /work/wizard-demo && clear" Enter
Show

Type "overbrainer tui" Sleep 300ms Enter
Sleep 2s

# 1. Project name.
Type "wizard-demo" Sleep 500ms
Enter Sleep 1s

# 2. Provider: cycle the choice to the NanoGPT preset.
Right Sleep 500ms
Enter Sleep 1s

# 3. The API key, typed and masked; never a real one.
Type "sk-demo" Sleep 500ms
Enter Sleep 1s

# 4. Models: the generator and the parent; reasoning and the embedder stay at
# their defaults.
Type "demo-generator" Sleep 300ms
Down Sleep 300ms
Type "demo-parent" Sleep 300ms
Enter Sleep 1s

# 5. Topics: add one, with a couple of subtopics and questions per subtopic.
Type "a" Sleep 500ms
Type "rust_basics" Sleep 300ms
Down Sleep 200ms
Type "Core Rust language features" Sleep 300ms
Down Sleep 200ms
Type "3" Sleep 200ms
Down Sleep 200ms
Type "5" Sleep 300ms
Enter Sleep 1s
Enter Sleep 1s

# 6. Training: skip, the default choice.
Enter Sleep 1s

# 7. Summary, then write the files.
Sleep 1.5s
Enter Sleep 2.5s

# Its own confirmation: no paid run now, so it opens the Project view.
Type "n" Sleep 2s

Type "q" Sleep 500ms
