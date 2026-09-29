# The init wizard: `overbrainer tui` in a directory without overbrainer.toml
# runs it before opening. Render with docs/assets/record.sh; this tape needs
# no demo project, just an empty directory made in the hidden setup below. It
# starts no paid work: the key typed is a placeholder, training is skipped,
# and the wizard's own "Start auto now?" is answered n, so it opens the
# Project view instead of a paid run.

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
Type "mkdir -p /work/wizard-demo && cd /work/wizard-demo && export COLORTERM=truecolor && clear" Enter
Show

Type "overbrainer tui" Sleep 300ms Enter
Sleep 2s

# 1. Project name: prefilled with the directory's name; keep it.
Sleep 1.5s
Enter Sleep 1s

# 2. Provider: from OpenRouter, the first preset, to NanoGPT.
Right Sleep 800ms
Enter Sleep 1s

# 3. The API key, typed and masked; never a real one.
Type "sk-demo" Sleep 500ms
Enter Sleep 1s

# 4. Models: the generator and the parent; reasoning and the embedder stay at
# their defaults.
Type "demo-generator" Sleep 300ms
Down Sleep 300ms
Type "demo-parent" Sleep 500ms
Enter Sleep 1s

# 5. Topics: a adds one. Subtopics and questions per subtopic come filled
# with 10 and 30: clear them before typing.
Type "a" Sleep 600ms
Type "rust_basics" Sleep 300ms
Down Sleep 200ms
Type "Core Rust language features" Sleep 300ms
Down Sleep 200ms
Backspace 2 Sleep 200ms
Type "3" Sleep 200ms
Down Sleep 200ms
Backspace 2 Sleep 200ms
Type "5" Sleep 800ms
Enter Sleep 1.2s
Enter Sleep 1s

# 6. Training: the target starts on local; one step left is skip, so no
# child model screen follows.
Left Sleep 1.2s
Enter Sleep 1s

# 7. Summary, then Enter writes the files.
Sleep 2.5s
Enter Sleep 2s

# Its own confirmation: no paid run now, so it opens the Project view.
Type "n" Sleep 3.5s

Type "q" Sleep 1s
