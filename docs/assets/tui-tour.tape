# The README hero: a tour of `overbrainer tui` on a finished demo project.
# Render with docs/assets/record.sh, which mounts the demo project at /demo.
# It starts no paid work: no r, t, A, a, s or E. The only dialog is answered
# with its default, n.

Output tui-tour.gif

Set Shell "bash"
Set FontSize 16
Set Width 1264
Set Height 720
Set Padding 16
Set Theme "Catppuccin Mocha"
Set CursorBlink false
Set TypingSpeed 60ms

Hide
Type "cd /demo && export COLORTERM=truecolor && clear" Enter
Show

Type "overbrainer tui" Sleep 300ms Enter
Sleep 3s

# Project: it opens here. Move through a few fields; the stats pane on the
# right is always shown, no key needed.
Type "j" Sleep 400ms
Type "j" Sleep 400ms
Type "j" Sleep 400ms
Type "k" Sleep 400ms
Sleep 2.5s

# Dataset: open a topic, a subtopic, a question and its answer.
Type "2" Sleep 800ms
Type "l" Sleep 500ms
Type "j" Sleep 400ms
Type "l" Sleep 600ms
Type "j" Sleep 400ms
Type "l" Sleep 500ms
Type "j" Sleep 1.5s

# The merged answer pane: jump the detail from the question to its reasoning,
# then its answer, with [ and ].
Type "]" Sleep 1.5s
Type "]" Sleep 2s
Type "[" Sleep 1.2s

# A dialog on the selected question, answered with its default: nothing is
# deleted.
Type "d" Sleep 2.5s
Type "n" Sleep 1.2s

# Filter the tree, keep it, then clear it.
Type "/" Sleep 400ms
Type "borrow" Sleep 600ms
Enter Sleep 2.5s
Escape Sleep 1s

# The help overlay.
Type "?" Sleep 3.5s
Escape Sleep 1s

# Training: the latest run, listed first and selected, with its 48-step loss
# chart. The footer's version and live cost have been visible throughout.
Type "4" Sleep 9s
