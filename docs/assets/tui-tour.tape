# The README hero: a tour of `overbrainer tui` on a finished demo project.
# Render with docs/assets/record.sh, which mounts the demo project at /demo.
# It starts no paid work: no r, no t, no a. The only dialog is answered n.

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
Sleep 2s

# Dataset: open the topic, a subtopic, a question and its answer.
Type "l" Sleep 500ms
Type "j" Sleep 400ms
Type "l" Sleep 600ms
Type "j" Sleep 400ms
Type "l" Sleep 500ms
Type "j" Sleep 3s
PageDown Sleep 1.5s

# Stats of the topic, then back to the detail pane.
Type "s" Sleep 2.5s
Type "s" Sleep 500ms

# Filter the tree, keep it, then clear it.
Type "/" Sleep 400ms
Type "borrow" Sleep 600ms
Enter Sleep 2.5s
Escape Sleep 1s

# The help overlay.
Type "?" Sleep 3s
Escape Sleep 1s

# Training: the succeeded run and its loss chart.
Type "3" Sleep 1.5s
Type "j" Sleep 500ms
Type "j" Sleep 3.5s

# Logs, then back to the dataset.
Type "4" Sleep 1.5s
Type "1" Sleep 1.2s
Type "j" Sleep 400ms
Type "j" Sleep 800ms

# A dialog on a question, answered with its default: nothing is deleted.
Type "d" Sleep 3s
Type "n" Sleep 1.5s
