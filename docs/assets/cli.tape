# A short CLI demo: init a project, check a configured one, list its runs.
# Render with docs/assets/record.sh, which mounts the demo project at /demo.
# The keys below are placeholders set in the hidden setup, never real ones.

Output cli.gif

Set Shell "bash"
Set FontSize 16
Set Width 1264
Set Height 720
Set Padding 16
Set Theme "Catppuccin Mocha"
Set CursorBlink false
Set TypingSpeed 50ms

Hide
Type "export OVERBRAINER_PROVIDERS__NANOGPT__BASE_URL=https://nano-gpt.com/api/v1" Enter
Type "export OVERBRAINER_PROVIDERS__NANOGPT__API_KEY=sk-demo OVERBRAINER_RUNPOD__API_KEY=rpa-demo" Enter
Type "mkdir -p /work && cd /work && clear" Enter
Show

Type "overbrainer init my-project" Sleep 300ms Enter
Sleep 1.5s
Type "ls -A my-project" Sleep 300ms Enter
Sleep 2s
Type "clear" Enter
Type "cd /demo" Sleep 200ms Enter
Type "overbrainer config check" Sleep 300ms Enter
Sleep 5s
Type "clear" Enter
Type "overbrainer runs ls" Sleep 300ms Enter
Sleep 4s
