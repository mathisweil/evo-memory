# tmux Quick Reference

## Basic Workflow

```bash
tmux new -s train          # start a named session
# run training inside here
# laptop sleeps → session stays alive
ssh <machine>              # reconnect later
tmux attach -t train       # reattach to session
```

## Session Management

| Command                     | Action               |
| --------------------------- | -------------------- |
| `tmux new -s NAME`          | Create named session |
| `tmux ls`                   | List all sessions    |
| `tmux attach -t NAME`       | Reattach to session  |
| `tmux kill-session -t NAME` | Kill a session       |

## Inside tmux (prefix = Ctrl-b)

| Keys                    | Action                  |
| ----------------------- | ----------------------- |
| `Ctrl-b d`              | Detach (leave running)  |
| `Ctrl-b c`              | New window              |
| `Ctrl-b n` / `Ctrl-b p` | Next / previous window  |
| `Ctrl-b %`              | Split pane vertically   |
| `Ctrl-b "`              | Split pane horizontally |
| `Ctrl-b arrow`          | Switch pane             |
| `Ctrl-b [`              | Scroll mode (q to exit) |
| `Ctrl-b x`              | Kill current pane       |
