#!/usr/bin/env bash
# smrun — run every remote command inside a named tmux session, with a log.
#
# Usage:  smrun <session-name> <command...>
#   - refuses to reuse an existing session name
#   - appends a row to tmux/registry.tsv: time, session, cwd, git HEAD, command
#   - pipes stdout+stderr to tmux/<session>.log and records EXIT=<code>
#   - the session stays alive afterwards (window remains attachable)
#
# Inspect:  tmux ls | tmux attach -t <session> | tmux capture-pane -pt <session> -S -200
set -u
ROOT="${SMRUN_ROOT:-/train_result/log/supermimic}"
SESS="${1:?usage: smrun <session> <command...>}"; shift
CMD="$*"
LOG="$ROOT/tmux/$SESS.log"
REG="$ROOT/tmux/registry.tsv"
mkdir -p "$ROOT/tmux"
if tmux has-session -t "$SESS" 2>/dev/null; then
  echo "smrun: session already exists: $SESS" >&2
  exit 3
fi
HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "-")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$SESS" "$PWD" "$HEAD" "$CMD" >> "$REG"
tmux new-session -d -s "$SESS" -c "$PWD" \
  "bash -c 'set -o pipefail; echo START=\$(date -Is) | tee -a $LOG; ($CMD) 2>&1 | tee -a $LOG; echo EXIT=\$? END=\$(date -Is) | tee -a $LOG; sleep 20'"
echo "smrun: started $SESS -> $LOG"
