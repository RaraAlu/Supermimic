#!/usr/bin/env bash
# smrun：在命名的 tmux 会话中执行远程命令，并保存日志。
#
# 用法：smrun <session-name> <command...>
# 会话名已存在时拒绝启动；在 tmux/registry.tsv 追加时间、会话、目录、Git 提交和命令。
# 标准输出与错误写入 tmux/<session>.log，并记录 EXIT=<退出码>。
# 命令结束后保留会话 20 秒，便于连接查看。
# 查看：tmux ls | tmux attach -t <session> | tmux capture-pane -pt <session> -S -200
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
