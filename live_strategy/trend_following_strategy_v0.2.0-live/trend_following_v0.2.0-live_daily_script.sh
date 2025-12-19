#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/adheerchauhan/Documents/git/trend_following"
PY="$REPO_ROOT/live_strategy/trend_following_strategy_v0.2.0-live/trend_following_v0.2.0-live.py"
LOG_DIR="$REPO_ROOT/live_strategy/trend_following_strategy_v0.2.0-live/state"
mkdir -p "$LOG_DIR"

# ---- DEBUG LOGGING (temporary while we stabilize) ----
{
  echo "----- $(date -u '+%F %T')Z cron start -----"
  echo "whoami=$(whoami)  host=$(hostname)"
  echo "HOME=$HOME"
  echo "PWD(before)=$(pwd)"
} >> "$LOG_DIR/cron_env_debug.log" 2>&1

cd "$REPO_ROOT" || { echo "cd failed"; exit 1; }

# Minimal but safe locale for Python
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# make your repo importable
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# --- load email credentials (Gmail app password etc.) ---
# Put your exports in ~/tf_email.env (chmod 600) as:
#   export TF_EMAIL_ENABLED=1
#   export TF_SMTP_HOST="smtp.gmail.com"
#   export TF_SMTP_PORT=587
#   export TF_SMTP_USER="chauhan4@gmail.com"
#   export TF_SMTP_PASS="your-16-char-app-password"
#   export TF_EMAIL_FROM="Trend Bot <chauhan4@gmail.com>"
#   export TF_EMAIL_TO="chauhan4@gmail.com"
[ -f "$HOME/tf_email.env" ] && . "$HOME/tf_email.env"

# Use the env’s Python directly (avoid 'conda run' in cron)
PYTHON="/opt/anaconda3/envs/crypto_prod/bin/python"

# Quick sanity probe of Python BEFORE running your app
{
  echo "PYTHON=$PYTHON"
  "$PYTHON" -V
  "$PYTHON" - <<'PY'
import os, sys, pathlib
print("cwd:", os.getcwd())
print("sys.executable:", sys.executable)
print("sys.version_info:", sys.version_info)
print("path_exists(REPO_ROOT):", pathlib.Path(os.getenv("REPO_ROOT","")).exists())
PY
} >> "$LOG_DIR/cron_env_debug.log" 2>&1 || {
  echo "[fatal] python probe failed" >> "$LOG_DIR/cron_env_debug.log"
  exit 1
}

export PYTHONUNBUFFERED=1

# Run your app
#"$PYTHON" "$PY" --dry-run --force-run --run-at-utc-hour 0 --gate-minutes 500 >> "$LOG_DIR/live_run.log" 2>&1
"$PYTHON" "$PY" --run-at-utc-hour 0 --gate-minutes 5 >> "$LOG_DIR/live_run.log" 2>&1
