#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/adheerchauhan/git/trend_following"
PY="$REPO_ROOT/live_strategy/trend_following_strategy_v020_live/trend_following_v020_live.py"
LOG_DIR="/Users/adheerchauhan/live_strategy_logs/trend_following_v020_live/"
mkdir -p "$LOG_DIR"

# ---- DEBUG LOGGING (temporary while we stabilize) ----
{
  echo "----- $(date -u '+%F %T')Z cron start -----"
  echo "whoami=$(whoami)  host=$(hostname)"
  echo "HOME=$HOME"
  echo "PWD(before)=<skipped>"
} >> "$LOG_DIR/cron_env_debug.log" 2>&1

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/anaconda3/bin"
export PYTHONNOUSERSITE=1
cd "$REPO_ROOT" || { echo "cd failed"; exit 1; }
{
  echo "PWD(after_cd)=$(pwd)"
} >> "$LOG_DIR/cron_env_debug.log" 2>&1

export REPO_ROOT
unset PYTHONHOME PYTHONSAFEPATH PYTHONUSERBASE __PYVENV_LAUNCHER__ VIRTUAL_ENV

# Minimal but safe locale for Python
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# make your repo importable
#export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
#export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$REPO_ROOT"

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
#  "$PYTHON" -c 'import os,sys,pathlib; print("cwd:", os.getcwd()); print("sys.executable:", sys.executable); print("sys.version:", sys.version); print("repo_exists:", pathlib.Path(os.environ.get("REPO_ROOT","")).exists())'
echo "=== python stat-diag start ==="
"$PYTHON" -c '
import os, sys

print("sys.executable:", sys.executable)

# Get cwd safely once
try:
    cwd = os.getcwd()
    print("cwd:", cwd)
except Exception as e:
    print("cwd: <FAIL>", type(e).__name__, str(e))
    os.chdir(os.path.expanduser("~"))
    cwd = os.getcwd()
    print("cwd(after chdir home):", cwd)

print("---- sys.path ----")
for p in sys.path:
    print(p)

print("---- stat checks ----")
for p in sys.path:
    pp = p or cwd
    try:
        os.lstat(pp)
        islink = os.path.islink(pp)
        rp = os.path.realpath(pp)
        os.stat(pp)
        print("OK:", pp, "| islink=", islink, "| realpath=", rp)
    except Exception as e:
        print("FAIL:", pp, type(e).__name__, str(e))
'
echo "=== python stat-diag end ==="
} >> "$LOG_DIR/cron_env_debug.log" 2>&1 || {
  echo "[warn] python probe failed (continuing)" >> "$LOG_DIR/cron_env_debug.log"
}

export PYTHONUNBUFFERED=1

# Run your app
#"$PYTHON" "$PY" --dry-run --force-run --run-at-utc-hour 0 --gate-minutes 500 >> "$LOG_DIR/live_run.log" 2>&1
"$PYTHON" "$PY" --run-at-utc-hour 0 --gate-minutes 5 >> "$LOG_DIR/live_run.log" 2>&1
