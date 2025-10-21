#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/adheerchauhan/Documents/git/trend_following"
PY="$REPO_ROOT/live_strategy/trend_following_strategy_v0.1.0-live/trend_following_v0.1.0-live.py"
LOG_DIR="$REPO_ROOT/live_strategy/trend_following_strategy_v0.1.0-live/state"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

CONDA_BIN="/opt/anaconda3/condabin/conda"

# make your top-level repo importable (so 'strategy_signal', 'utils', etc are found)
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

export PYTHONUNBUFFERED=1
"$CONDA_BIN" run -n crypto_prod --no-capture-output \
  python "$PY" --run-at-utc-hour 0 --gate-minutes 5 >> "$LOG_DIR/live_run.log" 2>&1
