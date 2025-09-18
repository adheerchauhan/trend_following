#!/bin/bash
set -euo pipefail

# --- config you can edit ---
export PRICE_SNAPSHOT_DIR="${PRICE_SNAPSHOT_DIR:-$HOME/Library/Application Support/coinbase_daily}"  # where snapshots/parquet go
SCRIPT_DIR="$HOME/Documents/git/trend_following/live_strategy/"                                                  # folder containing your .py script
SCRIPT_NAME="coinbase_daily_close_open_prices.py"
LOG_DIR="$HOME/Documents/git/trend_following/daily_scripts/logs"
LOCK_DIR="$HOME/.locks/coinbase_prices.lockdir"   # simple cross-platform lock

PY="/opt/anaconda3/envs/crypto_prod/bin/python"   # <-- your env's Python

# --- (optional) conda/venv activation ---
# CONDA:
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate tf

# or venv:
# source "$HOME/.venvs/tf/bin/activate"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$LOCK_DIR")"   # ensure ~/.locks exists (parent only)

# acquire a simple lock so overlapping cron fires don’t run concurrently
if mkdir "$LOCK_DIR" 2>/dev/null; then
  trap 'rmdir "$LOCK_DIR"' EXIT
else
  echo "$(date -u +"%F %T Z") skip: lock held" >> "$LOG_DIR/coinbase_prices.log"
  exit 0
fi

# run
cd "$SCRIPT_DIR"
"$PY" "$SCRIPT_NAME" >> "$LOG_DIR/coinbase_prices.log" 2>&1
