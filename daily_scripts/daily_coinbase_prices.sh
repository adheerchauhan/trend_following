#!/bin/bash
set -euo pipefail

# --- config you can edit ---
export PRICE_SNAPSHOT_DIR="${PRICE_SNAPSHOT_DIR:-$HOME/Documents/git/trend_following/data_floder/coinbase_daily}"  # where snapshots/parquet go
SCRIPT_DIR="$HOME/Documents/git/trend_following/daily_scripts/"                                                  # folder containing your .py script
SCRIPT_NAME="daily_coinbase_prices.py"
LOG_DIR="$HOME/Documents/git/trend_following/daily_scripts/logs"
LOCK_DIR="$HOME/.locks/coinbase_prices.lockdir"   # simple cross-platform lock

# --- (optional) conda/venv activation ---
# CONDA:
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate tf

# or venv:
# source "$HOME/.venvs/tf/bin/activate"

mkdir -p "$LOG_DIR"

# acquire a simple lock so overlapping cron fires don’t run concurrently
if mkdir "$LOCK_DIR" 2>/dev/null; then
  trap 'rmdir "$LOCK_DIR"' EXIT
else
  echo "$(date -u +"%F %T Z") skip: lock held" >> "$LOG_DIR/coinbase_prices.log"
  exit 0
fi

# run
cd "$SCRIPT_DIR"
python "$SCRIPT_NAME" >> "$LOG_DIR/coinbase_prices.log" 2>&1
