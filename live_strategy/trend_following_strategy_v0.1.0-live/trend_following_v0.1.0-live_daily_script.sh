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

export PYTHONUNBUFFERED=1
"$CONDA_BIN" run -n crypto_prod --no-capture-output \
  python "$PY" --run-at-utc-hour 0 --gate-minutes 5 >> "$LOG_DIR/live_run.log" 2>&1
