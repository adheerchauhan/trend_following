import pandas as pd
import json, os
from datetime import datetime, timedelta, timezone, date as date_cls
from pathlib import Path
from typing import Dict, Tuple, Optional

DEFAULT_COOLDOWN_DAYS = 7

# ---------- basic utils ----------
# def _as_utc_date(x) -> date_cls:
#     if isinstance(x, datetime):
#         return x.astimezone(timezone.utc).date()
#     return x  # assume it's already a date

def _as_utc_date(x):
    """
    Return a *date* in UTC terms.
    - If x is tz-aware: convert to UTC then take .date()
    - If x is tz-naive (your current setup): just take .date()
      (because you already pass normalized midnight timestamps / trading-day anchors)
    """
    if x is None:
        return None

    # If it's already a python date (not datetime), keep it
    if isinstance(x, date_cls) and not hasattr(x, "hour"):
        return x

    ts = pd.Timestamp(x)

    if ts.tz is None:
        # tz-naive: can't tz_convert; treat as already "day label"
        return ts.date()

    return ts.tz_convert("UTC").date()

def _iso_date(d: date_cls) -> str:
    return d.isoformat()

def _from_iso_date(s: str) -> date_cls:
    return datetime.fromisoformat(s).date()

# ---------- state file I/O ----------
def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            content = f.read().strip()
            if not content:  # empty file
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        # optionally quarantine the bad file so it doesn't keep breaking runs
        try:
            bad = path.with_suffix(".corrupt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            os.replace(path, bad)
            print(f"[warn] State file was invalid JSON. Moved to: {bad}")
        except Exception:
            # if we can't move it, just ignore and continue with empty state
            pass
        return {}

def save_state(path: Path, state: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)  # atomic on POSIX

# ---------- append-only event log (optional) ----------
def append_event_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

# ---------- API ----------
def start_cooldown(
    state_path: Path,
    ticker: str,
    breach_date,
    cooldown_counter_threshold: int = DEFAULT_COOLDOWN_DAYS,
    note: str = "",
    log_path: Optional[Path] = None
) -> dict:
    """Start/refresh cooldown from breach_date (UTC date)."""
    state = load_state(state_path)
    bdate = _as_utc_date(breach_date)
    until = bdate + timedelta(days=cooldown_counter_threshold)  # buys allowed on/after 'until'
    rec = {
        "last_breach_date": _iso_date(bdate),
        "cooldown_until":   _iso_date(until),
        "note":             note or "stop_breached"
    }
    state[ticker] = rec
    save_state(state_path, state)

    if log_path:
        append_event_log(log_path, {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "event": "start_cooldown",
            "breach_date": _iso_date(bdate),
            "cooldown_until": _iso_date(until),
            "note": note,
        })
    return rec

def is_in_cooldown(state_path: Path, ticker: str, today) -> Tuple[bool, int]:
    """Return (active, days_remaining). days_remaining >= 0 while active."""
    state = load_state(state_path)
    rec = state.get(ticker)
    if not rec:
        return False, 0
    today_d = _as_utc_date(today)
    until_d = _from_iso_date(rec["cooldown_until"])
    if today_d < until_d:
        return True, (until_d - today_d).days
    return False, 0
