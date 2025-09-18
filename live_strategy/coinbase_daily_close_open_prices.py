#!/usr/bin/env python3
import os, time, json, pathlib, requests
import pandas as pd
import sys
import datetime as dt

BASE = "https://api.coinbase.com/api/v3/brokerage"
HEADERS = {"Cache-Control": "no-cache", "User-Agent": "tf-daily-opens/1.0"}  # bypass 1s cache; be nice to servers
UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD"]  # edit as needed

# ---- storage config ----
BASE_DIR = pathlib.Path(
    os.environ.get(
        "PRICE_SNAPSHOT_DIR",
        str(pathlib.Path.home() / "Library" / "Application Support" / "coinbase_daily")
    )
).expanduser()
SNAP_DIR = BASE_DIR / "snapshots"
APPEND_PATH = BASE_DIR / "daily_opens_closes.parquet"
DONE_FLAG_DIR = BASE_DIR / "done_flags"  # to avoid double-runs

SNAP_DIR.mkdir(parents=True, exist_ok=True)
DONE_FLAG_DIR.mkdir(parents=True, exist_ok=True)


def get_server_time_utc():
    r = requests.get(f"{BASE}/time", timeout=10)
    r.raise_for_status()
    js = r.json()
    # API returns iso and epoch strings
    return dt.datetime.fromtimestamp(int(js["epochSeconds"]), dt.timezone.utc)


def epoch(dt_utc):
    return int(dt_utc.replace(tzinfo=dt.timezone.utc).timestamp())


def bucket_bounds_utc(day_utc):
    start = dt.datetime(day_utc.year, day_utc.month, day_utc.day, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    return start, end


def fetch_candles(product_id, start_utc, end_utc, granularity):
    params = {
        "start": str(epoch(start_utc)),
        "end": str(epoch(end_utc)),
        "granularity": granularity,  # ONE_DAY or ONE_MINUTE
        "limit": "350"
    }
    url = f"{BASE}/market/products/{product_id}/candles"
    for attempt in range(3):
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        js = resp.json()
        candles = js.get("candles", []) or []
        # Normalize order: oldest -> newest
        candles.sort(key=lambda c: int(c["start"]))
        return candles
    return []


def get_yday_daily(product_id, today_utc_date):
    """
    Return yesterday's daily open/close for product_id.

    Strategy:
      1) Ask for the ONE_DAY candle whose start == yesterday 00:00:00 UTC.
      2) If missing/empty, fall back to ONE_MINUTE candles over [00:00, 24:00):
         open = first 1m candle's open; close = last 1m candle's close.
    """
    yday = today_utc_date - dt.timedelta(days=1)
    y_start, y_end = bucket_bounds_utc(yday)
    y_start_epoch = epoch(y_start)
    y_end_epoch = epoch(y_end)

    # --- Primary: daily candle at exactly yesterday 00:00Z ---
    daily = fetch_candles(product_id, y_start, y_end + dt.timedelta(seconds=1), "ONE_DAY")  # tiny end buffer
    if daily:
        # ensure oldest→newest in case fetch_candles doesn't already do this
        daily.sort(key=lambda c: int(c["start"]))
        # exact midnight match first
        pick = next((c for c in daily if int(c["start"]) == y_start_epoch), None)
        if pick is None:
            # otherwise any candle that starts within [start, end)
            elig = [c for c in daily if y_start_epoch <= int(c["start"]) < y_end_epoch]
            pick = elig[0] if elig else None
        if pick is not None:
            return {
                "yday_start": dt.datetime.fromtimestamp(int(pick["start"]), dt.timezone.utc),
                "yday_open": float(pick["open"]),
                "yday_close": float(pick["close"]),
            }

    # --- Fallback: derive from 1-minute candles over the day ---
    one_min = fetch_candles(product_id, y_start, y_end, "ONE_MINUTE")
    if not one_min:
        return None  # nothing to do (exchange delay or network issue)

    one_min.sort(key=lambda c: int(c["start"]))  # oldest→newest
    first_c = one_min[0]
    last_c = one_min[-1]
    return {
        "yday_start": dt.datetime.fromtimestamp(int(first_c["start"]), dt.timezone.utc),
        "yday_open": float(first_c["open"]),
        "yday_close": float(last_c["close"]),
    }


def get_today_open(product_id, today_utc_date):
    s, e = bucket_bounds_utc(today_utc_date)
    s_epoch = epoch(s)
    deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=3)  # hard stop

    while dt.datetime.now(dt.timezone.utc) < deadline:
        candles = fetch_candles(product_id, s, s + dt.timedelta(minutes=2), "ONE_MINUTE")
        if candles:
            candles.sort(key=lambda c: int(c["start"])) # oldest->newest
            pick = next((c for c in candles if int(c["start"]) == s_epoch), None)
            if pick:
                return {
                    "today_open_time": dt.datetime.fromtimestamp(int(pick["start"]), dt.timezone.utc),
                    "today_open": float(pick["open"])
                }
        time.sleep(2)
    return None


def _log(msg: str, level: str = "INFO"):
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"{ts} [{level}] {msg}", flush=True)


def main():
    t0 = dt.datetime.now(dt.timezone.utc)

    now = get_server_time_utc()  # Coinbase server time (UTC)
    today_date = now.date()  # UTC date
    FORCE_RUN = os.environ.get("FORCE_RUN") == "1"

    # Gate: only run between 00:00-00:05 UTC unless FORCERUN flag is set
    if not FORCE_RUN and not (now.hour == 0 and 0 <= now.minute <= 5):
        _log(f"skip: outside UTC window (now={now.isoformat()}Z, force={FORCE_RUN})")
        return

    # Paths and done flag
    flag = DONE_FLAG_DIR / f"{today_date}.done"
    _log(
        f"start: date_utc={today_date} force={FORCE_RUN} "
        f"paths: SNAP_DIR='{SNAP_DIR}' APPEND_PATH='{APPEND_PATH}' DONE_FLAG_DIR='{DONE_FLAG_DIR}'"
    )
    if flag.exists():
        _log(f"skip: done-flag exists at {flag}")
        return  # already ran today

    # Build Dataframe
    rows = []
    for pid in UNIVERSE:
        yday = get_yday_daily(pid, today_date)

        # poll for 00:00 1m open price
        t_open = None
        for _ in range(6):  # up to ~3 minutes
            t_open = get_today_open(pid, today_date)
            if t_open:
                break
            time.sleep(30)

        rows.append({
            "product_id": pid,
            "date_utc": str(today_date),
            "yday_open": yday["yday_open"] if yday else None,
            "yday_close": yday["yday_close"] if yday else None,
            "today_open_utc": t_open["today_open"] if t_open else None,
        })

    # Write snapshot CSV
    df = pd.DataFrame(rows)
    run_ts = dt.datetime.now(dt.timezone.utc)
    snap_name = f"opens_closes_{today_date}_{run_ts.strftime('%H%M%SZ')}.csv"
    out_path = SNAP_DIR / snap_name
    df.to_csv(out_path, index=False)
    _log(f"snapshot: wrote {len(df)} rows to {out_path}")

    # keep an append-only parquet table for analytics
    try:
        if APPEND_PATH.exists():
            old = pd.read_parquet(APPEND_PATH)
            all_df = pd.concat([old, df], ignore_index=True)
            # drop duplicates on (date_utc, product_id)
            all_df = all_df.drop_duplicates(subset=["date_utc", "product_id"], keep="last")
            all_df.to_parquet(APPEND_PATH, index=False)
        else:
            df.to_parquet(APPEND_PATH, index=False)
    except Exception as e:
        # don't fail the run if parquet is unavailable
        _log(f"parquet: append failed: {e}", level="WARN")

    # Mark success
    flag.touch()

    elapsed = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
    _log(f"done: elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
