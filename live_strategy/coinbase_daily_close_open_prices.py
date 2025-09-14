#!/usr/bin/env python3
import os, time, json, pathlib, requests, datetime as dt
import pandas as pd

BASE = "https://api.coinbase.com/api/v3/brokerage"
HEADERS = {"Cache-Control": "no-cache", "User-Agent": "tf-daily-opens/1.0"}  # bypass 1s cache; be nice to servers
UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD"]  # edit as needed

# ---- storage config ----
BASE_DIR = pathlib.Path(os.environ.get("PRICE_SNAPSHOT_DIR", "~/Documents/git/trend_following/data_folder/coinbase_daily")).expanduser()
SNAP_DIR = BASE_DIR / "snapshots"
APPEND_PATH = BASE_DIR / "daily_opens_closes.parquet"
DONE_FLAG_DIR = BASE_DIR / "done_flags"       # to avoid double-runs

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
    end   = start + dt.timedelta(days=1)
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
        return js.get("candles", [])
    return []


def get_yday_daily(product_id, today_utc_date):
    yday = today_utc_date - dt.timedelta(days=1)
    s, e = bucket_bounds_utc(yday)
    candles = fetch_candles(product_id, s, e, "ONE_DAY")
    # Expect exactly one bucket; defensive parse:
    if not candles:
        return None
    c = candles[-1]
    return {
        "yday_start": dt.datetime.fromtimestamp(int(c["start"]), dt.timezone.utc),
        "yday_open": float(c["open"]),
        "yday_close": float(c["close"])
    }


def get_today_open(product_id, today_utc_date):
    s, e = bucket_bounds_utc(today_utc_date)
    # first 1m candle of the day
    candles = fetch_candles(product_id, s, s + dt.timedelta(minutes=1), "ONE_MINUTE")
    if not candles:
        return None
    c = candles[0]
    return {
        "today_open_time": dt.datetime.fromtimestamp(int(c["start"]), dt.timezone.utc),
        "today_open": float(c["open"])
    }


def main():
    now = get_server_time_utc()  # Coinbase server time (UTC)
    today_date = now.date()      # UTC date
    # Gate: only run between 00:01–00:15 UTC; else exit quietly.
    if not (now.hour == 0 and 1 <= now.minute <= 15):
        return

    flag = DONE_FLAG_DIR / f"{today_date}.done"
    if flag.exists():
        return  # already ran today

    rows = []
    for pid in UNIVERSE:
        yday = get_yday_daily(pid, today_date)
        # retry a couple times if the 1m candle isn't posted yet
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

    df = pd.DataFrame(rows)
    run_ts = dt.datetime.now(dt.timezone.utc)
    snap_name = f"opens_closes_{today_date}_{run_ts.strftime('%H%M%SZ')}.csv"
    out_path = SNAP_DIR / snap_name
    df.to_csv(out_path, index=False)

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
        print("Parquet append failed:", e)

    flag.touch()
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
