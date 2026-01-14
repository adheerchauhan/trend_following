# Import all the necessary modules
import os, sys
# from .../research/notebooks -> go up two levels to repo root
repo_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
import numpy as np
import math
import datetime
from datetime import datetime, timezone, timedelta
from typing import Iterable
import argparse
import json
import traceback
import itertools
import ast
from strategy_signal.trend_following_signal import (
    get_trend_donchian_signal_for_portfolio_with_rolling_r_sqr_vol_of_vol
)
from portfolio.strategy_performance import (calculate_sharpe_ratio, calculate_calmar_ratio, calculate_CAGR, calculate_risk_and_performance_metrics,
                                          calculate_compounded_cumulative_returns, estimate_fee_per_trade, rolling_sharpe_ratio)
from utils import coinbase_utils as cn
from portfolio import strategy_performance as perf
from sizing import position_sizing_binary_utils as size_bin
from sizing import position_sizing_continuous_utils as size_cont
from utils import stop_loss_cooldown_state as state
from strategy_signal import trend_following_signal as tf
from pathlib import Path
import yaml
import uuid
from trend_following_email_summary_v020 import send_summary_email


STATE_DIR = Path("/Users/adheerchauhan/Documents/git/trend_following/live_strategy/trend_following_strategy_v0.2.0-live/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
COOLDOWN_STATE_FILE = STATE_DIR / "stop_loss_breach_cooldown_state.json"
COOLDOWN_LOG_FILE   = STATE_DIR / "stop_loss_breach_cooldown_log.jsonl"
DONE_FLAG_DIR       = STATE_DIR / "done_flags"
RUN_LOG             = STATE_DIR / "live_run.log"

# JSONL log files
LIVE_ERRORS_LOG       = STATE_DIR / "live_errors.jsonl"
HEARTBEAT_LOG         = STATE_DIR / "heartbeat.jsonl"
DESIRED_TRADES_LOG    = STATE_DIR / "desired_trades_log.jsonl"
ORDER_BUILD_LOG       = STATE_DIR / "order_build_log.jsonl"
ORDER_SUBMIT_LOG      = STATE_DIR / "order_submit_log.jsonl"
DUST_BUILD_LOG        = STATE_DIR / "dust_build_log.jsonl"
DUST_SUBMIT_LOG       = STATE_DIR / "dust_submit_log.jsonl"
STOP_UPDATE_LOG       = STATE_DIR / "stop_update_log.jsonl"
DAILY_SUMMARY_DIR = STATE_DIR / "daily_summaries"
DAILY_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_RUN_ID = None

def new_run_id(now_utc=None) -> str:
    now_utc = now_utc or utc_now()
    return now_utc.strftime("%Y-%m-%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

def set_run_id(run_id: str):
    global CURRENT_RUN_ID
    CURRENT_RUN_ID = run_id

def daily_summary_path(day) -> Path:
    return DAILY_SUMMARY_DIR / f"daily_summary_{day.isoformat()}.json"


def utc_now():
    return datetime.now(timezone.utc)


def _utc_ts(d):
    """UTC tz-aware Timestamp (00:00Z if 'd' is a date)."""
    ts = pd.Timestamp(d)
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-run", action="store_true")
    ap.add_argument("--run-at-utc-hour", type=int, default=0)
    ap.add_argument("--gate-minutes", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


def write_jsonl(path: Path, obj: dict):
    """
    Append a JSON line to `path`. If this is the first heartbeat of a run
    (event == 'config_loaded'), prepend a blank line and a human-readable
    header with date/time to visually separate runs. If it's 'run_complete',
    also add a footer and a trailing blank line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_heartbeat = isinstance(obj, dict) and ("event" in obj)

    # Detect start/end markers from your existing log_event() usage
    is_run_start = is_heartbeat and obj.get("event") == "config_loaded"
    is_run_end   = is_heartbeat and obj.get("event") == "run_complete"

    # Build optional header/footer lines
    header = ""
    footer = ""
    if is_run_start:
        # Add a blank line only if file already has content
        needs_blank = path.exists() and path.stat().st_size > 0
        ts = obj.get("ts") or datetime.now(timezone.utc).isoformat()
        title = obj.get("portfolio") or "run"
        header = (
            ("\n" if needs_blank else "")
            + f"# ===== RUN START: {ts} — {title} =====\n"
        )
    elif is_run_end:
        ts = obj.get("ts") or datetime.now(timezone.utc).isoformat()
        footer = f"# ----- RUN END:   {ts} -----\n\n"

    if isinstance(obj, dict) and CURRENT_RUN_ID and "run_id" not in obj:
        obj = dict(obj)
        obj["run_id"] = CURRENT_RUN_ID

    line = json.dumps(obj, default=str) + "\n"

    with open(path, "a") as f:
        if header:
            f.write(header)
        f.write(line)
        if footer:
            f.write(footer)


def log_event(kind: str, **payload):
    write_jsonl(HEARTBEAT_LOG, {"ts": utc_now_iso(), "event": kind, **payload})


def write_daily_summary(
    *,
    cfg: dict,
    day,
    run_id: str,
    dry_run: bool,
    started_at: str,
    completed_at: str,
    df: pd.DataFrame,
    ticker_list: list,
    ticker_to_sleeve: dict,
    desired_positions: dict,
    current_positions: dict,
    rebalance_orders: list,
    dust_orders: list,
    stop_results: dict,
    errors: list,
):

    # Use today's row (fallback last row)
    date_ts = pd.Timestamp(day).normalize()
    pos = df.index.searchsorted(date_ts, side="left")
    date = df.index[min(pos, len(df.index) - 1)]

    # Portfolio headline only (lean)
    portfolio = {
        "total_portfolio_value": float(df.loc[date, "total_portfolio_value"]),
        "available_cash": float(df.loc[date, "available_cash"]),
        "daily_portfolio_volatility": float(df.loc[date, "daily_portfolio_volatility"]),
        "final_scaling_factor": float(df.loc[date, "final_scaling_factor"]),
        "total_target_notional": float(df.loc[date, "total_target_notional"]),
        "total_actual_position_notional": float(df.loc[date, "total_actual_position_notional"]),
        "count_of_positions": int(df.loc[date, "count_of_positions"]),
    }

    # Sleeves: keep earlier format, but accurate + explicit
    sleeves = {}
    for sname, sconf in cfg["universe"]["sleeves"].items():
        tickers = sconf.get("tickers", [])

        alloc_share = 0.0      # post-scale (actual portfolio allocation share)
        pre_scale_sum = 0.0    # sum of sleeve_risk_adj_weights (optimizer output space)
        mults = []

        for t in tickers:
            col_alloc = f"{t}_target_vol_normalized_weight"
            if col_alloc in df.columns:
                v = df.loc[date, col_alloc]
                if pd.notna(v):
                    alloc_share += float(v)

            col_rb = f"{t}_sleeve_risk_adj_weights"
            if col_rb in df.columns:
                v = df.loc[date, col_rb]
                if pd.notna(v):
                    pre_scale_sum += float(v)

            # col_m = f"{t}_sleeve_risk_multiplier"
            # if col_m in df.columns:
            #     v = df.loc[date, col_m]
            #     if pd.notna(v):
            #         mults.append(float(v))

        sleeves[sname] = {
            "budget_weight": float(sconf.get("weight", 0.0)),
            "allocation_share": float(alloc_share),             # sum of final weights in sleeve
            "pre_scale_weight_sum": float(pre_scale_sum),       # sum of rb_weights in sleeve
            # "avg_multiplier": float(np.mean(mults)) if mults else None,
        }

    # Tickers: only fields needed for email (same idea, just cleaned names)
    tickers_out = {}
    for t in ticker_list:
        dp = desired_positions.get(t, {}) or {}
        cp = current_positions.get(t, {}) or {}

        tickers_out[t] = {
            "sleeve": ticker_to_sleeve.get(t),

            # The signal (pre-risk-budget)
            "vol_adj_weight": float(df.loc[date, f"{t}_vol_adjusted_trend_signal"])
            if f"{t}_vol_adjusted_trend_signal" in df.columns else None,

            # Optimizer output (pre-scale)
            "sleeve_risk_adj_weight": float(df.loc[date, f"{t}_sleeve_risk_adj_weights"])
            if f"{t}_sleeve_risk_adj_weights" in df.columns else None,

            # Final Scaled Weights
            "final_weight": float(df.loc[date, f"{t}_target_vol_normalized_weight"])
            if f"{t}_target_vol_normalized_weight" in df.columns else None,

            "sleeve_multiplier": float(df.loc[date, f"{t}_sleeve_risk_multiplier"])
            if f"{t}_sleeve_risk_multiplier" in df.columns else None,

            "target_notional": float(df.loc[date, f"{t}_target_notional"])
            if f"{t}_target_notional" in df.columns else None,

            "open_notional": float(df.loc[date, f"{t}_open_position_notional"])
            if f"{t}_open_position_notional" in df.columns else None,

            "trade_notional": float(dp.get("new_trade_notional", 0.0)),
            "trade_fees_est": float(dp.get("trade_fees", 0.0)),
            "trade_reason": dp.get("reason"),

            "stop": stop_results.get(t),

            "mid_price": float(cp.get("ticker_mid_price")) if cp.get("ticker_mid_price") is not None else None,
            "pos_qty": float(cp.get("ticker_qty")) if cp.get("ticker_qty") is not None else None,
        }

    summary = {
        "run": {
            "run_id": run_id,
            "date": str(day),
            "dry_run": bool(dry_run),
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "portfolio": portfolio,
        "sleeves": sleeves,
        "tickers": tickers_out,
        "orders": {
            "rebalance_built_count": len(rebalance_orders or []),
            "dust_built_count": len(dust_orders or []),
        },
        "errors": errors or [],
    }

    out = daily_summary_path(day)
    out.parent.mkdir(parents=True, exist_ok=True)
    # out.write_text(json.dumps(summary, indent=2, default=str))
    out.write_text(json.dumps(summary, separators=(",", ":"), default=str))

    return out


## Load Config file for the strategy
def load_prod_strategy_config(strategy_version='v0.2.0'):
    # nb_cwd = Path.cwd()  # git/trend_following/research/notebooks
    # config_path = (
    #         nb_cwd.parents[1]  # -> git/trend_following
    #         / "live_strategy"
    #         / f"trend_following_strategy_{strategy_version}-live"
    #         / "config"
    #         / f"trend_strategy_config_{strategy_version}.yaml"
    # )

    here = Path(__file__).resolve().parent
    cfg_dir = Path(os.getenv("TF_CONFIG_DIR", here / "config"))

    fname = f"trend_strategy_config_{strategy_version}.yaml"
    config_path = cfg_dir / fname

    print(config_path)  # sanity check
    print(config_path.exists())  # should be True

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


## Generate weighted and scaled final signal
def get_strategy_trend_signal(cfg):
    end_date = utc_now().date()
    start_date = end_date - pd.Timedelta(days=cfg['run']['warmup_days'])

    # Build kwargs directly from cfg sections
    sig_kwargs = {
        # Dates
        "start_date": start_date,
        "end_date": end_date,

        # Universe
        "ticker_list": cfg["universe"]["tickers"],

        # Moving Average Signal
        "fast_mavg": cfg["signals"]["moving_average"]["fast_mavg"],
        "slow_mavg": cfg["signals"]["moving_average"]["slow_mavg"],
        "mavg_stepsize": cfg["signals"]["moving_average"]["mavg_stepsize"],
        "mavg_z_score_window": cfg["signals"]["moving_average"]["mavg_z_score_window"],

        # Donchain Channel Signal
        "entry_rolling_donchian_window": cfg["signals"]["donchian"]["entry_rolling_donchian_window"],
        "exit_rolling_donchian_window": cfg["signals"]["donchian"]["exit_rolling_donchian_window"],
        "use_donchian_exit_gate": cfg["signals"]["donchian"]["use_donchian_exit_gate"],

        # Signal Weights
        "ma_crossover_signal_weight": cfg["signals"]["weighting"]["ma_crossover_signal_weight"],
        "donchian_signal_weight": cfg["signals"]["weighting"]["donchian_signal_weight"],
        "weighted_signal_ewm_window": cfg["signals"]["weighting"]["weighted_signal_ewm_window"],
        "rolling_r2_window": cfg["signals"]["filters"]["rolling_r2"]["rolling_r2_window"],

        # Rolling R Squared Filter
        "lower_r_sqr_limit": cfg["signals"]["filters"]["rolling_r2"]["lower_r_sqr_limit"],
        "upper_r_sqr_limit": cfg["signals"]["filters"]["rolling_r2"]["upper_r_sqr_limit"],
        "r2_smooth_window": cfg["signals"]["filters"]["rolling_r2"]["r2_smooth_window"],
        "r2_confirm_days": cfg["signals"]["filters"]["rolling_r2"]["r2_confirm_days"],

        # Vol of Vol Filter
        "log_std_window": cfg["signals"]["filters"]["vol_of_vol"]["log_std_window"],
        "coef_of_variation_window": cfg["signals"]["filters"]["vol_of_vol"]["coef_of_variation_window"],
        "vol_of_vol_z_score_window": cfg["signals"]["filters"]["vol_of_vol"]["vol_of_vol_z_score_window"],
        "vol_of_vol_p_min": cfg["signals"]["filters"]["vol_of_vol"]["vol_of_vol_p_min"],
        "r2_strong_threshold": cfg["signals"]["filters"]["rolling_r2"]["r2_strong_threshold"],

        # Signal & Data Parameters
        "use_activation": cfg["signals"]["activation"]["use_activation"],
        "tanh_activation_constant_dict": cfg["signals"]["activation"]["tanh_activation_constant_dict"],
        "moving_avg_type": cfg["data"]["moving_avg_type"],
        "long_only": cfg["run"]["long_only"],
        "price_or_returns_calc": cfg["data"]["price_or_returns_calc"],
        "use_coinbase_data": cfg["data"]["use_coinbase_data"],
        "use_saved_files": False,
        "saved_file_end_date": None  # cfg["data"]["saved_file_end_date"]
    }

    df_trend = get_trend_donchian_signal_for_portfolio_with_rolling_r_sqr_vol_of_vol(**sig_kwargs)

    print('Generating Volatility Adjusted Trend Signal!!')
    ## Get Volatility Adjusted Trend Signal
    df_signal = size_cont.get_volatility_adjusted_trend_signal_continuous(df_trend,
                                                                          ticker_list=cfg['universe']['tickers'],
                                                                          volatility_window=cfg['risk_and_sizing'][
                                                                              'volatility_window'],
                                                                          annual_trading_days=cfg['run'][
                                                                              'annual_trading_days'])

    return df_signal


def calculate_average_true_range_live(date, ticker, rolling_atr_window=20):
    end_date = date
    start_date = date - pd.Timedelta(days=(rolling_atr_window + 200))
    df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                        end_date=end_date, save_to_file=False,
                                                        portfolio_name='Trend Following')

    # Make sure index is UTC tz-aware and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    df = df.sort_index()

    df.columns = [f'{ticker}_{x}' for x in df.columns]

    ## Get T-1 Close Price
    df[f'{ticker}_t_1_close'] = df[f'{ticker}_close'].shift(1)

    # Calculate the True Range (TR) and Average True Range (ATR)
    df[f'{ticker}_high-low'] = df[f'{ticker}_high'] - df[f'{ticker}_low']
    df[f'{ticker}_high-close'] = np.abs(df[f'{ticker}_high'] - df[f'{ticker}_close'].shift(1))
    df[f'{ticker}_low-close'] = np.abs(df[f'{ticker}_low'] - df[f'{ticker}_close'].shift(1))
    df[f'{ticker}_true_range_price'] = df[
        [f'{ticker}_high-low', f'{ticker}_high-close', f'{ticker}_low-close']].max(axis=1)
    df[f'{ticker}_{rolling_atr_window}_avg_true_range_price'] = df[f'{ticker}_true_range_price'].ewm(
        span=rolling_atr_window, adjust=False).mean()

    ## Shift by 1 to avoid look-ahead bias
    df[f'{ticker}_{rolling_atr_window}_avg_true_range_price'] = df[
        f'{ticker}_{rolling_atr_window}_avg_true_range_price'].shift(1)

    return df


def chandelier_stop_long(date, ticker, highest_high_window, rolling_atr_window, atr_multiplier):
    ## Get Average True Range
    df_atr = calculate_average_true_range_live(date=date, ticker=ticker, rolling_atr_window=rolling_atr_window)
    key = _utc_ts(date).floor('D')
    atr_col = f'{ticker}_{rolling_atr_window}_avg_true_range_price'
    high_col = f'{ticker}_high'

    # As-of ATR and highest high (safe if 'key' isn’t present yet)
    atr_series = df_atr[atr_col].loc[:key]
    if atr_series.empty:
        raise ValueError(f"No ATR data on or before {key} for {ticker}.")
    atr = float(atr_series.iloc[-1])

    ## Get the Highest High from previous date
    highest_high_t_1_series = df_atr[high_col].rolling(highest_high_window).max().shift(1).loc[:key]
    if highest_high_t_1_series.empty:
        raise ValueError(f"No price data on or before {key} for {ticker}.")
    highest_high_t_1 = float(highest_high_t_1_series.iloc[-1])
    chandelier_stop = highest_high_t_1 - atr_multiplier * atr

    return chandelier_stop


def refresh_cooldowns_from_stop_fills(
    client,
    tickers: Iterable[str],
    today_date,
    state_file: Path,
    log_file: Path,
    get_stop_fills_fn=cn.get_stop_fills,         # inject coinbase_utils.get_stop_fills
    cooldown_counter_threshold: int = 7,
    lookback_days: int = 10,   # scan recent window for safety
    effective_recent_days: int = 2  # treat fills in last N days as trigger (covers “not obvious for 4–5 days” in time series)
):
    """
    - Pull STOP fills over a lookback window.
    - If any fill is within the last 'effective_recent_days' (e.g., yesterday/today), start/refresh cooldown.
    """
    fills_start = today_date - timedelta(days=lookback_days)
    recent_cutoff = today_date - timedelta(days=effective_recent_days - 1)  # e.g., if N=2, cutoff is (today - 1)

    for ticker in tickers:
        try:
            fills = get_stop_fills_fn(
                client=client,
                product_id=ticker,
                start=fills_start,
                end=today_date,
                client_id_prefix="stop-"
            )
            print(fills)
            # 'fills' is [(ts, price), ...] sorted ascending
            for ts, px in reversed(fills):
                # recent STOP fill ⇒ start cooldown dated to fill date
                if ts.date() >= recent_cutoff:
                    print('date > recent_cutoff')
                    state.start_cooldown(
                        state_path=state_file,
                        ticker=ticker,
                        breach_date=ts.date(),
                        cooldown_counter_threshold=cooldown_counter_threshold,
                        note=f"stop_fill@{px}",
                        log_path=log_file
                    )
                    break  # only the most recent fill matters
        except Exception as e:
            print(f"[warn] refresh_cooldowns_from_stop_fills({ticker}) failed: {e}", flush=True)


def build_rebalance_orders(desired_positions, date, current_positions, client, order_type, limit_price_buffer=0.0):
    """
    Build market orders for new trades. You may switch to limit orders.
    Returns a list of dicts: {product_id, side, type, size, client_order_id}
    """
    orders = []
    today_str = date.strftime("%Y%m%d")

    for ticker, d in desired_positions.items():
        raw_notional = float(d.get("new_trade_notional", 0.0))
        if abs(raw_notional) < 1e-9:
            continue

        side = 'BUY' if raw_notional > 0 else 'SELL'
        mid_px = current_positions[ticker]['ticker_mid_price']
        if not (np.isfinite(mid_px) and mid_px > 0):
            continue

        prod_specs = cn.get_product_meta(client, ticker)
        base_inc = prod_specs["base_increment"]
        base_min = prod_specs["base_min_size"]
        quote_min = prod_specs["quote_min_size"]  # min notional in quote currency
        price_inc = prod_specs["price_increment"]

        # Size in BASE currency: |notional| / price
        raw_size = abs(raw_notional) / mid_px

        # Quantize size to base_increment
        q_size = cn.round_down(raw_size, base_inc)

        # Enforce base_min_size
        if q_size < base_min:
            # Try rounding up once (only for buys), else skip as dust
            if side == "BUY":
                q_size_up = cn.round_up(raw_size, base_inc)
                if q_size_up >= base_min:
                    q_size = q_size_up
                else:
                    # still dust, skip
                    continue
            else:
                # sell smaller than min size => skip (or you may aggregate to a later day)
                continue

        # Enforce quote_min_size (notional)
        q_notional = q_size * mid_px
        if quote_min and q_notional < quote_min:
            # For buys, see if one more increment clears the bar
            if side == "BUY":
                q_size_up = cn.round_up((quote_min / mid_px), base_inc)
                if q_size_up * mid_px >= quote_min:
                    q_size = q_size_up
                    q_notional = q_size * mid_px
                else:
                    continue
            else:
                # for sells, if position leftover < min you might need to fully close later; skip now
                continue

        # Prepare order dict
        cl_order_id = f"{ticker}-{today_str}-rebalance-{uuid.uuid4().hex[:8]}"

        order = {
            "product_id": ticker,
            "side": side,
            "type": order_type,  # "market" (default) or "limit"
            "size": float(q_size),  # BASE units
            "client_order_id": cl_order_id,
        }

        # Coinbase Advanced Trade API expects "order_configuration"
        if order_type == "market":
            order_configuration = {
                "market_market_ioc": {
                    "base_size": str(float(q_size))  # base units as string
                }
            }
        elif order_type == "limit":
            if side == "BUY":
                px = mid_px * (1 + abs(limit_price_buffer))
            else:
                px = mid_px * (1 - abs(limit_price_buffer))
            if price_inc:
                px = cn.round_to_increment(px, price_inc)

            order_configuration = {
                "limit_limit_gtc": {
                    "base_size": str(float(q_size)),
                    "limit_price": str(float(px))
                }
            }
        else:
            raise ValueError(f"Unsupported order_type: {order_type}")

        orders.append({
            "client_order_id": cl_order_id,
            "product_id": ticker,
            "side": side,
            "order_configuration": order_configuration,
        })

    return orders


def submit_daily_rebalance_orders(client, orders, *, preview=True):
    results = []
    for od in orders:
        try:
            if preview:
                r = client.preview_order(
                    product_id=od["product_id"],
                    side=od["side"],
                    order_configuration=od["order_configuration"],
                )
            else:
                r = client.create_order(
                    client_order_id=od["client_order_id"],
                    product_id=od["product_id"],
                    side=od["side"],
                    order_configuration=od["order_configuration"],
                )
            results.append({"ok": True, "request": od, "response": cn._as_dict(r)})
        except Exception as e:
            results.append({"ok": False, "request": od, "error": f"{type(e).__name__}: {e}"})
    return results


# --- Helpers ----------------------------------------------------------------
def _long_stop_for_today(date, ticker, highest_high_window, rolling_atr_window, atr_multiplier):
    """Chandelier stop (long) for *today*, computed from live ATR & T-1 highest high."""
    return float(chandelier_stop_long(
        date=date,
        ticker=ticker,
        highest_high_window=highest_high_window,
        rolling_atr_window=rolling_atr_window,
        atr_multiplier=atr_multiplier
    ))


def _entry_allowed_long(curr_price, stop_today, eps=0.0):
    """
    Gate for NEW/ADDED long risk:
    - allow only if current price is strictly above today's stop (plus optional eps).
    """
    return np.isfinite(curr_price) and np.isfinite(stop_today) and (curr_price > stop_today * (1 + eps))


## Calculate a multiplier by date that when applied to the volatility adjusted signals, brings the
## risk contribution of each sleeve close to the allocated Risk Budget per sleeve
def risk_budget_by_sleeve_optimized_by_signal(signals, daily_cov_matrix, ticker_list, ticker_to_sleeve, sleeve_budgets,
                                              max_iter=100, tol=1e-5, step=0.5,
                                              min_signal_eps=1e-4,
                                              mode="cap"
                                              # "cap" = treat budgets as max, "target" = your current behaviour
                                              ):
    """
    signals: 1d np.array of base weights (vol-adjusted signals), length N
    daily_cov_matrix: NxN covariance matrix
    ticker_to_sleeve: dict {ticker: sleeve_name}
    sleeve_budgets:   dict {sleeve_name: {'weight': target_risk_share}}

    mode:
      - "cap": only reduce risk of sleeves whose risk_share > budget
      - "target": symmetric adjustment (your original behaviour)
    """
    ## Convert signals and covariance matrix to floating point numbers
    signals = np.asarray(signals, dtype=float)
    daily_cov_matrix = np.asarray(daily_cov_matrix, dtype=float)

    ## Get a list of all Sleeves in the Portfolio
    sleeves = list(sleeve_budgets.keys())

    ## Start with a Multiplier of 1 for Each Sleeve
    risk_multiplier = {s: 1.0 for s in sleeves}

    ## Identify active sleeves based on signal magnitude
    sleeve_signal_abs = {s: 0.0 for s in sleeves}
    for i, t in enumerate(ticker_list):
        s = ticker_to_sleeve[t]
        sleeve_signal_abs[s] += abs(signals[i])
    active_sleeves = {s for s in sleeves if sleeve_signal_abs[s] > min_signal_eps}

    ## If no sleeve has any signal, just return zeros
    if not active_sleeves:
        return np.zeros_like(signals, dtype=float), risk_multiplier

    ## If only one sleeve is active, no need for optimization – just use original signals
    if len(active_sleeves) == 1:
        return np.asarray(signals, dtype=float), risk_multiplier

    ## Re-normalize budgets over active sleeves only (optional but helpful)
    total_budget_active = sum(sleeve_budgets[s]['weight'] for s in active_sleeves)
    eff_budget = {}
    for s in sleeves:
        if s in active_sleeves and total_budget_active > 0:
            eff_budget[s] = sleeve_budgets[s]['weight'] / total_budget_active
        else:
            eff_budget[s] = 0.0

    ## Iterate through to calculate the multiplier to minimize the max absolute error between the
    ## allocated and actual risk budgets per sleeve
    for _ in range(max_iter):
        ## Build weights per ticker multiplying the vol adjusted signal with the current multiplier
        sleeve_risk_adj_weights = np.array(
            [signals[i] * risk_multiplier[ticker_to_sleeve[t]] for i, t in enumerate(ticker_list)])

        ## If all signals are zero return
        if np.allclose(sleeve_risk_adj_weights, 0):
            return sleeve_risk_adj_weights, risk_multiplier

        ## Calculate the Portfolio Variance and Standard Deviation for today given the current weights
        sigma2 = float(sleeve_risk_adj_weights @ daily_cov_matrix @ sleeve_risk_adj_weights)
        sigma = np.sqrt(sigma2)

        ## Calculate the Marginal Risk Contribution to Portfolio Risk
        ## This calculates the unit change in the portfolio volatility for a unit change in the weight of a coin
        ## This is essentially the first derivative of portfolio weight wrt signal weight w
        ## σ(w) = sqrt(w⊤Σw)
        ## ∂σ/∂w = 1 / (2*sqrt(w⊤Σw)) * 2Σw = Σw / σ
        marginal_risk_unit = daily_cov_matrix @ sleeve_risk_adj_weights / sigma  # length N

        ## This calculates the risk contribution per coin
        ## This metric shows how much of the portfolio volatility comes from each coin in the portfolio
        ## Important Property: Sum of the risk contribution of all coins equals the portfolio volatility
        ## RCi = wi * marginali = wi * (Σw)i / σ
        risk_contribution_coin = sleeve_risk_adj_weights * marginal_risk_unit  # length N

        ## Calculate sleeve level volatility metrics and risk share
        ## RCs = i∈s∑ RCi
        risk_contribution_sleeve = {s: 0.0 for s in sleeves}
        for i, t in enumerate(ticker_list):
            s = ticker_to_sleeve[t]
            risk_contribution_sleeve[s] += risk_contribution_coin[i]

        ## Calculate percent of risk contribution per sleeve or risk share
        ## s∑risk_shares = 1/σ * ∑RCs = 1/σ * i∑RCi = 1
        risk_share_sleeve = {s: risk_contribution_sleeve[s] / sigma for s in sleeves}

        ## Calculate the maximum absolute difference between actual risk share per sleeve and desired risk budget
        ## If the maximum absolute difference is below the tolerance threshold, we exit and keep the calculated multiplier
        # risk_allocation_error = max(abs(risk_share_sleeve[s] - sleeve_budgets[s]['weight']) for s in sleeves)
        risk_allocation_error = max(abs(risk_share_sleeve[s] - eff_budget[s]) for s in active_sleeves)
        if risk_allocation_error < tol:
            break

        ## Multiplicative Update:
        ## If a sleeve has too much risk, we shrink its multiplier
        ### For too much risk, desired risk budget / actual risk per sleeve is less than 1
        ### We square this fraction by the step leading to a smaller multiplier for the next iteration

        ## If a sleeve has too little risk, we grow its multiplier
        ### For too little risk, desired risk budget / actual risk per sleeve is greater than 1
        ### We square this fraction by the step leading to a larger multiplier for the next iteration
        for s in active_sleeves:
            rs = risk_share_sleeve[s]
            b = eff_budget[s]
            if rs <= 0:
                continue

            if mode == 'target':
                ratio = b / rs
                risk_multiplier[s] *= ratio ** step

            elif mode == 'cap':
                ## only shrink sleeves that are ABOVE their budget
                if rs > b and b > 0:
                    ratio = b / rs
                    risk_multiplier[s] *= ratio ** step
                ## if rs <= b, leave multiplier unchanged (don't lever up weak sleeves)

    ## Final Sleeve Risk Adjusted Weights
    sleeve_risk_adj_weights = np.array([signals[i] * risk_multiplier[ticker_to_sleeve[t]]
                                        for i, t in enumerate(ticker_list)], dtype=float)

    return sleeve_risk_adj_weights, risk_multiplier


def get_target_volatility_position_sizing_with_risk_multiplier_sleeve_weights_opt(
        df, cov_matrix, date, ticker_list,
        daily_target_volatility,
        total_portfolio_value_upper_limit,
        ticker_to_sleeve, sleeve_budgets,
        risk_max_iterations, risk_sleeve_budget_tolerance,
        risk_optimizer_step, risk_min_signal, sleeve_risk_mode
):
    ## Scale weights of positions to ensure the portfolio is in line with the target volatility
    unscaled_weight_cols = [f'{ticker}_vol_adjusted_trend_signal' for ticker in ticker_list]
    scaled_weight_cols = [f'{ticker}_target_vol_normalized_weight' for ticker in ticker_list]
    target_notional_cols = [f'{ticker}_target_notional' for ticker in ticker_list]
    t_1_price_cols = [f'{ticker}_t_1_close' for ticker in ticker_list]
    returns_cols = [f'{ticker}_t_1_close_pct_returns' for ticker in ticker_list]
    sleeve_risk_multiplier_cols = [f'{ticker}_sleeve_risk_multiplier' for ticker in ticker_list]
    sleeve_risk_adj_cols = [f'{ticker}_sleeve_risk_adj_weights' for ticker in ticker_list]

    if date not in df.index or date not in cov_matrix.index:
        raise ValueError(f"Date {date} not found in DataFrame or covariance matrix index.")

    ## Iterate through each day and get the unscaled weights and calculate the daily covariance matrix
    daily_weights = df.loc[date, unscaled_weight_cols].values
    daily_cov_matrix = cov_matrix.loc[date].loc[returns_cols, returns_cols].values

    ## Apply the Sleeve Risk Adjusted Multiplier to the Daily Weights
    if ticker_to_sleeve is not None and sleeve_budgets is not None:
        rb_weights, sleeve_risk_multiplier = risk_budget_by_sleeve_optimized_by_signal(
            signals=daily_weights,
            daily_cov_matrix=daily_cov_matrix,
            ticker_list=ticker_list,
            ticker_to_sleeve=ticker_to_sleeve,
            sleeve_budgets=sleeve_budgets,
            max_iter=risk_max_iterations,
            tol=risk_sleeve_budget_tolerance,
            step=risk_optimizer_step,
            min_signal_eps=risk_min_signal,
            mode=sleeve_risk_mode
        )
        sleeve_risk_multiplier = np.array(
            [float(sleeve_risk_multiplier.get(ticker_to_sleeve[t], 1.0)) for t in ticker_list],
            dtype=float
        )
    else:
        rb_weights = daily_weights.copy()
        sleeve_risk_multiplier = np.ones(len(ticker_list))
    rb_weights = np.asarray(rb_weights, dtype=float)
    rb_weights = np.clip(rb_weights, 0.0, None)
    sleeve_risk_multiplier = np.asarray(sleeve_risk_multiplier, dtype=float)
    df.loc[date, sleeve_risk_adj_cols] = rb_weights
    df.loc[date, sleeve_risk_multiplier_cols] = sleeve_risk_multiplier

    ## If all weights are zero, we can just zero out and return
    if np.allclose(rb_weights, 0):
        df.loc[date, scaled_weight_cols] = 0.0
        df.loc[date, target_notional_cols] = 0.0
        df.loc[date, 'daily_portfolio_volatility'] = 0.0
        df.loc[date, 'target_vol_scaling_factor'] = 0.0
        df.loc[date, 'cash_scaling_factor'] = 1.0
        df.loc[date, 'final_scaling_factor'] = 0.0
        df.loc[date, 'total_target_notional'] = 0.0
        return df

    ## Calculate the portfolio volatility based on the new weights
    daily_portfolio_volatility = size_bin.calculate_portfolio_volatility(rb_weights, daily_cov_matrix)
    df.loc[date, 'daily_portfolio_volatility'] = daily_portfolio_volatility
    if daily_portfolio_volatility > 0:
        vol_scaling_factor = daily_target_volatility / daily_portfolio_volatility
    else:
        vol_scaling_factor = 0

    ## Apply Scaling Factor with No Leverage
    gross_weight_sum = np.sum(np.abs(rb_weights))
    cash_scaling_factor = 1.0 / np.maximum(gross_weight_sum, 1e-12)  # ∑ w ≤ 1  (long‑only)
    final_scaling_factor = min(vol_scaling_factor, cash_scaling_factor)

    df.loc[date, 'target_vol_scaling_factor'] = vol_scaling_factor
    df.loc[date, 'cash_scaling_factor'] = cash_scaling_factor
    df.loc[date, 'final_scaling_factor'] = final_scaling_factor

    # Scale the weights to target volatility
    scaled_weights = rb_weights * final_scaling_factor
    df.loc[date, scaled_weight_cols] = scaled_weights

    ## Calculate the target notional and size
    target_notionals = scaled_weights * total_portfolio_value_upper_limit
    df.loc[date, target_notional_cols] = target_notionals
    target_sizes = target_notionals / df.loc[date, t_1_price_cols].values

    for i, ticker in enumerate(ticker_list):
        df.loc[date, f'{ticker}_target_size'] = target_sizes[i]

    total_target_notional = target_notionals.sum()
    df.loc[date, 'total_target_notional'] = total_target_notional

    return df


def get_desired_trades_from_target_notional(df, date, ticker_list, current_positions, transaction_cost_est,
                                            passive_trade_rate, notional_threshold_pct,
                                            total_portfolio_value, cash_buffer_percentage, min_trade_notional_abs,
                                            cooldown_counter_threshold):
    # --- 4) Build desired trades with STOP-GATE for new/added longs --------------
    ## Get Desired Trades based on Target Notionals and Current Notional Values by Ticker
    desired_positions = {}
    cash_debit = 0.0  # buys + fees
    cash_credit = 0.0  # sells - fees
    available_cash_for_trading = df.loc[date, 'available_cash'] * (1 - cash_buffer_percentage)

    ## Estimated Transaction Costs and Fees
    est_fees = (transaction_cost_est + perf.estimate_fee_per_trade(passive_trade_rate))

    for ticker in ticker_list:
        ## Calculate the cash need from all new target positions
        target_notional = df.loc[date, f'{ticker}_target_notional']
        current_notional = df.loc[date, f'{ticker}_open_position_notional']
        new_trade_notional = target_notional - current_notional
        trade_fees = abs(new_trade_notional) * est_fees
        mid_px = float(current_positions[ticker]['ticker_mid_price'])

        ## Calculate notional difference to determine if a trade is warranted
        portfolio_equity_trade_threshold = notional_threshold_pct * total_portfolio_value
        notional_threshold = notional_threshold_pct * abs(target_notional)
        notional_floors_list = [
            portfolio_equity_trade_threshold, notional_threshold, min_trade_notional_abs
        ]
        notional_floor = max(notional_floors_list)

        # --- STOP-GATE: block NEW/ADDED long exposure if stop is breached/invalid ---
        # We only gate when the delta adds long dollar risk (delta > 0).
        if new_trade_notional > 0:
            cooldown_active, days_left = state.is_in_cooldown(COOLDOWN_STATE_FILE, ticker, today=date)
            if cooldown_active:
                df.loc[date, f'{ticker}_stopout_flag'] = True
                df.loc[date, f'{ticker}_cooldown_counter'] = float(days_left)
                df.loc[date, f'{ticker}_event'] = f'cooldown_active({int(days_left)}d_left)'
                desired_positions[ticker] = {'new_trade_notional': 0.0,
                                             'trade_fees': 0.0,
                                             'reason': 'cooldown_active'}
                continue

            stop_today = df.loc[date, f'{ticker}_stop_loss']  # _long_stop_for_today(ticker)
            print(f'Ticker: {ticker}, Ticker Mid Price: {mid_px}, Stop Loss Price: {stop_today}')
            if not _entry_allowed_long(curr_price=mid_px, stop_today=stop_today, eps=0.0):
                # Block the buy; keep delta at zero but still record the stop for transparency.
                df.loc[date, f'{ticker}_stopout_flag'] = True
                df.loc[date, f'{ticker}_event'] = 'Stop Breached'
                # df.loc[date, f'{ticker}_stop_loss']    = float(stop_today)
                # Start cooldown (and log it); buys will be blocked from now on.
                state.start_cooldown(
                    state_path=COOLDOWN_STATE_FILE,
                    ticker=ticker,
                    breach_date=date,
                    cooldown_counter_threshold=cooldown_counter_threshold,
                    note=f"gate_block@{mid_px}",
                    log_path=COOLDOWN_LOG_FILE
                )
                _, days_left = state.is_in_cooldown(COOLDOWN_STATE_FILE, ticker, today=date)
                df.loc[date, f'{ticker}_cooldown_counter'] = float(days_left)
                desired_positions[ticker] = {'new_trade_notional': 0.0,
                                             'trade_fees': 0.0,
                                             'reason': 'stop_breached'}
                continue
            else:
                df.loc[date, f'{ticker}_stopout_flag'] = False
                df.loc[date, f'{ticker}_stop_loss'] = float(stop_today)
                df.loc[date, f'{ticker}_cooldown_counter'] = 0.0
        # For sells or flat, we don’t block: let risk come off if needed.

        if abs(new_trade_notional) > notional_floor:
            desired_positions[ticker] = {'new_trade_notional': new_trade_notional,
                                         'trade_fees': trade_fees,
                                         'reason': 'threshold_pass'}
        else:
            desired_positions[ticker] = {'new_trade_notional': 0,
                                         'trade_fees': 0,
                                         'reason': 'below_threshold'}

        if new_trade_notional >= 0:
            ## Buys
            cash_debit = cash_debit + new_trade_notional
        else:
            ## Sells
            net_trade_notional = new_trade_notional + trade_fees
            cash_credit = cash_credit + abs(net_trade_notional)

    ## Calculate Cash Shrink Factor for the portfolio for the day
    net_cash_need = cash_debit - cash_credit
    if net_cash_need > available_cash_for_trading + 1e-6:
        cash_shrink_factor = available_cash_for_trading / net_cash_need  # 0 < shrink < 1
    else:
        cash_shrink_factor = 1.0

    df.loc[date, f'cash_shrink_factor'] = cash_shrink_factor

    ## Apply Cash Shrink Factor to Desired Positions for Buys Only
    for ticker in ticker_list:
        if desired_positions[ticker]['new_trade_notional'] > 0:
            desired_positions[ticker]['new_trade_notional'] = desired_positions[ticker][
                                                                  'new_trade_notional'] * cash_shrink_factor
            desired_positions[ticker]['trade_fees'] = desired_positions[ticker]['trade_fees'] * cash_shrink_factor

        ## In the backtesting engine, new position notional is net of fees but this is not the case in
        ## the live strategy setup
        df.loc[date, f'{ticker}_new_position_notional'] = desired_positions[ticker]['new_trade_notional']
        df.loc[date, f'{ticker}_new_position_size'] = desired_positions[ticker]['new_trade_notional'] / \
                                                      current_positions[ticker]['ticker_mid_price']
        df.loc[date, f'{ticker}_actual_position_notional'] = df.loc[date, f'{ticker}_new_position_notional'] + \
                                                             df.loc[date, f'{ticker}_open_position_notional']
        df.loc[date, f'{ticker}_actual_position_size'] = df.loc[date, f'{ticker}_actual_position_notional'] / \
                                                         current_positions[ticker]['ticker_mid_price']

    return df, desired_positions


def ensure_cols(df: pd.DataFrame, col_defaults: dict) -> pd.DataFrame:
    missing = {c: v for c, v in col_defaults.items() if c not in df.columns}
    if not missing:
        return df
    # One concat = far fewer block insertions
    add = pd.DataFrame(missing, index=df.index)
    df = pd.concat([df, add], axis=1)
    return df


def get_desired_trades_by_ticker(client, cfg, date):

    ## Strategy Inputs
    ticker_list = cfg['universe']['tickers']
    initial_capital = cfg['run']['initial_capital']
    rolling_cov_window = cfg['risk_and_sizing']['rolling_cov_window']
    rolling_atr_window = cfg['risk_and_sizing']['rolling_atr_window']
    atr_multiplier = cfg['risk_and_sizing']['atr_multiplier']
    highest_high_window = cfg['risk_and_sizing']['highest_high_window']
    cash_buffer_percentage = cfg['risk_and_sizing']['cash_buffer_percentage']
    annualized_target_volatility = cfg['risk_and_sizing']['annualized_target_volatility']
    risk_min_signal = float(cfg['risk_and_sizing'].get('risk_min_signal', 1e-4))
    sleeve_risk_mode = str(cfg['risk_and_sizing'].get('sleeve_risk_mode', 'cap')).strip().lower()
    risk_sleeve_budget_tolerance = float(cfg['risk_and_sizing'].get('risk_sleeve_budget_tolerance', 1e-5))
    risk_optimizer_step = float(cfg['risk_and_sizing'].get('risk_optimizer_step', 0.5))
    risk_max_iterations = int(cfg['risk_and_sizing'].get('risk_max_iterations', 100))

    transaction_cost_est = cfg['execution_and_costs']['transaction_cost_est']
    passive_trade_rate = cfg['execution_and_costs']['passive_trade_rate']
    notional_threshold_pct = cfg['execution_and_costs']['notional_threshold_pct']
    min_trade_notional_abs = cfg['execution_and_costs']['min_trade_notional_abs']
    cooldown_counter_threshold = cfg['execution_and_costs']['cooldown_counter_threshold']
    annual_trading_days = cfg['run']['annual_trading_days']
    portfolio_name = cfg['portfolio']['name']

    ## Get Sleeve Budgets
    sleeve_budgets = cfg['universe']['sleeves']
    ticker_to_sleeve = {}
    for sleeve in sleeve_budgets.keys():
        print(sleeve)
        sleeve_tickers = sleeve_budgets[sleeve]['tickers']
        for ticker in sleeve_tickers:
            ticker_to_sleeve[ticker] = sleeve

    # --- 1) Build signal DF & covariance ----------------------------------------
    ## Generate Strategy Signal from T-1 Data
    df = get_strategy_trend_signal(cfg)

    # --- Ensure df has a normalized, tz-naive DatetimeIndex ---
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True).tz_localize(None)
    elif df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.sort_index(inplace=True)

    ## Get Target Notionals by Ticker in Universe
    print(f'Covariance Matrix Time: {datetime.now()}')
    ## Calculate the covariance matrix for tickers in the portfolio
    returns_cols = [f'{ticker}_t_1_close_pct_returns' for ticker in ticker_list]
    cov_matrix = df[returns_cols].rolling(rolling_cov_window).cov(pairwise=True).dropna()

    ## Delete rows prior to the first available date of the covariance matrix
    cov_matrix_start_date = cov_matrix.index.get_level_values(0).min()
    df = df[df.index >= cov_matrix_start_date]

    # --- 2) Live portfolio state -------------------------------------------------
    ## Get Portfolio Positions and Cash
    print(f'Start Time: {datetime.now()}')
    ## Create Coinbase Client & Portfolio UUID
    # client = cn.get_coinbase_rest_api_client(portfolio_name=portfolio_name)
    portfolio_uuid = cn.get_portfolio_uuid(client, portfolio_name=portfolio_name)

    print(f'Get Portfolio Equity and Cash Time: {datetime.now()}')
    ## Get Live Portfolio Equity
    portfolio_equity, available_cash = cn.get_live_portfolio_equity_and_cash(client=client,
                                                                             portfolio_name=portfolio_name)

    print(f'Get Current Positions Time: {datetime.now()}')
    ## Get Current Positions using Mid-Price
    ## TODO: CHECK TO SEE IF THE MID-PRICE BEING CAPTURED IS ACCURATE FROM COINBASE
    current_positions = cn.get_current_positions_from_portfolio(client, ticker_list=ticker_list,
                                                                portfolio_name=portfolio_name)

    # --- Refresh cooldowns from actual STOP fills (robust source of truth) ------
    refresh_cooldowns_from_stop_fills(
        client=client,
        tickers=ticker_list,
        today_date=date,
        state_file=COOLDOWN_STATE_FILE,
        log_file=COOLDOWN_LOG_FILE,
        get_stop_fills_fn=cn.get_stop_fills,  # your function
        cooldown_counter_threshold=cooldown_counter_threshold,
        lookback_days=10,
        effective_recent_days=2
    )

    ## Identify Daily Positions starting from day 2
    # previous_date = df.index[df.index.get_loc(date) - 1]
    # --- Resolve today's row and the previous trading day robustly ---
    date_ts = pd.Timestamp(date).normalize()
    idx = df.index

    # first index position >= date_ts
    cur_pos = idx.searchsorted(date_ts, side="left")
    if cur_pos >= len(idx):
        raise ValueError(f"{date_ts.date()} is after last available data ({idx[-1].date()})")
    # previous trading day MUST exist to seed T-1
    prev_pos = cur_pos - 1
    if prev_pos < 0:
        raise ValueError(f"Not enough history before {date_ts.date()} to seed previous day")

    date = idx[cur_pos]  # trading day used for 'today'
    previous_date = idx[prev_pos]  # trading day used for T-1

    col_defaults = {}
    for ticker in ticker_list:
        col_defaults.update({
            f'{ticker}_new_position_notional': 0.0,
            f'{ticker}_open_position_size': 0.0,
            f'{ticker}_open_position_notional': 0.0,
            f'{ticker}_actual_position_size': 0.0,
            f'{ticker}_actual_position_notional': 0.0,
            f'{ticker}_short_sale_proceeds': 0.0,
            f'{ticker}_new_position_entry_exit_price': 0.0,
            f'{ticker}_target_vol_normalized_weight': 0.0,
            f'{ticker}_target_notional': 0.0,
            f'{ticker}_target_size': 0.0,
            f'{ticker}_cash_shrink_factor': 0.0,
            f'{ticker}_stop_loss': 0.0,
            f'{ticker}_stopout_flag': False,
            f'{ticker}_cooldown_counter': 0.0,
            f'{ticker}_sleeve_risk_multiplier': 1.0,
            f'{ticker}_sleeve_risk_adj_weights': 0.0,
            f'{ticker}_event': pd.Series(pd.NA, index=df.index, dtype="string"),
        })

    df = ensure_cols(df, col_defaults)
    ord_cols = size_bin.reorder_columns_by_ticker(df.columns, ticker_list)
    df = df[ord_cols]

    ## Portfolio Level Cash and Positions are all set to 0
    df['daily_portfolio_volatility'] = 0.0
    df['available_cash'] = 0.0
    df['count_of_positions'] = 0.0
    df['total_actual_position_notional'] = 0.0
    df['total_target_notional'] = 0.0
    df['total_portfolio_value'] = 0.0
    df['total_portfolio_value_upper_limit'] = 0.0
    df['target_vol_scaling_factor'] = 1.0
    df['cash_scaling_factor'] = 1.0
    df['cash_shrink_factor'] = 1.0
    df['final_scaling_factor'] = 1.0

    # Seed state (+ carry over T-1 actuals, open notionals, and any existing stop levels)
    ## Assign Live Cash and Positions
    for ticker in ticker_list:
        # Actuals as of T-1
        df.loc[previous_date, f'{ticker}_actual_position_notional'] = current_positions[ticker][
            'ticker_current_notional']
        df.loc[previous_date, f'{ticker}_actual_position_size'] = current_positions[ticker]['ticker_qty']

        # Open Positions at T
        df.loc[date, f'{ticker}_open_position_notional'] = current_positions[ticker]['ticker_current_notional']
        df.loc[date, f'{ticker}_open_position_size'] = current_positions[ticker]['ticker_qty']

        # Carry Forward any Open Stop Loss Positions from T-1
        open_stop_loss = cn.get_open_stop_price(client, product_id=ticker, client_id_prefix='stop-')
        df.loc[previous_date, f'{ticker}_stop_loss'] = np.where(pd.isna(open_stop_loss), 0.0,
                                                                float(open_stop_loss)).item()

        # Pull in updated Stop Loss Values for Today
        df.loc[date, f'{ticker}_stop_loss'] = _long_stop_for_today(date, ticker, highest_high_window,
                                                                   rolling_atr_window, atr_multiplier)

    ## Portfolio Aggregates for today
    # Update Available Cash based on cash in Coinbase portfolio
    df.loc[date, 'available_cash'] = available_cash

    # Calculate Total Portfolio Value from Portfolio Positions
    short_sale_proceeds_cols = [f'{ticker}_short_sale_proceeds' for ticker in ticker_list]
    open_position_notional_cols = [f'{ticker}_open_position_notional' for ticker in ticker_list]
    df.loc[date, 'total_actual_position_notional'] = df[open_position_notional_cols].loc[date].sum()
    total_portfolio_value = (df.loc[date, 'available_cash'] +
                             df.loc[date, short_sale_proceeds_cols].sum() +
                             df.loc[date, 'total_actual_position_notional'])
    df.loc[date, 'total_portfolio_value'] = total_portfolio_value

    # Update Total Portfolio Value Upper Limit based on the Total Portfolio Value
    total_portfolio_value_upper_limit = (df.loc[date, 'total_portfolio_value'] *
                                         (1 - cash_buffer_percentage))
    df.loc[date, 'total_portfolio_value_upper_limit'] = total_portfolio_value_upper_limit

    # --- 3) Target notionals via target-vol sizing -------------------------------
    print(f'Target Volatility Position Sizing Time: {datetime.now()}')

    ## Derive the Daily Target Portfolio Volatility
    daily_target_volatility = annualized_target_volatility / np.sqrt(annual_trading_days)

    ## TODO: THIS NEEDS TO CHANGE TO ACCOUNT FOR THE RISK MULTIPLIER
    ## Calculate the target notional by ticker
    df = get_target_volatility_position_sizing_with_risk_multiplier_sleeve_weights_opt(
        df, cov_matrix, date, ticker_list, daily_target_volatility, total_portfolio_value_upper_limit,
        ticker_to_sleeve, sleeve_budgets, risk_max_iterations, risk_sleeve_budget_tolerance,
        risk_optimizer_step, risk_min_signal, sleeve_risk_mode)

    # --- 4) Build desired trades with STOP-GATE for new/added longs --------------
    ## Get Desired Trades based on Target Notionals and Current Notional Values by Ticker
    df, desired_positions = get_desired_trades_from_target_notional(df, date, ticker_list, current_positions,
                                                                    transaction_cost_est, passive_trade_rate,
                                                                    notional_threshold_pct,
                                                                    total_portfolio_value, cash_buffer_percentage,
                                                                    min_trade_notional_abs, cooldown_counter_threshold)

    return df, desired_positions, current_positions


def should_close_dust(df, date, actual_notional, min_notional_abs, pct_of_portfolio=0.002):
    """Close if notional is very small in absolute terms OR as a % of portfolio."""
    total_portfolio_value = float(df.loc[date, 'total_portfolio_value'])
    return (abs(actual_notional) < float(min_notional_abs)) or (abs(actual_notional) <
                                                                pct_of_portfolio * total_portfolio_value)


def estimated_close_cost_usd(notional_usd, transaction_cost_est, passive_trade_rate, extra_slippage_bp=0.0):
    """
    Estimated T-Cost in USD
    """
    est_fees = (transaction_cost_est + perf.estimate_fee_per_trade(passive_trade_rate))
    fee_cost = notional_usd * est_fees
    return fee_cost


def build_dust_close_orders(client, df, date, ticker_list, min_trade_notional_abs,
                            max_cost_usd=0.05,               # don't spend >$0.05 to clean dust
                            transaction_cost_est=0.001,
                            passive_trade_rate=0.05,
                            use_ioc=True):
    """
    Flatten tiny residual positions to zero, but only if expected cost is tiny.
    Sends marketable LIMIT IOC (safer) when use_ioc=True; else MARKET.
    """
    # 0) make sure index is a normalized DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True).tz_localize(None)
    elif df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.sort_index(inplace=True)

    # 1) resolve the trading row for the provided calendar date
    date_ts = pd.Timestamp(date).normalize()
    idx = df.index
    pos = idx.searchsorted(date_ts, side="left")  # first index >= date
    if pos >= len(idx):
        raise ValueError(f"{date_ts.date()} is after last available data ({idx[-1].date()})")
    date = idx[pos]

    orders = []
    price_map = cn.get_price_map(client, ticker_list)
    for ticker in ticker_list:
        actual_notional = float(df.loc[date, f'{ticker}_actual_position_notional'])
        if not should_close_dust(df, date, actual_notional, min_trade_notional_abs):
            continue

        size_now = float(df.loc[date, f'{ticker}_actual_position_size'])
        if abs(size_now) < 1e-12:
            continue

        # Check product constraints
        specs = cn.get_product_meta(client, ticker)  # should return price_increment, base_increment, quote_min_size, best_bid/ask if you include it
        quote_min = float(specs.get('quote_min_size') or 0.0)
        if abs(actual_notional) < quote_min:
            # below exchange min; skip (or accumulate until above)
            continue

        # Cost guardrail
        est_cost = estimated_close_cost_usd(abs(actual_notional), transaction_cost_est,
                                            passive_trade_rate, extra_slippage_bp=0.0)
        if est_cost > max_cost_usd:
            continue

        side = 'SELL' if size_now > 0 else 'BUY'
        base_sz = cn.round_to_increment(abs(size_now), float(specs['base_increment']))

        if use_ioc:
            # marketable limit: cross the spread slightly to fill now, avoid wild prints
            # You'll need a fresh price; if your specs() doesn’t return bid/ask, fetch your own last/quote.
            best_bid = float(price_map[ticker]['best_bid_price'] or 0.0)
            best_ask = float(price_map[ticker]['best_ask_price'] or 0.0)
            inc = float(specs['price_increment'])
            if side == 'SELL':
                limit_px = cn.round_down(best_bid if best_bid else 0, inc)
                tif = "IOC"
                order_cfg = {"limit_limit_ioc": {"base_size": f"{base_sz}",
                                                 "limit_price": f"{limit_px}",
                                                 "post_only": False}}
                otype = "limit"
            else:
                limit_px = cn.round_up(best_ask if best_ask else 0, inc)
                tif = "IOC"
                order_cfg = {"limit_limit_ioc": {"base_size": f"{base_sz}",
                                                 "limit_price": f"{limit_px}",
                                                 "post_only": False}}
                otype = "limit"
        else:
            # pure market
            order_cfg = {"market_market_ioc": {"base_size": f"{base_sz}"}}
            otype = "market"

        orders.append({
            "product_id": ticker,
            "side": side.upper(),
            "type": otype,
            "size": base_sz,
            "client_order_id": f"{ticker}-{pd.Timestamp(date).strftime('%Y%m%d')}-dustclose",
            # If you use your create_order helpers, you can pass the generic dict,
            # or directly call client.create_order with order_configuration:
            "order_configuration": order_cfg,
        })

        # Reflect intended flatten in df (optional; you might prefer to do this post-trade)
        df.loc[date, f'{ticker}_actual_position_notional'] = 0.0
        df.loc[date, f'{ticker}_actual_position_size'] = 0.0

    return orders


def submit_dust_close_orders(client, orders, preview=True):
    """
    Use your helpers directly if you want; here we call Coinbase endpoints
    with the already-built order_configuration.
    Returns a list of per-order results.
    """
    results = []
    for od in orders:
        try:
            if preview:
                r = client.preview_order(
                    product_id=od["product_id"],
                    side=od["side"],
                    order_configuration=od["order_configuration"],
                )
            else:
                r = client.create_order(
                    client_order_id=od["client_order_id"],
                    product_id=od["product_id"],
                    side=od["side"],
                    order_configuration=od["order_configuration"],
                )
            results.append({"ok": True, "request": od, "response": r})
        except Exception as e:
            results.append({"ok": False, "request": od, "error": f"{type(e).__name__}: {e}"})
    return results


def update_trailing_stop_chandelier(
        client, df, ticker, date,
        highest_high_window=56, rolling_atr_window=20, atr_multiplier=2.5,
        stop_loss_replace_threshold_ticks=1, client_id_prefix="stop-",
        limit_price_buffer=0.005
):
    # --- ensure normalized DatetimeIndex ---
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True).tz_localize(None)
    elif df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.sort_index(inplace=True)

    # --- resolve “today” and “yesterday” on the index ---
    target = pd.Timestamp(date).normalize()
    idx = df.index
    cur_pos = idx.searchsorted(target, side="left")  # first index >= calendar date
    if cur_pos >= len(idx):
        return {"ok": False, "action": "skip", "reason": f"date {target.date()} after last data {idx[-1].date()}"}
    date = idx[cur_pos]
    previous_date = idx[cur_pos - 1] if cur_pos > 0 else None

    # --- specs & compute desired stop ---
    specs = cn.get_product_meta(client, product_id=ticker)
    tick = float(specs['price_increment'])
    result_dict = {}

    stop_today = float(chandelier_stop_long(date, ticker, highest_high_window, rolling_atr_window, atr_multiplier))
    stop_prev = df.get(f'{ticker}_stop_loss', pd.Series(index=df.index, dtype=float)).shift(1).loc[date]

    # monotone ratchet
    candidates = [x for x in (stop_prev, stop_today) if np.isfinite(x)]
    desired_stop = max(candidates) if candidates else stop_today
    desired_stop = cn.round_to_increment(desired_stop, tick)

    # skip if not a meaningful ratchet
    threshold = (stop_prev if np.isfinite(stop_prev) else -np.inf) + (stop_loss_replace_threshold_ticks * tick)
    if np.isfinite(stop_prev) and desired_stop < threshold:
        df.loc[date, f'{ticker}_stop_loss'] = float(stop_prev)
        result_dict = {
            "ok": True,
            "action": "skip",
            "reason": "no_ratchet",
            "open_stop_price": float(stop_prev),
            "stop_today": float(stop_today),
            "desired_stop": float(desired_stop),
            "tick": float(tick)
        }
        return result_dict

    # position size
    try:
        pos_size = float(df.loc[date, f'{ticker}_actual_position_size'])
    except Exception:
        pos_map = (cn.get_current_positions_from_portfolio(client, [ticker]) or {}).get(ticker, {})
        pos_size = float(pos_map.get('ticker_qty', 0.0))
    if pos_size <= 0:
        df.loc[date, f'{ticker}_stop_loss'] = float(desired_stop)
        result_dict = {
            "ok": True,
            "action": "skip",
            "reason": "no_position",
            "open_stop_price": float(stop_prev) if np.isfinite(stop_prev) else None,
            "stop_today": float(stop_today),
            "desired_stop": float(desired_stop),
            "tick": float(tick)
        }
        return result_dict

    client_order_id = f"{client_id_prefix}{ticker}-{date:%Y%m%d}-{int(round(desired_stop / tick))}"

    # --- PREVIEW FIRST (do NOT cancel yet) ---
    pv = cn.place_stop_limit_order(
        client=client,
        product_id=ticker,
        side="SELL",
        stop_price=desired_stop,
        size=pos_size,
        client_order_id=client_order_id,
        buffer_bps=50,
        preview=True,
        price_increment=specs["price_increment"],
        base_increment=specs["base_increment"],
        quote_min_size=specs["quote_min_size"]
    )
    pv_d = cn._as_dict(pv)
    errs = pv_d.get("errs") or []

    if errs:
        # keep old stop; report clearly
        df.loc[date, f'{ticker}_stop_loss'] = float(stop_prev)  # persist your computed target for next run
        result_dict = {
            "ok": False,
            "action": "no_change",
            "reason": "preview_error",
            "preview_errors": errs,  # e.g., ["PREVIEW_STOP_PRICE_ABOVE_LAST_TRADE_PRICE"]
            "open_stop_price": float(stop_prev) if np.isfinite(stop_prev) else None,
            "stop_today": float(stop_today),
            "desired_stop": float(desired_stop),
            "tick": float(tick),
            "preview_id": pv_d.get("preview_id")
        }
        return result_dict

    # --- PREVIEW OK → now cancel existing stops, then place live ---
    try:
        open_orders = cn.list_open_stop_orders(client, product_id=ticker) or []
        for o in open_orders:
            otype = str(o.get('type') or '').lower()
            if ('stop' in otype) or (o.get('stop_price') is not None):
                cn.cancel_order_by_id(client, order_id=o['order_id'])
    except Exception as e:
        # still attempt to place the new one
        print(f"[warn] cancel stop failed for {ticker}: {e}")

    cr = cn.place_stop_limit_order(
        client=client,
        product_id=ticker,
        side="SELL",
        stop_price=desired_stop,
        size=pos_size,
        client_order_id=client_order_id,
        buffer_bps=50,
        preview=False,  # LIVE
        price_increment=specs["price_increment"],
        base_increment=specs["base_increment"],
        quote_min_size=specs["quote_min_size"]
    )
    cr_d = cn._as_dict(cr)

    df.loc[date, f'{ticker}_stop_loss'] = float(desired_stop)
    result_dict = {
        "ok": True,
        "action": "replaced",
        "client_order_id": client_order_id,
        "new_order_id": cr_d.get("response", {}).get("order_id") or cr_d.get("order_id"),
        "desired_stop": float(desired_stop),
        "pos_size": float(pos_size),
    }

    return result_dict


# ====== MAIN ORCHESTRATOR (uses your order functions) ======
def main():
    args = parse_args()
    now = utc_now()
    today = now.date()

    # Generate Run Id for the run and log start time
    run_id = new_run_id(now)
    set_run_id(run_id)
    started_at = utc_now_iso()
    run_errors = []
    stop_results = {}

    # UTC gate (run in first N minutes of the target UTC hour)
    if not args.force_run:
        if not (now.hour == args.run_at_utc_hour and 0 <= now.minute <= args.gate_minutes):
            msg = f"exit: gate skip at {now.isoformat()}Z"
            print(msg, flush=True)
            log_event("gate_skip", hour=now.hour, minute=now.minute)
            return

    DONE_FLAG_DIR.mkdir(parents=True, exist_ok=True)
    flag = DONE_FLAG_DIR / f"{today.isoformat()}.done"
    if flag.exists() and not args.force_run:
        msg = f"exit: already ran today {flag}"
        print(msg, flush=True)
        log_event("already_ran", flag=str(flag))
        return

    try:
        # 1) Load config (adjust import to your config file)
        cfg = load_prod_strategy_config(strategy_version='v0.2.0')
        portfolio_name = cfg['portfolio']['name']
        ticker_list = cfg['universe']['tickers']
        sleeve_budgets = cfg['universe']['sleeves']
        min_trade_notional_abs = cfg['execution_and_costs']['min_trade_notional_abs']
        transaction_cost_est = cfg['execution_and_costs']['transaction_cost_est']
        passive_trade_rate = cfg['execution_and_costs']['passive_trade_rate']
        highest_high_window = cfg['risk_and_sizing']['highest_high_window']
        rolling_atr_window = cfg['risk_and_sizing']['rolling_atr_window']
        atr_multiplier = cfg['risk_and_sizing']['atr_multiplier']

        ## Get Sleeve Budgets
        sleeve_budgets = cfg['universe']['sleeves']
        ticker_to_sleeve = {}
        for sleeve in sleeve_budgets.keys():
            sleeve_tickers = sleeve_budgets[sleeve]['tickers']
            for ticker in sleeve_tickers:
                ticker_to_sleeve[ticker] = sleeve

        log_event("config_loaded",
                  portfolio=portfolio_name,
                  tickers=ticker_list,
                  dry_run=bool(args.dry_run))

        # 2) Coinbase client
        client = cn.get_coinbase_rest_api_client(portfolio_name=portfolio_name)
        _ = cn.get_portfolio_uuid(client, portfolio_name=portfolio_name)  # validate
        log_event("client_ready", portfolio=portfolio_name)

        # 3) Build desired trades (your function already applies stop & cooldown gates)
        #    Returns df (with *_stop_loss, *_event, *_cooldown_counter) and desired_positions dict
        df, desired_positions, current_positions = get_desired_trades_by_ticker(client, cfg, date=today)

        # Log desired positions summary & any gate reasons
        try:
            total_buys = sum(1 for t, d in desired_positions.items() if float(d.get("new_trade_notional", 0.0)) > 0)
            total_sells = sum(1 for t, d in desired_positions.items() if float(d.get("new_trade_notional", 0.0)) < 0)
            total_zero = sum(
                1 for t, d in desired_positions.items() if abs(float(d.get("new_trade_notional", 0.0))) < 1e-9)
        except Exception:
            total_buys = total_sells = total_zero = None

        log_event("desired_trades_ready",
                  total_buys=total_buys, total_sells=total_sells, total_zero=total_zero)

        # Per-ticker log (compact)
        for t, d in desired_positions.items():
            write_jsonl(DESIRED_TRADES_LOG, {
                "ts": utc_now_iso(),
                "date": str(today),
                "ticker": t,
                "new_trade_notional": float(d.get("new_trade_notional", 0.0)),
                "reason": d.get("reason"),
            })

        # 4) Create **rebalance orders** using YOUR builder
        #    Most implementations expect either target sizes/weights or a per-ticker desired notional.
        #    If your builder needs a specific shape, adapt the dict below.
        #    Example: desired_positions[t] = {'new_trade_notional': float, 'reason': 'threshold_pass'|...}
        rebalance_orders = build_rebalance_orders(desired_positions=desired_positions,
                                                  date=today, current_positions=current_positions,
                                                  client=client, order_type='market', limit_price_buffer=0.0)

        # Log all built orders
        for o in rebalance_orders:
            write_jsonl(ORDER_BUILD_LOG, {"ts": utc_now_iso(), "stage": "rebalance_build", **o})
        log_event("rebalance_built", count=len(rebalance_orders))

        # 5) Submit daily rebalance orders
        if rebalance_orders:
            preview_flag = bool(args.dry_run)
            try:
                resp = submit_daily_rebalance_orders(client, rebalance_orders, preview=preview_flag)
                write_jsonl(ORDER_SUBMIT_LOG, {
                    "ts": utc_now_iso(),
                    "stage": "rebalance_submit",
                    "preview": preview_flag,
                    "orders_count": len(rebalance_orders),
                    "response": cn._as_dict(resp) if 'resp' in locals() else None
                })
            except Exception as e:
                err = {"ts": utc_now_iso(), "where": "submit_daily_rebalance_orders",
                       "preview": preview_flag, "error": str(e)}
                write_jsonl(LIVE_ERRORS_LOG, err)
                run_errors.append(err)
                print(f"[warn] rebalance submit failed: {e}", flush=True)

        # 6) Build and submit **dust close** orders (your helpers)
        dust_close_orders = build_dust_close_orders(
            client=client, df=df, date=today, ticker_list=ticker_list,
            min_trade_notional_abs=min_trade_notional_abs,
            max_cost_usd=0.05,               # don't spend >$0.05 to clean dust
            transaction_cost_est=transaction_cost_est,
            passive_trade_rate=passive_trade_rate,
            use_ioc=True)

        for o in (dust_close_orders or []):
            write_jsonl(DUST_BUILD_LOG, {"ts": utc_now_iso(), "stage": "dust_build", **o})
        log_event("dust_built", count=len(dust_close_orders or []))

        if dust_close_orders:
            preview_flag = bool(args.dry_run)
            try:
                resp = submit_dust_close_orders(client=client, orders=dust_close_orders, preview=preview_flag)
                write_jsonl(DUST_SUBMIT_LOG, {
                    "ts": utc_now_iso(),
                    "stage": "dust_submit",
                    "preview": preview_flag,
                    "orders_count": len(dust_close_orders),
                    "response": cn._as_dict(resp) if 'resp' in locals() else None
                })
            except Exception as e:
                err = {"ts": utc_now_iso(), "where": "submit_dust_close_orders",
                       "preview": preview_flag, "error": str(e)}
                write_jsonl(LIVE_ERRORS_LOG, err)
                run_errors.append(err)
                print(f"[warn] dust submit failed: {e}", flush=True)

        # 8) Update trailing stops (Chandelier)
        for ticker in ticker_list:
            try:
                stop_loss_dict = update_trailing_stop_chandelier(
                    client=client, df=df, ticker=ticker, date=today,
                    highest_high_window=highest_high_window,
                    rolling_atr_window=rolling_atr_window,
                    atr_multiplier=atr_multiplier,
                    stop_loss_replace_threshold_ticks=1,
                    client_id_prefix="stop-",
                    limit_price_buffer=0.005
                )
                stop_results[ticker] = stop_loss_dict or {"ok": True, "action": "none"}
                write_jsonl(STOP_UPDATE_LOG, {
                    "ts": utc_now_iso(),
                    "ticker": ticker,
                    **(stop_loss_dict or {})
                })
            except Exception as e:
                err = {"ts": utc_now_iso(), "where": "update_trailing_stop_chandelier",
                       "ticker": ticker, "error": str(e)}
                write_jsonl(LIVE_ERRORS_LOG, err)
                run_errors.append(err)
                stop_results[ticker] = {"ok": False, "error": str(e)}
                print(f"[warn] update_trailing_stop_chandelier({ticker}) failed: {e}", flush=True)

        # 9) Generate Email Summary, send email and log completed timestamp
        completed_at = utc_now_iso()

        # Write the authoritative daily snapshot
        summary_path = write_daily_summary(
            cfg=cfg,
            day=today,
            run_id=CURRENT_RUN_ID or "unknown",  # or your run_id variable if you have it
            dry_run=bool(args.dry_run),
            started_at=started_at,
            completed_at=completed_at,
            df=df,
            ticker_list=ticker_list,
            ticker_to_sleeve=ticker_to_sleeve,
            desired_positions=desired_positions,
            current_positions=current_positions,
            rebalance_orders=rebalance_orders,
            dust_orders=dust_close_orders,
            stop_results=stop_results,
            errors=run_errors,
        )
        log_event("daily_summary_written", path=str(summary_path))

        # Send summary email (ideally read daily_summary_YYYY-MM-DD.json)
        try:
            ok, msg = send_summary_email(STATE_DIR, today)  # see email tweak below
            log_event("email_sent", ok=bool(ok), msg=str(msg))
            print(f"[email] summary: {ok} ({msg})", flush=True)
        except Exception as e:
            # don’t let email failure crash the run; log it
            err = {"ts": utc_now_iso(), "where": "send_summary_email", "error": str(e)}
            write_jsonl(LIVE_ERRORS_LOG, err)
            run_errors.append(err)

        # 10) Done flag & heartbeat
        flag.write_text(now.isoformat())
        log_event("run_complete", date=str(today))

        print(f"success: {today} complete", flush=True)

    except Exception as e:
        # Fatal catch & log
        tb = traceback.format_exc()
        write_jsonl(LIVE_ERRORS_LOG, {"ts": utc_now_iso(), "error": str(e), "traceback": tb})
        print(f"[fatal] {e}", flush=True)
        raise

    return


if __name__ == "__main__":
    main()

