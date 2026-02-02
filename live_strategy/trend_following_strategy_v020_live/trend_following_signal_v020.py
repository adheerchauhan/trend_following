import time
import numpy as np
import pandas as pd
import requests
import datetime as dt
import scipy
from utils import coinbase_utils as cn


def get_coinbase_daily_historical_price_data_with_retry_logic(
    client,
    ticker,
    start_timestamp: int,
    end_timestamp: int,
    retries: int = 2,
    delay: float = 1.0,
    retry_on_empty: bool = True,
):
    """
    Fetch Coinbase daily candles and ALWAYS return a dataframe indexed by every day in the
    requested window [start_timestamp, end_timestamp] (inclusive), with NaNs where data is missing.

    Key points:
      - Preserves NaNs (does not drop them)
      - Ensures the index contains *all* days in the requested window (reindex)
      - Robustly converts SDK candle objects -> dicts so pandas columns are correct
      - Fast on parse/shape errors (no long sleep loops)
      - Only sleeps on ConnectionError
    """
    granularity = "ONE_DAY"
    cols = ["open", "high", "low", "close", "volume"]

    # Inclusive daily index for this chunk (tz-naive midnight)
    start_dt = pd.to_datetime(int(start_timestamp), unit="s", utc=True).tz_localize(None).normalize()
    end_dt   = pd.to_datetime(int(end_timestamp),   unit="s", utc=True).tz_localize(None).normalize()

    if end_dt < start_dt:
        return pd.DataFrame(columns=cols).rename_axis("date")

    expected_idx = pd.date_range(start=start_dt, end=end_dt, freq="D", name="date")

    def nan_frame():
        return pd.DataFrame(index=expected_idx, columns=cols, dtype=float).rename_axis("date")

    def candle_to_dict(c):
        """
        Coinbase SDKs can return dicts OR objects. Convert robustly.
        """
        if isinstance(c, dict):
            return c
        # Some SDK objects have model_dump()/dict()
        if hasattr(c, "model_dump"):
            return c.model_dump()
        if hasattr(c, "dict"):
            try:
                return c.dict()
            except Exception:
                pass
        # Fallback to __dict__ (works for many simple objects)
        if hasattr(c, "__dict__") and isinstance(c.__dict__, dict):
            return c.__dict__
        # Last resort: try casting
        try:
            return dict(c)
        except Exception:
            return None

    def build_output_from_candles(candle_list):
        # Convert all candles to dicts
        rows = []
        for c in candle_list:
            d = candle_to_dict(c)
            if d is not None:
                rows.append(d)

        if not rows:
            return None  # indicates parse failure

        df = pd.DataFrame(rows)

        # Normalize expected key names: some return 'start' instead of 'date'
        if "date" not in df.columns and "start" in df.columns:
            df = df.rename(columns={"start": "date"})

        if "date" not in df.columns:
            return None

        # Parse 'date' robustly: could be epoch seconds as string/num
        dt_series = pd.to_datetime(pd.to_numeric(df["date"], errors="coerce"), unit="s", utc=True, errors="coerce")
        dt_series = dt_series.dt.tz_localize(None).dt.normalize()

        df["date"] = dt_series
        df = df.set_index("date").sort_index()
        df.index.name = "date"

        # Build OHLCV numeric columns (preserve NaN)
        out = pd.DataFrame(index=df.index)
        for c in cols:
            out[c] = pd.to_numeric(df.get(c), errors="coerce")

        # De-dup days and reindex to expected
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out = out.reindex(expected_idx)

        return out

    # --- Main fetch with limited retries ---
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # IMPORTANT: many candle APIs treat end as exclusive; if you want inclusive,
            # you can optionally pass end_timestamp + 86400 in the CALLER.
            candle_list = client.get_candles(
                product_id=ticker,
                start=int(start_timestamp),
                end=int(end_timestamp),
                granularity=granularity
            ).candles

            if not candle_list:
                # Wait-and-retry on empty (important around daily close / 00:00 UTC)
                if retry_on_empty and attempt < retries:
                    time.sleep(delay)
                    continue
                return nan_frame()


            out = build_output_from_candles(candle_list)
            if out is None:
                # Parse/shape issue -> fast fail (do NOT sleep; return NaNs)
                return nan_frame()

            # If the last day is missing/NaN, treat as incomplete and retry
            last_day = expected_idx[-1]
            if retry_on_empty and attempt < retries:
                if (last_day not in out.index) or out.loc[last_day, "close"] != out.loc[last_day, "close"]:  # NaN check
                    time.sleep(delay)
                    continue

            return out

        except requests.exceptions.ConnectionError as e:
            last_err = e
            if attempt < retries:
                time.sleep(delay)
                continue
            return nan_frame()
        except Exception as e:
            # Unknown exception: do NOT loop/sleep in production; return NaN frame fast.
            last_err = e
            return nan_frame()

    return nan_frame()


def save_historical_crypto_prices_from_coinbase_with_delay(
    ticker,
    user_start_date=False,
    start_date=None,
    end_date=None,
    save_to_file=False,
    portfolio_name='Default',
    retries=5,
    delay=10,
    retry_on_empty=True
):
    client = cn.get_coinbase_rest_api_client(portfolio_name=portfolio_name)

    # --- start_date ---
    if user_start_date:
        start_date = pd.Timestamp(start_date)
    else:
        start_date = cn.coinbase_start_date_by_ticker_dict.get(ticker)
        if not start_date:
            print(f"Start date for {ticker} is not included in the dictionary!")
            return None
        start_date = pd.Timestamp(start_date)

    # --- end_date ---
    end_date = pd.Timestamp(end_date)

    # Make BOTH tz-aware UTC + normalized (so comparisons work and timestamps are correct)
    start_date_utc = start_date.tz_localize("UTC") if start_date.tzinfo is None else start_date.tz_convert("UTC")
    end_date_utc   = end_date.tz_localize("UTC")   if end_date.tzinfo is None   else end_date.tz_convert("UTC")

    temp_start_date = start_date_utc.normalize()
    end_date_utc = end_date_utc.normalize()
    current_end_date = temp_start_date

    crypto_price_list = []

    while current_end_date < end_date_utc:
        current_end_date = temp_start_date + dt.timedelta(weeks=6)
        if current_end_date > end_date_utc:
            current_end_date = end_date_utc

        start_timestamp = int(temp_start_date.timestamp())
        end_timestamp   = int(current_end_date.timestamp())

        crypto_price_list.append(
            get_coinbase_daily_historical_price_data_with_retry_logic(
                client, ticker, start_timestamp, end_timestamp,
                retries=retries, delay=delay, retry_on_empty=retry_on_empty
            )
        )

        temp_start_date = current_end_date + dt.timedelta(days=1)

    df = pd.concat(crypto_price_list, axis=0)

    if save_to_file:
        # use tz-naive dates just for filename readability
        filename = f"{ticker}-pickle-{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}"
        output_file = f'coinbase_historical_price_folder/{filename}'
        df.to_pickle(output_file)

    return df


def pct_rank(x, window=250):
    return x.rank(pct=True)


def calculate_portfolio_volatility(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))


def get_returns_volatility(df, vol_range_list=[10], close_px_col='BTC-USD_close'):
    df[f'{close_px_col}_pct_returns'] = df[close_px_col].pct_change()
    for vol_range in vol_range_list:
        df[f'{close_px_col}_volatility_{vol_range}'] = df[f'{close_px_col}_pct_returns'].rolling(vol_range).std()

    return df


def create_trend_strategy_log_space_prod(df, ticker, mavg_start, mavg_end, mavg_stepsize, mavg_z_score_window=252):
    df_working = df.copy()

    # ---- constants ----
    windows = np.geomspace(mavg_start, mavg_end, mavg_stepsize).round().astype(int)
    windows = np.unique(windows)
    x = np.log(windows[::-1])
    xm = x - x.mean()
    varx = (xm ** 2).sum()

    # ---- compute MAs (vectorised) ----
    df_working[f'{ticker}_t_1_close_log'] = np.log(df_working[f'{ticker}_t_1_close'])
    for w in windows:
        df_working[f'{ticker}_{w}_t_1_ema'] = df_working[f'{ticker}_t_1_close_log'].ewm(span=w, adjust=False).mean()

    mavg_mat = df_working[[f'{ticker}_{w}_t_1_ema' for w in windows]].to_numpy()

    # ---- slope (vectorised) ----
    slope = mavg_mat.dot(xm) / varx  # ndarray (T,)
    slope = pd.Series(slope, index=df_working.index)  # lag to avoid look-ahead

    # ---- z-score & rank ----
    z = ((slope - slope.rolling(mavg_z_score_window, min_periods=mavg_z_score_window).mean()) /
         slope.rolling(mavg_z_score_window, min_periods=mavg_z_score_window).std())

    # Optional Tail Cap
    z = z.clip(-4, 4)

    # Calculate the Percentile Rank based on CDF
    rank = scipy.stats.norm.cdf(z) - 0.5  # centered 0 ↔ ±0.5

    trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
    trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
    df_working[trend_continuous_signal_col] = slope
    df_working[trend_continuous_signal_rank_col] = rank

    return df_working


def calculate_donchian_channel_dual_window_prod(df, ticker, entry_rolling_donchian_window=20,
                                                exit_rolling_donchian_window=20):
    df_working = df.copy()

    ## Entry Channel
    # Rolling maximum of returns (upper channel)
    df_working[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band'] = (
        df_working[f'{ticker}_t_1_high'].rolling(window=entry_rolling_donchian_window).max())

    # Rolling minimum of returns (lower channel)
    df_working[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band'] = (
        df_working[f'{ticker}_t_1_low'].rolling(window=entry_rolling_donchian_window).min())

    ## Exit Channel
    # Rolling maximum of returns (upper channel)
    df_working[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band'] = (
        df_working[f'{ticker}_t_1_high'].rolling(window=exit_rolling_donchian_window).max())

    # Rolling minimum of returns (lower channel)
    df_working[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band'] = (
        df_working[f'{ticker}_t_1_low'].rolling(window=exit_rolling_donchian_window).min())

    # Middle of the channel (optional, could be just average of upper and lower)
    # Entry Middle Band
    df_working[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_middle_band'] = (
            (df_working[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band'] +
             df_working[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band']) / 2)

    # Exit Middle Band
    df_working[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_middle_band'] = (
            (df_working[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band'] +
             df_working[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band']) / 2)

    return df_working


def calculate_rolling_r2_prod(df, ticker, t_1_close_price_col, rolling_r2_window=30, lower_r_sqr_limit=0.2,
                              upper_r_sqr_limit=0.8, r2_smooth_window=3):

    df_working = df.copy()
    log_price_col = f'{ticker}_t_1_close_price_log'
    df_working[log_price_col] = np.log(df_working[t_1_close_price_col])

    ## Define the variables
    y = df_working[log_price_col]
    x = np.arange(len(y), dtype=float)  # Time

    ## Compute rolling sums for rolling R2 calculation
    x_sum = pd.Series(x, y.index).rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()
    y_sum = y.rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()
    x_sqr = pd.Series(x ** 2, y.index).rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()
    y_sqr = (y ** 2).rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()
    xy_sum = pd.Series(x, y.index).mul(y).rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()

    ## Calculate the R squared
    n = rolling_r2_window
    numerator = n * xy_sum - x_sum * y_sum
    denominator = np.sqrt((n * x_sqr) - (x_sum ** 2)) * np.sqrt((n * y_sqr) - (y_sum ** 2))
    df_working[f'{ticker}_rolling_r_sqr'] = (numerator / denominator) ** 2

    ## Normalize the R Squared centered around 0.5 where values below the lower limit are
    ## clipped to 0 and values above the upper limit are clipped to 1
    df_working[f'{ticker}_rolling_r_sqr'] = np.clip(
        (df_working[f'{ticker}_rolling_r_sqr'] - lower_r_sqr_limit) / (upper_r_sqr_limit - lower_r_sqr_limit),
        0, 1)

    ## Smoothing the Rolling R Squared Signal
    if r2_smooth_window >= 1:
        df_working[f'{ticker}_rolling_r_sqr'] = df_working[f'{ticker}_rolling_r_sqr'].ewm(span=r2_smooth_window, adjust=False).mean()

    return df_working


def generate_vol_of_vol_signal_log_space_prod(df, ticker, t_1_close_price_col, log_std_window=14,
                                              coef_of_variation_window=30, vol_of_vol_z_score_window=252,
                                              vol_of_vol_p_min=0.6):

    df_working = df.copy()
    log_returns_col = f'{ticker}_t_1_log_returns'
    realized_log_returns_vol = f'{ticker}_ann_log_volatility'
    df_working[log_returns_col] = np.log(df_working[t_1_close_price_col] / df_working[t_1_close_price_col].shift(1))
    eps = 1e-12

    ## Realized Volatility of Log Returns
    df_working[realized_log_returns_vol] = (
        df_working[log_returns_col].ewm(span=log_std_window, adjust=False,
                                        min_periods=log_std_window).std() * np.sqrt(365)
    )

    ## Coefficient of Variation in Volatility
    df_working[f'{ticker}_coef_variation_vol'] = (
            df_working[realized_log_returns_vol].rolling(coef_of_variation_window,
                                                         min_periods=coef_of_variation_window).std() /
            df_working[realized_log_returns_vol].rolling(coef_of_variation_window,
                                                         min_periods=coef_of_variation_window).mean().clip(lower=eps)
    )

    ## Calculate Robust Z-Score of the Coefficient of Variation
    cov_rolling_median = (
        df_working[f'{ticker}_coef_variation_vol'].rolling(vol_of_vol_z_score_window,
                                                   min_periods=vol_of_vol_z_score_window).median()
    )
    df_working[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median'] = cov_rolling_median
    cov_rolling_mad = (
        (df_working[f'{ticker}_coef_variation_vol'] - df_working[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median']).abs()
        .rolling(vol_of_vol_z_score_window, min_periods=vol_of_vol_z_score_window).median()
    )
    df_working[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median_abs_dev'] = cov_rolling_mad
    df_working[f'{ticker}_vol_of_vol_robust_z_score'] = (
        (df_working[f'{ticker}_coef_variation_vol'] - df_working[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median']) /
        (1.4826 * df_working[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median_abs_dev']).clip(lower=eps)
    )
    df_working[f'{ticker}_vol_of_vol_robust_z_score'] = (
        df_working[f'{ticker}_vol_of_vol_robust_z_score'].replace([np.inf, -np.inf], 0.0)
        .fillna(0.0).clip(lower=-3, upper=3)
    )

    ## Create Vol of Vol Thresholds
    ## z0 represents low volatility and z1 represents high volatility
    ## The vol of vol penalty will go from 1 to p_min where 1 represents no penalty
    z0, z1 = 0.5, 1.5                # z_vov below 0.5 → no penalty; above 1.5 → max raw penalty
    p_min = vol_of_vol_p_min         # even at max raw penalty, keep at least 60% exposure

    ## Compute a 0..1 raw penalty that rises from 0→1 as z_vov goes z0→z1
    df_working[f'{ticker}_vol_of_vol_signal_raw'] = (df_working[f'{ticker}_vol_of_vol_robust_z_score'] - z0) / max((z1 - z0), eps)

    ## Clip the signal to [0, 1]
    df_working[f'{ticker}_vol_of_vol_signal_raw'] = df_working[f'{ticker}_vol_of_vol_signal_raw'].clip(0, 1)

    ## Invert so that the raw penalty goes from 1 to 0 instead of 0 to 1
    df_working[f'{ticker}_vol_of_vol_penalty'] = 1 - df_working[f'{ticker}_vol_of_vol_signal_raw']

    ## Floor the penalty at p_min
    df_working[f'{ticker}_vol_of_vol_penalty'] = df_working[f'{ticker}_vol_of_vol_penalty'].clip(lower=p_min, upper=1)

    return df_working


def generate_trend_signal_with_donchian_channel_continuous_with_rolling_r_sqr_vol_of_vol_prod(
        start_date, end_date, ticker, fast_mavg, slow_mavg, mavg_stepsize, mavg_z_score_window,
        entry_rolling_donchian_window, exit_rolling_donchian_window, use_donchian_exit_gate, donchian_shift,
        ma_crossover_signal_weight, donchian_signal_weight, weighted_signal_ewm_window,
        rolling_r2_window=30, lower_r_sqr_limit=0.2, upper_r_sqr_limit=0.8, r2_smooth_window=3, r2_confirm_days=0,
        log_std_window=14, coef_of_variation_window=30, vol_of_vol_z_score_window=252, vol_of_vol_p_min=0.6,
        r2_strong_threshold=0.8, long_only=False):
    ## Pull prices from Coinbase only through yesterday (last fully-formed daily bar)
    run_end_date = pd.Timestamp(end_date) - pd.Timedelta(days=1)
    df = save_historical_crypto_prices_from_coinbase_with_delay(
        ticker=ticker,
        user_start_date=True,
        start_date=start_date,
        end_date=run_end_date,
        save_to_file=False,
        portfolio_name='Default',
        retries=5,
        delay=10,
        retry_on_empty=True,
    )

    # Keep needed cols + rename
    rename_map = {
        "close": f"{ticker}_t_1_close",
        "open": f"{ticker}_t_1_open",
        "high": f"{ticker}_t_1_high",
        "low": f"{ticker}_t_1_low",
        "volume": f"{ticker}_t_1_volume",
    }
    df = df[["close", "open", "high", "low", "volume"]].rename(columns=rename_map)

    # Ensure datetime index (tz-naive) and slice cleanly
    df.index = pd.to_datetime(df.index, errors="coerce").normalize()
    df = df.sort_index().loc[pd.Timestamp(start_date):run_end_date]

    # Add today's placeholder row (so shift produces "t-1" values aligned to today)
    df.loc[pd.Timestamp(end_date).normalize()] = np.nan

    # Lag everything by 1 day (today row becomes yesterday's bar)
    df = df.shift(1)

    # Create Column Names
    t_1_close_col = f'{ticker}_t_1_close'
    donchian_binary_signal_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_binary_signal'
    donchian_continuous_signal_col = f'{ticker}_donchian_continuous_signal'
    donchian_continuous_signal_rank_col = f'{ticker}_donchian_continuous_signal_rank'
    trend_binary_signal_col = f'{ticker}_trend_signal'
    trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
    trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
    final_binary_signal_col = f'{ticker}_final_binary_signal'
    final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'
    final_signal_col = f'{ticker}_final_signal'

    ## Generate Trend Signal in Log Space
    df_trend = create_trend_strategy_log_space_prod(df, ticker, mavg_start=fast_mavg, mavg_end=slow_mavg,
                                                    mavg_stepsize=mavg_stepsize,
                                                    mavg_z_score_window=mavg_z_score_window)

    ## Generate Donchian Channels
    # Donchian Buy signal: Price crosses above upper band
    # Donchian Sell signal: Price crosses below lower band
    df_donchian = calculate_donchian_channel_dual_window_prod(df, ticker=ticker,
                                                              entry_rolling_donchian_window=entry_rolling_donchian_window,
                                                              exit_rolling_donchian_window=exit_rolling_donchian_window)

    donchian_entry_upper_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band'
    donchian_entry_lower_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band'
    donchian_entry_middle_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_middle_band'
    donchian_exit_upper_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band'
    donchian_exit_lower_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band'
    donchian_exit_middle_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_middle_band'
    shift_cols = [donchian_entry_upper_band_col, donchian_entry_lower_band_col, donchian_entry_middle_band_col,
                  donchian_exit_upper_band_col, donchian_exit_lower_band_col, donchian_exit_middle_band_col]
    for col in shift_cols:
        df_donchian[f'{col}_t_1'] = df_donchian[col].shift(1)

    # Donchian Continuous Signal
    if donchian_shift:
        df_donchian[donchian_continuous_signal_col] = (
                (df_donchian[t_1_close_col] - df_donchian[f'{donchian_entry_middle_band_col}_t_1']) /
                (df_donchian[f'{donchian_entry_upper_band_col}_t_1'] - df_donchian[
                    f'{donchian_entry_lower_band_col}_t_1'])
        )
    else:
        df_donchian[donchian_continuous_signal_col] = (
                (df_donchian[t_1_close_col] - df_donchian[f'{donchian_entry_middle_band_col}']) /
                (df_donchian[f'{donchian_entry_upper_band_col}'] - df_donchian[f'{donchian_entry_lower_band_col}'])
        )

    ## Calculate Donchian Channel Rank
    ## Adjust the percentage ranks by 0.5 as without, the ranks go from 0 to 1. Recentering the function by giving it a steeper
    ## slope near the origin takes into account even little information
    df_donchian[donchian_continuous_signal_rank_col] = pct_rank(df_donchian[donchian_continuous_signal_col]) - 0.5

    # Donchian Binary Signal
    if donchian_shift:
        gate_long_condition = df_donchian[t_1_close_col] >= df_donchian[f'{donchian_exit_lower_band_col}_t_1']
        gate_short_condition = df_donchian[t_1_close_col] <= df_donchian[f'{donchian_exit_upper_band_col}_t_1']
    else:
        gate_long_condition = df_donchian[t_1_close_col] >= df_donchian[f'{donchian_exit_lower_band_col}']
        gate_short_condition = df_donchian[t_1_close_col] <= df_donchian[f'{donchian_exit_upper_band_col}']
    # sign of *entry* score decides direction
    entry_sign = np.sign(df_donchian[donchian_continuous_signal_col])
    # treat exact zero as "flat but allowed" (gate=1) so ranking not wiped out
    entry_sign = np.where(entry_sign == 0, 1, entry_sign)  # default to long-side keep
    df_donchian[donchian_binary_signal_col] = np.where(
        entry_sign > 0, gate_long_condition, gate_short_condition).astype(float)

    # Merging the Trend and Donchian Dataframes
    if donchian_shift:
        donchian_cols = [f'{donchian_entry_upper_band_col}_t_1', f'{donchian_entry_lower_band_col}_t_1',
                         f'{donchian_entry_middle_band_col}_t_1', f'{donchian_exit_upper_band_col}_t_1',
                         f'{donchian_exit_lower_band_col}_t_1', f'{donchian_exit_middle_band_col}_t_1',
                         donchian_binary_signal_col, donchian_continuous_signal_col,
                         donchian_continuous_signal_rank_col]
    else:
        donchian_cols = [f'{donchian_entry_upper_band_col}', f'{donchian_entry_lower_band_col}',
                         f'{donchian_entry_middle_band_col}', f'{donchian_exit_upper_band_col}',
                         f'{donchian_exit_lower_band_col}', f'{donchian_exit_middle_band_col}',
                         donchian_binary_signal_col, donchian_continuous_signal_col,
                         donchian_continuous_signal_rank_col]
    df_trend = pd.merge(df_trend, df_donchian[donchian_cols], left_index=True, right_index=True, how='left')

    ## Trend and Donchian Channel Signal
    # Calculate the exponential weighted average of the ranked signals to remove short-term flip-flops (whiplash)
    df_trend[[trend_continuous_signal_rank_col, donchian_continuous_signal_rank_col]] = (
        df_trend[[trend_continuous_signal_rank_col, donchian_continuous_signal_rank_col]].ewm(
            span=weighted_signal_ewm_window, adjust=False).mean())

    # Weighted Sum of Rank Columns
    df_trend[final_weighted_additive_signal_col] = (
            ma_crossover_signal_weight * df_trend[trend_continuous_signal_rank_col] +
            donchian_signal_weight * df_trend[donchian_continuous_signal_rank_col])

    # Apply Binary Gate
    if use_donchian_exit_gate:
        df_trend[final_weighted_additive_signal_col] = df_trend[final_weighted_additive_signal_col] * df_trend[
            donchian_binary_signal_col]

    ## Calculate Rolling R Squared Signal
    df_trend = calculate_rolling_r2_prod(df_trend, ticker=ticker, t_1_close_price_col=t_1_close_col,
                                         rolling_r2_window=rolling_r2_window, lower_r_sqr_limit=lower_r_sqr_limit,
                                         upper_r_sqr_limit=upper_r_sqr_limit, r2_smooth_window=r2_smooth_window)

    ## Calculate Vol of Vol Signal
    df_trend = generate_vol_of_vol_signal_log_space_prod(df_trend, ticker=ticker, t_1_close_price_col=t_1_close_col,
                                                         log_std_window=log_std_window,
                                                         coef_of_variation_window=coef_of_variation_window,
                                                         vol_of_vol_z_score_window=vol_of_vol_z_score_window,
                                                         vol_of_vol_p_min=vol_of_vol_p_min)

    ## Apply Regime Filters
    strong_rolling_r_sqr_cond = (df_trend[f'{ticker}_rolling_r_sqr'] >= r2_strong_threshold)
    df_trend[f'{ticker}_regime_filter'] = (
        np.where(strong_rolling_r_sqr_cond, df_trend[f'{ticker}_rolling_r_sqr'],
                 df_trend[f'{ticker}_rolling_r_sqr'] * df_trend[f'{ticker}_vol_of_vol_penalty']).astype(float)
    )
    df_trend[final_signal_col] = df_trend[final_weighted_additive_signal_col] * df_trend[f'{ticker}_regime_filter']

    # Introduce a Confirmation period for Rolling R Squared Signal
    if r2_confirm_days >= 1:
        df_trend[f'{ticker}_r2_enable'] = (
            (df_trend[f'{ticker}_rolling_r_sqr'] > 0.5).rolling(r2_confirm_days, min_periods=r2_confirm_days).min()
            .fillna(0.0).astype(float)
        )
        df_trend[final_signal_col] = df_trend[final_signal_col] * df_trend[f'{ticker}_r2_enable']
    else:
        df_trend[final_signal_col] = df_trend[final_signal_col]

    ## Long-Only Filter
    df_trend[final_signal_col] = np.where(long_only, np.maximum(0, df_trend[final_signal_col]),
                                          df_trend[final_signal_col])

    return df_trend


def get_trend_donchian_signal_for_portfolio_with_rolling_r_sqr_vol_of_vol_prod(
    start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, mavg_z_score_window,
    entry_rolling_donchian_window, exit_rolling_donchian_window, use_donchian_exit_gate, donchian_shift,
    ma_crossover_signal_weight, donchian_signal_weight, weighted_signal_ewm_window,
    rolling_r2_window=30, lower_r_sqr_limit=0.2, upper_r_sqr_limit=0.8, r2_smooth_window=3, r2_confirm_days=0,
    log_std_window=14, coef_of_variation_window=30, vol_of_vol_z_score_window=252, vol_of_vol_p_min=0.6,
    r2_strong_threshold=0.8, long_only=False):

    ## Generate trend signal for all tickers
    trend_list = []
    date_list = cn.coinbase_start_date_by_ticker_dict

    for ticker in ticker_list:
        # Create Column Names
        close_price_col = f'{ticker}_t_1_close'
        open_price_col = f'{ticker}_t_1_open'
        trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
        trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
        donchian_continuous_signal_col = f'{ticker}_donchian_continuous_signal'
        donchian_continuous_signal_rank_col = f'{ticker}_donchian_continuous_signal_rank'
        final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'
        rolling_r2_col = f'{ticker}_rolling_r_sqr'
        vol_of_vol_penalty_col = f'{ticker}_vol_of_vol_penalty'
        regime_filter_col = f'{ticker}_regime_filter'
        # rolling_r2_enable_col = f'{ticker}_r2_enable'
        final_signal_col = f'{ticker}_final_signal'

        if pd.to_datetime(date_list[ticker]).date() > start_date:
            run_date = pd.to_datetime(date_list[ticker]).date()
        else:
            run_date = start_date

        df_trend = generate_trend_signal_with_donchian_channel_continuous_with_rolling_r_sqr_vol_of_vol_prod(
            start_date=run_date, end_date=end_date, ticker=ticker, fast_mavg=fast_mavg, slow_mavg=slow_mavg,
            mavg_stepsize=mavg_stepsize, mavg_z_score_window=mavg_z_score_window,
            entry_rolling_donchian_window=entry_rolling_donchian_window, exit_rolling_donchian_window=exit_rolling_donchian_window,
            use_donchian_exit_gate=use_donchian_exit_gate, donchian_shift=donchian_shift,
            ma_crossover_signal_weight=ma_crossover_signal_weight, donchian_signal_weight=donchian_signal_weight,
            weighted_signal_ewm_window=weighted_signal_ewm_window,
            rolling_r2_window=rolling_r2_window, lower_r_sqr_limit=lower_r_sqr_limit,
            upper_r_sqr_limit=upper_r_sqr_limit, r2_smooth_window=r2_smooth_window, r2_confirm_days=r2_confirm_days,
            log_std_window=log_std_window, coef_of_variation_window=coef_of_variation_window,
            vol_of_vol_z_score_window=vol_of_vol_z_score_window, vol_of_vol_p_min=vol_of_vol_p_min,
            r2_strong_threshold=r2_strong_threshold, long_only=long_only)

        trend_cols = [close_price_col, open_price_col, trend_continuous_signal_col, trend_continuous_signal_rank_col,
                      donchian_continuous_signal_col, donchian_continuous_signal_rank_col,
                      final_weighted_additive_signal_col, rolling_r2_col, vol_of_vol_penalty_col,
                      regime_filter_col, final_signal_col]
        df_trend = df_trend[trend_cols]
        trend_list.append(df_trend)

    df_trend = pd.concat(trend_list, axis=1)

    return df_trend


# Below we calculate the number of risk units deployed per ticker per day
# Get Volatility Adjusted Trend Signal for Target Volatility Strategy
def get_volatility_adjusted_trend_signal_continuous_prod(df, ticker_list, volatility_window, annual_trading_days=365):

    ticker_signal_dict = {}
    final_cols = []
    for ticker in ticker_list:
        close_price_col = f'{ticker}_t_1_close'
        open_price_col = f'{ticker}_t_1_open'
        trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
        trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
        donchian_continuous_signal_col = f'{ticker}_donchian_continuous_signal'
        donchian_continuous_signal_rank_col = f'{ticker}_donchian_continuous_signal_rank'
        final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'
        rolling_r2_col = f'{ticker}_rolling_r_sqr'
        vol_of_vol_penalty_col = f'{ticker}_vol_of_vol_penalty'
        regime_filter_col = f'{ticker}_regime_filter'
        # rolling_r2_enable_col = f'{ticker}_r2_enable'
        final_signal_col = f'{ticker}_final_signal'
        annualized_volatility_col = f'{ticker}_annualized_volatility_{volatility_window}'
        vol_adj_trend_signal_col = f'{ticker}_vol_adjusted_trend_signal'

        ## Calculate Position Volatility Adjusted Trend Signal
        # df[f'{ticker}_t_1_close'] = df[f'{ticker}_close'].shift(1)
        df = get_returns_volatility(df, vol_range_list=[volatility_window], close_px_col=f'{ticker}_t_1_close')
        df[annualized_volatility_col] = (df[f'{ticker}_t_1_close_volatility_{volatility_window}'] *
                                         np.sqrt(annual_trading_days))
        df[vol_adj_trend_signal_col] = (df[final_signal_col] / df[annualized_volatility_col])
        df[vol_adj_trend_signal_col] = df[vol_adj_trend_signal_col].fillna(0)
        trend_cols = [close_price_col, open_price_col, f'{ticker}_t_1_close_pct_returns',
                      trend_continuous_signal_col, trend_continuous_signal_rank_col,
                      donchian_continuous_signal_col, donchian_continuous_signal_rank_col,
                      final_weighted_additive_signal_col, rolling_r2_col, vol_of_vol_penalty_col,
                      regime_filter_col, final_signal_col, annualized_volatility_col,
                      vol_adj_trend_signal_col]
        final_cols.append(trend_cols)
        ticker_signal_dict[ticker] = df[trend_cols].copy()
    df_signal = pd.concat(ticker_signal_dict, axis=1)

    ## Assign new column names to the dataframe
    df_signal.columns = df_signal.columns.to_flat_index()
    final_cols = [item for sublist in final_cols for item in sublist]
    df_signal.columns = final_cols

    return df_signal


