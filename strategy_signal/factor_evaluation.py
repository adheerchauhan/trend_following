import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0) Data adapter (use your native column names)
# ============================================================
def prep_factor_df(
        df: pd.DataFrame,
        time_col: str = "time",
        asset_col: str = "altname",
        factor_col: str = "mom_42",
        ret_col: str = "fwd_6",
        eligible_col: str | None = None,
) -> pd.DataFrame:
    """
    Returns a standardized long DF with columns:
      time, asset, factor, fwd_ret, (optional) eligible
    """
    keep = [time_col, asset_col, factor_col, ret_col]
    if eligible_col is not None and eligible_col in df.columns:
        keep.append(eligible_col)

    out = df[keep].copy()
    rename_map = {time_col: "time", asset_col: "asset", factor_col: "factor", ret_col: "fwd_ret"}
    if eligible_col is not None and eligible_col in out.columns:
        rename_map[eligible_col] = "eligible"
    out = out.rename(columns=rename_map)

    # enforce dtypes
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["asset"] = out["asset"].astype(str)
    out["factor"] = pd.to_numeric(out["factor"], errors="coerce")
    out["fwd_ret"] = pd.to_numeric(out["fwd_ret"], errors="coerce")

    if "eligible" in out.columns:
        out["eligible"] = out["eligible"].astype(bool)

    return out


# ============================================================
# 1) Robust cross-sectional standardization
# ============================================================
def cs_robust_zscore(x: pd.Series, clip: float = 6.0) -> pd.Series:
    """
    Cross-sectional robust z-score using median/MAD.
    """
    x = x.astype(float)
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.nan, index=x.index)
    z = (x - med) / (1.4826 * mad)  # MAD->std scaling under normality
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


def add_cs_zscore(df: pd.DataFrame, factor_col: str = "factor", out_col: str = "z") -> pd.DataFrame:
    """
    Adds cross-sectional robust z-score per time.
    """
    df = df.copy()
    df[out_col] = df.groupby("time")[factor_col].transform(cs_robust_zscore)
    return df


# ============================================================
# 2) Quintile assignment
# ============================================================
def assign_quantiles(
        df: pd.DataFrame,
        score_col: str = "z",
        n_q: int = 5,
        min_names: int = 15,
        out_col: str = "q",
) -> pd.DataFrame:
    """
    Assign cross-sectional quantiles per time. Uses rank->qcut to be stable under ties.
    q in {1..n_q}.
    """

    def _qcut_ranked(x: pd.Series) -> pd.Series:
        ok = x.notna()
        if ok.sum() < min_names:
            return pd.Series(np.nan, index=x.index)
        r = x[ok].rank(method="first")
        try:
            q = pd.qcut(r, q=n_q, labels=False) + 1
            out = pd.Series(np.nan, index=x.index)
            out.loc[ok] = q.astype(float)
            return out
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    df = df.copy()
    df[out_col] = df.groupby("time")[score_col].transform(_qcut_ranked)
    return df


# ============================================================
# 3) Portfolio returns by quintile (EW and "alpha-weighted")
# ============================================================
def _normalize_positive_weights(w: pd.Series) -> pd.Series:
    w = w.where(w > 0, 0.0)
    s = float(w.sum())
    if s <= 0 or np.isnan(s):
        return pd.Series(0.0, index=w.index)
    return w / s


def quintile_returns_equal_weight(
        df: pd.DataFrame,
        q_col: str = "q",
        ret_col: str = "fwd_ret",
        n_q: int = 5,
) -> pd.DataFrame:
    """
    DataFrame indexed by time with columns Q1..Qn of equal-weight mean forward return.
    """
    out = (
        df.dropna(subset=[q_col, ret_col])
        .groupby(["time", q_col])[ret_col]
        .mean()
        .unstack(q_col)
        .reindex(columns=range(1, n_q + 1))
    )
    out.columns = [f"Q{int(c)}" for c in out.columns]
    out = out.sort_index()
    return out


def quintile_returns_alpha_weighted(
        df: pd.DataFrame,
        q_col: str = "q",
        score_col: str = "z",
        ret_col: str = "fwd_ret",
        n_q: int = 5,
) -> pd.DataFrame:
    """
    Within each quintile, weights proportional to |z| (positive), normalized to sum to 1.
    """
    rows = []
    gdf = df.dropna(subset=[q_col, score_col, ret_col]).copy()

    for (t, q), g in gdf.groupby(["time", q_col]):
        z = g[score_col].abs()
        w = _normalize_positive_weights(z)
        r = float((w * g[ret_col]).sum())
        rows.append((t, int(q), r))

    out = pd.DataFrame(rows, columns=["time", "q", "ret"]).pivot(index="time", columns="q", values="ret")
    out = out.reindex(columns=range(1, n_q + 1)).sort_index()
    out.columns = [f"Q{int(c)}" for c in out.columns]
    return out


def top_minus_bottom(qret: pd.DataFrame, top: str = "Q5", bottom: str = "Q1") -> pd.Series:
    return (qret[top] - qret[bottom]).rename(f"{top}-{bottom}")


# ============================================================
# 4) Rolling IR / Sharpe for a return series
# ============================================================
def rolling_sharpe(ret: pd.Series, window: int, ann_factor: float, min_periods: int | None = None) -> pd.Series:
    """
    Rolling annualized Sharpe/IR: mean/std * sqrt(ann_factor)
    Uses min_periods to avoid all-NaN for long windows.
    """
    if min_periods is None:
        min_periods = max(10, int(0.6 * window))  # show earlier but still stable
        min_periods = min(min_periods, window)

    mu = ret.rolling(window, min_periods=min_periods).mean()
    sd = ret.rolling(window, min_periods=min_periods).std(ddof=1)
    return (mu / sd) * np.sqrt(ann_factor)


def choose_rolling_window(n_points: int, preferred: int, min_w: int = 30, max_frac: float = 0.6) -> int:
    """
    Pick a rolling window that is:
      - at least min_w (when possible)
      - at most max_frac * n_points (so you get rolling values)
      - close to preferred when feasible
    """
    if n_points <= 1:
        return 1

    # If the sample is small, shrink min_w automatically
    min_w_eff = min(min_w, max(5, n_points // 3))
    max_w = max(min_w_eff, int(max_frac * n_points))

    w = int(preferred)
    w = max(min_w_eff, min(w, max_w))
    return w


# ============================================================
# 5) Cross-sectional "alpha" diagnostics
# ============================================================
def cs_quantile_lines(
        df: pd.DataFrame,
        score_col: str = "z",
        probs=(0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99),
) -> pd.DataFrame:
    """
    For each time, compute cross-sectional quantiles of score.
    """

    def _q(x: pd.Series) -> pd.Series:
        x = x.dropna()
        if len(x) == 0:
            return pd.Series({p: np.nan for p in probs})
        return x.quantile(probs)

    qdf = df.groupby("time")[score_col].apply(_q).unstack()
    qdf.columns = [f"q{p:g}" for p in qdf.columns]
    qdf = qdf.sort_index()
    return qdf


def per_asset_mean_score(df: pd.DataFrame, score_col: str = "z") -> pd.Series:
    return df.groupby("asset")[score_col].mean().dropna()


# ============================================================
# 6) Core evaluation driver
# ============================================================
def evaluate_factor_core(
        df: pd.DataFrame,
        factor_col: str = "factor",
        ret_col: str = "fwd_ret",
        eligible_col: str | None = "eligible",
        n_q: int = 5,
        min_names: int = 15,
        bars_per_year: float = 365.0,
        rolling_years: float = 2.0,
) -> dict:
    """
    Computes core tear-sheet components.
    Returns a dict (work df + timeseries frames/series).
    """
    work = df.copy()

    # Optional eligibility filter
    if eligible_col is not None and eligible_col in work.columns:
        work = work[work[eligible_col].astype(bool)].copy()

    # Ensure required cols exist
    for c in ["time", "asset", factor_col, ret_col]:
        if c not in work.columns:
            raise KeyError(f"Missing required column: {c}")

    # Z-score and quintiles
    work = add_cs_zscore(work, factor_col=factor_col, out_col="z")
    work = assign_quantiles(work, score_col="z", n_q=n_q, min_names=min_names, out_col="q")

    # Quintile returns
    qret_ew = quintile_returns_equal_weight(work, q_col="q", ret_col=ret_col, n_q=n_q)
    qret_aw = quintile_returns_alpha_weighted(work, q_col="q", score_col="z", ret_col=ret_col, n_q=n_q)

    # Spread
    spread_ew = top_minus_bottom(qret_ew, top=f"Q{n_q}", bottom="Q1").rename("Q5-Q1_EW")
    spread_aw = top_minus_bottom(qret_aw, top=f"Q{n_q}", bottom="Q1").rename("Q5-Q1_AW")

    # Rolling IR (adaptive window)
    preferred = int(round(rolling_years * bars_per_year))

    n_ew = int(spread_ew.dropna().shape[0])
    n_aw = int(spread_aw.dropna().shape[0])
    n_points = max(1, min(n_ew, n_aw))

    window = choose_rolling_window(n_points=n_points, preferred=preferred, min_w=30, max_frac=0.6)
    min_periods = max(10, int(0.6 * window))

    ir_ew = rolling_sharpe(spread_ew, window=window, ann_factor=bars_per_year, min_periods=min_periods).rename(
        "RollingIR_EW")
    ir_aw = rolling_sharpe(spread_aw, window=window, ann_factor=bars_per_year, min_periods=min_periods).rename(
        "RollingIR_AW")

    # Diagnostics
    q_lines = cs_quantile_lines(work, score_col="z")
    mean_score_by_asset = per_asset_mean_score(work, score_col="z")

    # Count of names (after eligibility filter; before min_names filter)
    counts = work.groupby("time")["asset"].nunique().rename("n_names").sort_index()

    # Keep fwd_ret name consistent for downstream stats (factor_summary_stats uses fwd_ret)
    if ret_col != "fwd_ret":
        work = work.rename(columns={ret_col: "fwd_ret"})

    return {
        "work": work,  # includes z, q, fwd_ret
        "counts": counts,
        "qret_ew": qret_ew,
        "qret_aw": qret_aw,
        "spread_ew": spread_ew,
        "spread_aw": spread_aw,
        "rolling_ir_ew": ir_ew,
        "rolling_ir_aw": ir_aw,
        "score_quantile_lines": q_lines,
        "mean_score_by_asset": mean_score_by_asset,
        "window": window,
        "bars_per_year": float(bars_per_year),
    }


# ============================================================
# 7) Summary stats + turnover (your current section, fixed/improved)
# ============================================================
def _t_stat(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 2:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(mu / (sd / np.sqrt(n)))


def _sharpe_like(x: pd.Series, ann_factor: float) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float((mu / sd) * np.sqrt(ann_factor))


def _ann_vol(x: pd.Series, ann_factor: float) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return np.nan
    return float(x.std(ddof=1) * np.sqrt(ann_factor))


def _spearman_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.Series(a)
    b = pd.Series(b)
    return float(a.corr(b, method="spearman"))


def ic_time_series(
        work: pd.DataFrame,
        score_col: str = "z",
        ret_col: str = "fwd_ret",
        min_names: int = 15,
        method: str = "spearman",
) -> pd.Series:
    """
    Per-time cross-sectional correlation between score and fwd return.
    """

    def _ic(g: pd.DataFrame) -> float:
        g = g[[score_col, ret_col]].dropna()
        if len(g) < min_names:
            return np.nan
        return g[score_col].corr(g[ret_col], method=method)

    return work.groupby("time", sort=True).apply(_ic).rename("IC").sort_index()


def _topq_weights_for_time(
        g: pd.DataFrame,
        top_q: int,
        q_col: str,
        score_col: str,
        weight_mode: str,
) -> pd.Series:
    """
    Returns weights indexed by asset for a single time slice.
    weight_mode: 'ew' or 'aw' (abs(z) within top quintile)
    """
    g = g[g[q_col] == top_q].copy()
    if g.empty:
        return pd.Series(dtype=float)

    g = g.drop_duplicates(subset=["asset"], keep="last")

    if weight_mode == "ew":
        w = np.full(len(g), 1.0 / len(g))
        return pd.Series(w, index=g["asset"].values)

    if weight_mode == "aw":
        z = g[score_col].abs().astype(float)
        s = float(z.sum())
        if s <= 0 or np.isnan(s):
            return pd.Series(dtype=float)
        return pd.Series((z / s).values, index=g["asset"].values)

    raise ValueError("weight_mode must be 'ew' or 'aw'")


def topq_turnover_series(
        work: pd.DataFrame,
        top_q: int = 5,
        q_col: str = "q",
        score_col: str = "z",
        weight_mode: str = "ew",
) -> pd.Series:
    """
    Standard one-way turnover: 0.5 * sum(|w_t - w_{t-1}|)
    """
    times = pd.Index(sorted(work["time"].dropna().unique()))
    prev = None
    vals = []

    for t in times:
        g = work[work["time"] == t]
        w = _topq_weights_for_time(g, top_q=top_q, q_col=q_col, score_col=score_col, weight_mode=weight_mode)

        if prev is None:
            vals.append(np.nan)
        else:
            idx = prev.index.union(w.index)
            tv = 0.5 * (w.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)).abs().sum()
            vals.append(float(tv))

        prev = w

    return pd.Series(vals, index=times, name=f"topQ_turnover_{weight_mode}").sort_index()


def factor_summary_stats(
        out: dict,
        ann_factor: float,
        cost_bps_assumed: float = 20.0,
        min_names_for_ic: int = 15,
        use_quantile_returns: str = "ew",  # 'ew' or 'aw'
) -> pd.Series:
    """
    Produces a metric block similar to your example.
    """
    work = out["work"]
    counts = out["counts"]
    window = out["window"]

    qret = out["qret_ew"] if use_quantile_returns == "ew" else out["qret_aw"]
    spread = out["spread_ew"] if use_quantile_returns == "ew" else out["spread_aw"]

    # IC
    ic = ic_time_series(work, score_col="z", ret_col="fwd_ret", min_names=min_names_for_ic, method="spearman")
    ic_mean = float(ic.mean(skipna=True))
    ic_t = _t_stat(ic)
    ic_sd = float(ic.std(ddof=1))
    ic_ir = float(ic_mean / ic_sd) if (ic_sd > 0 and not np.isnan(ic_sd)) else np.nan

    # Monotonicity (Spearman between quintile index and mean returns by quintile)
    q_means = qret.mean()
    q_means = q_means.reindex([f"Q{i}" for i in range(1, 6) if f"Q{i}" in q_means.index])
    mono = _spearman_corr(pd.Series(range(1, len(q_means) + 1)), q_means.values)

    # Spread stats
    spread_mean = float(spread.mean(skipna=True))
    spread_t = _t_stat(spread)
    spread_ir = _sharpe_like(spread, ann_factor=ann_factor)
    spread_risk = _ann_vol(spread, ann_factor=ann_factor)

    # Top quintile stats
    topq = qret["Q5"]
    topq_mean = float(topq.mean(skipna=True))
    topq_sharpe = _sharpe_like(topq, ann_factor=ann_factor)
    topq_risk = _ann_vol(topq, ann_factor=ann_factor)

    # Turnover (top quintile portfolio)
    turn = topq_turnover_series(work, top_q=5, q_col="q", score_col="z", weight_mode=use_quantile_returns)
    topq_turn_mean = float(turn.mean(skipna=True))

    # Net of costs
    cost_per_period = (cost_bps_assumed / 1e4) * turn.reindex(topq.index)
    topq_net = (topq - cost_per_period).rename("topQ_net")
    topq_net_mean = float(topq_net.mean(skipna=True))
    topq_net_sharpe = _sharpe_like(topq_net, ann_factor=ann_factor)

    # Counts
    n_rebalance_times = int(qret.dropna(how="all").shape[0])
    eligible_med = float(counts.median(skipna=True))
    eligible_min = float(counts.min(skipna=True))

    return pd.Series({
        "n_rebalance_times": n_rebalance_times,
        "eligible_names_median": eligible_med,
        "eligible_names_min": eligible_min,
        "window": window,
        "IC_mean": ic_mean,
        "IC_t": ic_t,
        "ICIR": ic_ir,
        "quintile_monotonicity_spearman": mono,
        "Q5_minus_Q1_mean": spread_mean,
        "Q5_minus_Q1_t": spread_t,
        "spread_IR_or_Sharpe": spread_ir,
        "spread_Risk_ann_vol": spread_risk,
        "topQ_mean": topq_mean,
        "topQ_sharpe_like": topq_sharpe,
        "topQ_risk_ann_vol": topq_risk,
        "topQ_turnover_mean": topq_turn_mean,
        "cost_bps_assumed": float(cost_bps_assumed),
        "topQ_net_mean": topq_net_mean,
        "topQ_net_sharpe_like": topq_net_sharpe,
    })


# ============================================================
# 8) One EW vs AW summary table
# ============================================================
def factor_summary_table(stats_ew: pd.Series, stats_aw: pd.Series) -> pd.DataFrame:
    """
    2-col table: EqualWeight vs AlphaWeight
    """
    df = pd.concat(
        [stats_ew.rename("EqualWeight"), stats_aw.rename("AlphaWeight")],
        axis=1
    )
    df = df.loc[stats_ew.index]  # preserve EW order
    return df


def print_factor_summary_table(df: pd.DataFrame, title: str | None = None, float_fmt: str = "{: .6f}"):
    if title:
        print(title)
        print("-" * len(title))

    def _fmt(v):
        if pd.isna(v):
            return " NaN"
        v = float(v)
        if v.is_integer() and abs(v) >= 1:
            return f"{int(v)}"
        return float_fmt.format(v)

    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(_fmt)

    idx_w = max(len(str(i)) for i in out.index)
    c1, c2 = out.columns[0], out.columns[1]
    c_w = max(len(c1), len(c2), 16)

    print(f"{'metric':<{idx_w}}  {c1:>{c_w}}  {c2:>{c_w}}")
    print(f"{'-' * idx_w}  {'-' * c_w}  {'-' * c_w}")
    for k, row in out.iterrows():
        print(f"{k:<{idx_w}}  {row[c1]:>{c_w}}  {row[c2]:>{c_w}}")


# ============================================================
# 9) Clean tear-sheet grid plots (2 columns, larger)
# ============================================================
def _cum_simple_df(r: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + r.fillna(0.0)).cumprod() - 1.0


def terminal_cum_returns(qret: pd.DataFrame) -> pd.Series:
    """
    Terminal cumulative simple return for each quintile column (Q1..Q5).
    Equivalent to the last point on the cumulative-return plot.
    """
    cum = (1.0 + qret.fillna(0.0)).cumprod() - 1.0
    if len(cum) == 0:
        return pd.Series({c: np.nan for c in qret.columns})
    return cum.iloc[-1]


def plot_factor_tearsheet_grid(
        out: dict,
        title: str = "",
        ncols: int = 2,
        figsize_per_row: tuple[int, int] = (13, 4),  # wider + taller per row
):
    """
    8 panels, 2 columns:
      1) eligible counts
      2) cumulative quintiles EW
      3) cumulative quintiles AW
      4) cumulative spread EW vs AW
      5) rolling IR EW vs AW
      6) bar: mean returns by quintile (EW)
      7) score quantile lines over time
      8) histogram: mean score by asset
    """
    panels = 8
    nrows = int(np.ceil(panels / ncols))
    fig_w, row_h = figsize_per_row

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, row_h * nrows),
        constrained_layout=True
    )
    axes = np.array(axes).reshape(-1)

    def _style(ax):
        ax.grid(True, alpha=0.25)

    # 1) counts
    ax = axes[0]
    out["counts"].plot(ax=ax, lw=1.6)
    ax.set_title("Eligible names (count)")
    _style(ax)

    # 2) cumulative quintiles (EW)
    ax = axes[1]
    _cum_simple_df(out["qret_ew"]).plot(ax=ax, lw=1.6)
    ax.set_title("Cumulative returns by quintile (Equal Weight)")
    _style(ax)

    # 3) cumulative quintiles (AW)
    ax = axes[2]
    _cum_simple_df(out["qret_aw"]).plot(ax=ax, lw=1.6)
    ax.set_title("Cumulative returns by quintile (Alpha Weight)")
    _style(ax)

    # 4) cumulative spread (EW vs AW)
    ax = axes[3]
    spreads = pd.DataFrame({"Spread_EW": out["spread_ew"], "Spread_AW": out["spread_aw"]})
    _cum_simple_df(spreads).plot(ax=ax, lw=1.6)
    ax.set_title("Cumulative Q5–Q1 spread (EW vs AW)")
    _style(ax)

    # 5) rolling IR
    ax = axes[4]
    pd.DataFrame({"RollingIR_EW": out["rolling_ir_ew"], "RollingIR_AW": out["rolling_ir_aw"]}).plot(ax=ax, lw=1.6)
    ax.axhline(0.0, lw=1.0)
    ax.set_title("Rolling IR / Sharpe of spread (EW vs AW)")
    _style(ax)

    # 6) Quintile bars (EW): Mean (left axis) vs Terminal cumulative (right axis)
    ax = axes[5]

    qret = out["qret_ew"].copy()
    means = qret.mean().reindex([f"Q{i}" for i in range(1, 6)])
    terminal = terminal_cum_returns(qret).reindex([f"Q{i}" for i in range(1, 6)])

    x = np.arange(len(means))
    bar_w = 0.38

    # Left axis: mean forward return
    ax.bar(x - bar_w / 2, means.values, width=bar_w, label="Mean_fwd_ret")
    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Mean forward return")
    ax.grid(True, axis="y", alpha=0.25)

    # Right axis: terminal cumulative return
    ax2 = ax.twinx()
    ax2.bar(x + bar_w / 2, terminal.values, width=bar_w, label="Terminal_cum_ret", color="orange")
    ax2.set_ylabel("Terminal cumulative return")

    ax.set_title("Quintile bars (EW): Mean vs Terminal cumulative")

    # Combined legend (from both axes)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    # 7) score quantile lines
    ax = axes[6]
    out["score_quantile_lines"].plot(ax=ax, lw=1.6)
    ax.axhline(0.0, lw=1.0)
    ax.set_title("Cross-sectional score quantiles over time")
    _style(ax)

    # 8) histogram
    ax = axes[7]
    out["mean_score_by_asset"].plot(kind="hist", bins=50, ax=ax)
    ax.set_title("Distribution: mean z-score by asset")
    ax.grid(True, axis="y", alpha=0.25)

    # reduce legend clutter
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            plt.setp(leg.get_texts(), fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.show()

# ============================================================
# 10) Example usage
# ============================================================
# df_eval = prep_factor_df(df_reb, asset_col="altname", factor_col="mom_42", ret_col="fwd_6")
#
# out = evaluate_factor_core(
#     df_eval,
#     factor_col="factor",
#     ret_col="fwd_ret",
#     eligible_col="eligible" if "eligible" in df_eval.columns else None,
#     n_q=5,
#     min_names=20,
#     bars_per_year=365,     # use 365*6 only if 4h bars
#     rolling_years=2.0
# )
#
# stats_ew = factor_summary_stats(out, ann_factor=out["bars_per_year"], cost_bps_assumed=20.0, use_quantile_returns="ew")
# stats_aw = factor_summary_stats(out, ann_factor=out["bars_per_year"], cost_bps_assumed=20.0, use_quantile_returns="aw")
# tbl = factor_summary_table(stats_ew, stats_aw)
# print_factor_summary_table(tbl, title="=== Factor Test Summary (EW vs AW) ===")
#
# plot_factor_tearsheet_grid(out, title="Factor tear sheet")