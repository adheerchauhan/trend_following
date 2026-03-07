import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0) Small utilities
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


def _infer_bars_per_day(bar_hours: int) -> int:
    return int(round(24 / bar_hours))


# ============================================================
# 1) Optional adapter
# ============================================================
def prep_factor_df(
    df: pd.DataFrame,
    time_col: str = "time",
    asset_col: str = "altname",
    factor_col: str = "factor",
    ret_col: str = "fwd_ret",
    eligible_col: str | None = None,
) -> pd.DataFrame:
    keep = [time_col, asset_col, factor_col, ret_col]
    if eligible_col is not None and eligible_col in df.columns:
        keep.append(eligible_col)

    out = df[keep].copy()
    rename_map = {
        time_col: "time",
        asset_col: "asset",
        factor_col: "factor",
        ret_col: "fwd_ret",
    }
    if eligible_col is not None and eligible_col in out.columns:
        rename_map[eligible_col] = "eligible"

    out = out.rename(columns=rename_map)
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["asset"] = out["asset"].astype(str)
    out["factor"] = pd.to_numeric(out["factor"], errors="coerce")
    out["fwd_ret"] = pd.to_numeric(out["fwd_ret"], errors="coerce")

    if "eligible" in out.columns:
        out["eligible"] = out["eligible"].astype(bool)

    return out


# ============================================================
# 2) Cross-sectional standardization
# ============================================================
def cs_robust_zscore(x: pd.Series, clip: float = 6.0) -> pd.Series:
    x = x.astype(float)
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.nan, index=x.index)

    z = (x - med) / (1.4826 * mad)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


def add_cs_zscore(df: pd.DataFrame, factor_col: str = "factor", out_col: str = "z") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = out.groupby("time")[factor_col].transform(cs_robust_zscore)
    return out


# ============================================================
# 3) Quintiles
# ============================================================
def assign_quantiles(
    df: pd.DataFrame,
    score_col: str = "z",
    n_q: int = 5,
    min_names: int = 15,
    out_col: str = "q",
) -> pd.DataFrame:
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

    out = df.copy()
    out[out_col] = out.groupby("time")[score_col].transform(_qcut_ranked)
    return out


# ============================================================
# 4) Quintile return construction
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
    out = (
        df.dropna(subset=[q_col, ret_col])
          .groupby(["time", q_col])[ret_col]
          .mean()
          .unstack(q_col)
          .reindex(columns=range(1, n_q + 1))
          .sort_index()
    )
    out.columns = [f"Q{int(c)}" for c in out.columns]
    return out


def quintile_returns_alpha_weighted(
    df: pd.DataFrame,
    q_col: str = "q",
    score_col: str = "z",
    ret_col: str = "fwd_ret",
    n_q: int = 5,
) -> pd.DataFrame:
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
# 5) Rolling IR / Sharpe
# ============================================================
def rolling_sharpe(ret: pd.Series, window: int, ann_factor: float, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(10, int(0.6 * window))
        min_periods = min(min_periods, window)

    mu = ret.rolling(window, min_periods=min_periods).mean()
    sd = ret.rolling(window, min_periods=min_periods).std(ddof=1)
    return (mu / sd) * np.sqrt(ann_factor)


def choose_rolling_window(n_points: int, preferred: int, min_w: int = 30, max_frac: float = 0.6) -> int:
    if n_points <= 1:
        return 1

    min_w_eff = min(min_w, max(5, n_points // 3))
    max_w = max(min_w_eff, int(max_frac * n_points))

    w = int(preferred)
    w = max(min_w_eff, min(w, max_w))
    return w


# ============================================================
# 6) Diagnostics
# ============================================================
def cs_quantile_lines(
    df: pd.DataFrame,
    score_col: str = "z",
    probs=(0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99),
) -> pd.DataFrame:
    def _q(x: pd.Series) -> pd.Series:
        x = x.dropna()
        if len(x) == 0:
            return pd.Series({p: np.nan for p in probs})
        return x.quantile(probs)

    out = df.groupby("time")[score_col].apply(_q).unstack()
    out.columns = [f"q{p:g}" for p in out.columns]
    return out.sort_index()


def per_asset_mean_score(df: pd.DataFrame, score_col: str = "z") -> pd.Series:
    return df.groupby("asset")[score_col].mean().dropna()


def ic_time_series(
    work: pd.DataFrame,
    score_col: str = "z",
    ret_col: str = "fwd_ret",
    min_names: int = 15,
    method: str = "spearman",
) -> pd.Series:
    def _ic(g: pd.DataFrame) -> float:
        g = g[[score_col, ret_col]].dropna()
        if len(g) < min_names:
            return np.nan
        return g[score_col].corr(g[ret_col], method=method)

    return work.groupby("time", sort=True).apply(_ic).rename("IC").sort_index()


# ============================================================
# 7) Core factor evaluation
# ============================================================
def evaluate_factor_core(
    df: pd.DataFrame,
    factor_col: str = "factor",
    ret_col: str = "fwd_ret",
    eligible_col: str | None = None,
    n_q: int = 5,
    min_names: int = 15,
    bars_per_year: float = 365.0,
    rolling_years: float = 0.5,
) -> dict:
    work = df.copy()

    if eligible_col is not None and eligible_col in work.columns:
        work = work[work[eligible_col].astype(bool)].copy()

    for c in ["time", "asset", factor_col, ret_col]:
        if c not in work.columns:
            raise KeyError(f"Missing required column: {c}")

    work = add_cs_zscore(work, factor_col=factor_col, out_col="z")
    work = assign_quantiles(work, score_col="z", n_q=n_q, min_names=min_names, out_col="q")

    qret_ew = quintile_returns_equal_weight(work, q_col="q", ret_col=ret_col, n_q=n_q)
    qret_aw = quintile_returns_alpha_weighted(work, q_col="q", score_col="z", ret_col=ret_col, n_q=n_q)

    spread_ew = top_minus_bottom(qret_ew, top=f"Q{n_q}", bottom="Q1").rename("Q5-Q1_EW")
    spread_aw = top_minus_bottom(qret_aw, top=f"Q{n_q}", bottom="Q1").rename("Q5-Q1_AW")

    preferred = int(round(rolling_years * bars_per_year))
    n_ew = int(spread_ew.dropna().shape[0])
    n_aw = int(spread_aw.dropna().shape[0])
    n_points = max(1, min(n_ew, n_aw))

    window = choose_rolling_window(n_points=n_points, preferred=preferred, min_w=30, max_frac=0.6)
    min_periods = max(10, int(0.6 * window))

    rolling_ir_ew = rolling_sharpe(spread_ew, window=window, ann_factor=bars_per_year, min_periods=min_periods)
    rolling_ir_aw = rolling_sharpe(spread_aw, window=window, ann_factor=bars_per_year, min_periods=min_periods)

    counts = work.groupby("time")["asset"].nunique().rename("n_names").sort_index()
    score_quantile_lines = cs_quantile_lines(work, score_col="z")
    mean_score_by_asset = per_asset_mean_score(work, score_col="z")

    if ret_col != "fwd_ret":
        work = work.rename(columns={ret_col: "fwd_ret"})

    return {
        "work": work,
        "counts": counts,
        "qret_ew": qret_ew,
        "qret_aw": qret_aw,
        "spread_ew": spread_ew,
        "spread_aw": spread_aw,
        "rolling_ir_ew": rolling_ir_ew.rename("RollingIR_EW"),
        "rolling_ir_aw": rolling_ir_aw.rename("RollingIR_AW"),
        "score_quantile_lines": score_quantile_lines,
        "mean_score_by_asset": mean_score_by_asset,
        "window": window,
        "bars_per_year": float(bars_per_year),
    }


# ============================================================
# 8) Turnover + summary stats
# ============================================================
def _topq_weights_for_time(
    g: pd.DataFrame,
    top_q: int,
    q_col: str,
    score_col: str,
    weight_mode: str,
) -> pd.Series:
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
    use_quantile_returns: str = "ew",
) -> pd.Series:
    work = out["work"]
    counts = out["counts"]
    window = out["window"]

    qret = out["qret_ew"] if use_quantile_returns == "ew" else out["qret_aw"]
    spread = out["spread_ew"] if use_quantile_returns == "ew" else out["spread_aw"]

    ic = ic_time_series(work, score_col="z", ret_col="fwd_ret", min_names=min_names_for_ic, method="spearman")
    ic_mean = float(ic.mean(skipna=True))
    ic_t = _t_stat(ic)
    ic_sd = float(ic.std(ddof=1))
    ic_ir = float(ic_mean / ic_sd) if (ic_sd > 0 and not np.isnan(ic_sd)) else np.nan

    q_means = qret.mean().reindex([f"Q{i}" for i in range(1, 6)])
    mono = _spearman_corr(pd.Series(range(1, len(q_means) + 1)), q_means.values)

    spread_mean = float(spread.mean(skipna=True))
    spread_t = _t_stat(spread)
    spread_ir = _sharpe_like(spread, ann_factor=ann_factor)
    spread_risk = _ann_vol(spread, ann_factor=ann_factor)

    topq = qret["Q5"]
    topq_mean = float(topq.mean(skipna=True))
    topq_sharpe = _sharpe_like(topq, ann_factor=ann_factor)
    topq_risk = _ann_vol(topq, ann_factor=ann_factor)

    turn = topq_turnover_series(work, top_q=5, q_col="q", score_col="z", weight_mode=use_quantile_returns)
    topq_turn_mean = float(turn.mean(skipna=True))

    cost_per_period = (cost_bps_assumed / 1e4) * turn.reindex(topq.index)
    topq_net = (topq - cost_per_period).rename("topQ_net")
    topq_net_mean = float(topq_net.mean(skipna=True))
    topq_net_sharpe = _sharpe_like(topq_net, ann_factor=ann_factor)

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
# 9) Summary table helpers
# ============================================================
def factor_summary_table(stats_ew: pd.Series, stats_aw: pd.Series) -> pd.DataFrame:
    out = pd.concat(
        [stats_ew.rename("EqualWeight"), stats_aw.rename("AlphaWeight")],
        axis=1
    )
    return out.loc[stats_ew.index]


def _format_metric_value(v, float_fmt: str = "{: .6f}") -> str:
    if pd.isna(v):
        return " NaN"
    v = float(v)
    if v.is_integer() and abs(v) >= 1:
        return f"{int(v)}"
    return float_fmt.format(v)


def summary_table_to_text(
    df: pd.DataFrame,
    title: str | None = None,
    float_fmt: str = "{: .6f}",
) -> str:
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

    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))

    lines.append(f"{'metric':<{idx_w}}  {c1:>{c_w}}  {c2:>{c_w}}")
    lines.append(f"{'-' * idx_w}  {'-' * c_w}  {'-' * c_w}")

    for k, row in out.iterrows():
        lines.append(f"{k:<{idx_w}}  {row[c1]:>{c_w}}  {row[c2]:>{c_w}}")

    return "\n".join(lines)


def print_factor_summary_table(df: pd.DataFrame, title: str | None = None, float_fmt: str = "{: .6f}"):
    print(summary_table_to_text(df, title=title, float_fmt=float_fmt))


# ============================================================
# 10) IC term structure
# ============================================================
def ic_term_structure(
    df: pd.DataFrame,
    factor_col: str,
    horizon_bars_list: list[int],
    bar_hours: int = 4,
    ret_prefix: str = "fwd_",
    ret_suffix: str = "_simple",
    method: str = "spearman",
    min_names: int = 15,
    decision_every_n_days: int = 1,
    non_overlapping: bool = True,
) -> pd.DataFrame:
    cols = ["time", "asset", factor_col] + [f"{ret_prefix}{h}{ret_suffix}" for h in horizon_bars_list]
    work = df[cols].copy()
    work = add_cs_zscore(work, factor_col=factor_col, out_col="z")

    bars_per_day = _infer_bars_per_day(bar_hours)
    rows = []

    for h in horizon_bars_list:
        ret_col = f"{ret_prefix}{h}{ret_suffix}"
        ic = ic_time_series(work, score_col="z", ret_col=ret_col, min_names=min_names, method=method)

        if non_overlapping:
            h_days = max(1, int(round(h / bars_per_day)))
            step = max(1, int(round(h_days / decision_every_n_days)))
            ic_used = ic.iloc[::step]
        else:
            ic_used = ic

        ic_mean = float(ic_used.mean(skipna=True))
        ic_t = _t_stat(ic_used)
        ic_sd = float(ic_used.std(ddof=1))
        icir = float(ic_mean / ic_sd) if (ic_sd > 0 and not np.isnan(ic_sd)) else np.nan
        n_obs = int(ic_used.dropna().shape[0])

        rows.append({
            "horizon_bars": h,
            "horizon_days": h / bars_per_day,
            "IC_mean": ic_mean,
            "IC_t": ic_t,
            "ICIR": icir,
            "n_obs": n_obs,
        })

    out = pd.DataFrame(rows).set_index("horizon_days").sort_index()
    return out


# ============================================================
# 11) Quantile / monotonicity term structure (optional but useful)
# ============================================================
def quantile_term_structure(
    df: pd.DataFrame,
    factor_col: str,
    horizon_bars_list: list[int],
    bar_hours: int = 4,
    weight_mode: str = "ew",
    n_q: int = 5,
    min_names: int = 15,
    ret_prefix: str = "fwd_",
    ret_suffix: str = "_simple",
) -> pd.DataFrame:
    cols = ["time", "asset", factor_col] + [f"{ret_prefix}{h}{ret_suffix}" for h in horizon_bars_list]
    work = df[cols].copy()
    work = add_cs_zscore(work, factor_col=factor_col, out_col="z")
    work = assign_quantiles(work, score_col="z", n_q=n_q, min_names=min_names, out_col="q")

    bars_per_day = _infer_bars_per_day(bar_hours)
    rows = []

    for h in horizon_bars_list:
        ret_col = f"{ret_prefix}{h}{ret_suffix}"

        if weight_mode == "ew":
            qret = quintile_returns_equal_weight(work, q_col="q", ret_col=ret_col, n_q=n_q)
        elif weight_mode == "aw":
            qret = quintile_returns_alpha_weighted(work, q_col="q", score_col="z", ret_col=ret_col, n_q=n_q)
        else:
            raise ValueError("weight_mode must be 'ew' or 'aw'")

        qmeans = qret.mean().reindex([f"Q{i}" for i in range(1, n_q + 1)])
        mono = _spearman_corr(pd.Series(range(1, n_q + 1)), qmeans.values)

        row = {
            "horizon_bars": h,
            "horizon_days": h / bars_per_day,
            "monotonicity_spearman": mono,
            "Q5_minus_Q1": float(qmeans["Q5"] - qmeans["Q1"]),
        }
        for i in range(1, n_q + 1):
            row[f"Q{i}"] = float(qmeans[f"Q{i}"])
        rows.append(row)

    return pd.DataFrame(rows).set_index("horizon_days").sort_index()


# ============================================================
# 12) Plot helpers
# ============================================================
def _cum_simple_df(r: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + r.fillna(0.0)).cumprod() - 1.0


def terminal_cum_returns(qret: pd.DataFrame) -> pd.Series:
    cum = (1.0 + qret.fillna(0.0)).cumprod() - 1.0
    if len(cum) == 0:
        return pd.Series({c: np.nan for c in qret.columns})
    return cum.iloc[-1]


def plot_ic_term_structure(ax, ic_ts: pd.DataFrame, title: str = "IC term structure"):
    if ic_ts is None or ic_ts.empty:
        ax.set_title(title + " (no data)")
        return

    x = ic_ts.index.values

    # Left axis: IC mean
    ax.plot(
        x,
        ic_ts["IC_mean"].values,
        marker="o",
        lw=1.7,
        color="tab:blue",
        label="IC_mean",
    )
    ax.axhline(0.0, lw=1.0, color="gray")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Mean IC")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # Right axis: ICIR
    ax2 = ax.twinx()
    ax2.plot(
        x,
        ic_ts["ICIR"].values,
        marker="s",
        lw=1.5,
        linestyle="--",
        color="tab:orange",
        label="ICIR",
    )
    ax2.set_ylabel("ICIR")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")


def plot_quantile_term_structure(
    ax,
    qts: pd.DataFrame | None,
    title: str = "Quintile / spread term structure (EW)",
):
    if qts is None or qts.empty:
        ax.set_title(title + " (no data)")
        return

    qcols = [c for c in ["Q1", "Q2", "Q3", "Q4", "Q5"] if c in qts.columns]
    x = qts.index.values

    qts[qcols].plot(ax=ax, lw=1.4, marker="o")
    ax.axhline(0.0, lw=1.0, color="gray")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Mean quintile return")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    if "Q5_minus_Q1" in qts.columns:
        ax2 = ax.twinx()
        ax2.plot(
            x,
            qts["Q5_minus_Q1"].values,
            lw=1.6,
            linestyle="--",
            marker="s",
            color="black",
            label="Q5-Q1",
        )
        ax2.set_ylabel("Q5-Q1 spread")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    else:
        ax.legend(fontsize=8, loc="best")


# ============================================================
# 13) Main tear-sheet plot
# ============================================================
def plot_factor_tearsheet_grid(
    out: dict,
    title: str = "",
    ic_ts: pd.DataFrame | None = None,
    qts_ew: pd.DataFrame | None = None,
    summary_table: pd.DataFrame | None = None,
    summary_title: str | None = None,
    figsize: tuple[int, int] = (18, 21),
):
    """
    Layout:
      Summary row (optional)
      Row 1: cumulative Q5-Q1 spread | eligible names
      Row 2: cumulative quintiles EW | cumulative quintiles AW
      Row 3: rolling IR              | quintile bars
      Row 4: score quantiles         | score distribution
      Row 5: IC term structure       | quintile/spread term structure
    """
    has_summary = summary_table is not None

    if has_summary:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(
            nrows=6,
            ncols=2,
            height_ratios=[2.4, 4.2, 4.2, 4.2, 4.2, 4.2],
        )

        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis("off")
        txt = summary_table_to_text(summary_table, title=summary_title)
        ax_text.text(
            0.0,
            0.98,
            txt,
            ha="left",
            va="top",
            family="monospace",
            fontsize=8.5,
        )
        row_offset = 1
    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1] - 3), constrained_layout=True)
        gs = fig.add_gridspec(nrows=5, ncols=2)
        row_offset = 0

    axes = []
    for r in range(5):
        for c in range(2):
            axes.append(fig.add_subplot(gs[r + row_offset, c]))

    def _style(ax):
        ax.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # Row 1, Col 1: cumulative spread
    # --------------------------------------------------------
    ax = axes[0]
    spreads = pd.DataFrame({
        "Spread_EW": out["spread_ew"],
        "Spread_AW": out["spread_aw"],
    })
    _cum_simple_df(spreads).plot(ax=ax, lw=1.5)
    ax.set_title("Cumulative Q5–Q1 spread (EW vs AW)")
    ax.set_xlabel("")
    _style(ax)

    # --------------------------------------------------------
    # Row 1, Col 2: eligible names
    # --------------------------------------------------------
    ax = axes[1]
    out["counts"].plot(ax=ax, lw=1.5)
    ax.set_title("Eligible names (count)")
    ax.set_xlabel("")
    _style(ax)

    # --------------------------------------------------------
    # Row 2, Col 1: cumulative quintiles EW
    # --------------------------------------------------------
    ax = axes[2]
    _cum_simple_df(out["qret_ew"]).plot(ax=ax, lw=1.5)
    ax.set_title("Cumulative returns by quintile (Equal Weight)")
    ax.set_xlabel("")
    _style(ax)

    # --------------------------------------------------------
    # Row 2, Col 2: cumulative quintiles AW
    # --------------------------------------------------------
    ax = axes[3]
    _cum_simple_df(out["qret_aw"]).plot(ax=ax, lw=1.5)
    ax.set_title("Cumulative returns by quintile (Alpha Weight)")
    ax.set_xlabel("")
    _style(ax)

    # --------------------------------------------------------
    # Row 3, Col 1: rolling IR
    # --------------------------------------------------------
    ax = axes[4]
    pd.DataFrame({
        "RollingIR_EW": out["rolling_ir_ew"],
        "RollingIR_AW": out["rolling_ir_aw"],
    }).plot(ax=ax, lw=1.5)
    ax.axhline(0.0, lw=1.0, color="gray")
    ax.set_title("Rolling IR / Sharpe of spread (EW vs AW)")
    ax.set_xlabel("")
    _style(ax)

    # --------------------------------------------------------
    # Row 3, Col 2: quintile bars
    # --------------------------------------------------------
    ax = axes[5]
    qret = out["qret_ew"].copy()
    means = qret.mean().reindex([f"Q{i}" for i in range(1, 6)])
    terminal = terminal_cum_returns(qret).reindex([f"Q{i}" for i in range(1, 6)])

    x = np.arange(len(means))
    bar_w = 0.38

    ax.bar(x - bar_w / 2, means.values, width=bar_w, label="Mean_fwd_ret")
    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Mean forward return")
    ax.grid(True, axis="y", alpha=0.25)

    ax2 = ax.twinx()
    ax2.bar(
        x + bar_w / 2,
        terminal.values,
        width=bar_w,
        color="orange",
        label="Terminal_cum_ret",
    )
    ax2.set_ylabel("Terminal cumulative return")

    ax.set_title("Quintile bars (EW): Mean vs Terminal cumulative")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    # --------------------------------------------------------
    # Row 4, Col 1: score quantiles
    # --------------------------------------------------------
    ax = axes[6]
    out["score_quantile_lines"].plot(ax=ax, lw=1.4)
    ax.axhline(0.0, lw=1.0, color="gray")
    ax.set_title("Cross-sectional score quantiles over time")
    ax.set_xlabel("")
    _style(ax)

    # --------------------------------------------------------
    # Row 4, Col 2: distribution
    # --------------------------------------------------------
    ax = axes[7]
    out["mean_score_by_asset"].plot(kind="hist", bins=40, ax=ax)
    ax.set_title("Distribution: mean z-score by asset")
    ax.set_xlabel("Mean z-score")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.25)

    # --------------------------------------------------------
    # Row 5, Col 1: IC term structure
    # --------------------------------------------------------
    ax = axes[8]
    plot_ic_term_structure(ax, ic_ts, title="IC term structure")

    # --------------------------------------------------------
    # Row 5, Col 2: quintile/spread term structure
    # --------------------------------------------------------
    ax = axes[9]
    plot_quantile_term_structure(ax, qts_ew, title="Quintile / spread term structure (EW)")

    # --------------------------------------------------------
    # Tidy legends
    # --------------------------------------------------------
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            plt.setp(leg.get_texts(), fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, y=0.998)

    plt.show()