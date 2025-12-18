import os
import json
import smtplib
from datetime import datetime, timezone, date as date_cls
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple, Optional
from email.message import EmailMessage


# ---- Config (env vars) -------------------------------------------------------
SMTP_HOST     = os.getenv("TF_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("TF_SMTP_PORT", "587"))            # STARTTLS
SMTP_USER     = os.getenv("TF_SMTP_USER")                        # your email
SMTP_PASS     = os.getenv("TF_SMTP_PASS")                        # app password
EMAIL_FROM    = os.getenv("TF_EMAIL_FROM", SMTP_USER or "")
EMAIL_TO      = os.getenv("TF_EMAIL_TO")                         # comma-separated
EMAIL_ENABLED = bool(int(os.getenv("TF_EMAIL_ENABLED", "1")))    # 1/0 toggle

# How much detail to include
TOP_SIGNALS_N = int(os.getenv("TF_EMAIL_TOP_SIGNALS", "10"))     # 0 disables Top signals section
TOP_TRADES_N  = int(os.getenv("TF_EMAIL_TOP_TRADES", "10"))      # 0 disables Trades section


# ---- Helpers ----------------------------------------------------------------
def _parse_ts(ts: str) -> datetime:
    """Accept ISO strings with/without timezone; assume UTC if none."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_for_day(ts: str, day: date_cls) -> bool:
    try:
        return _parse_ts(ts).date() == day
    except Exception:
        return False


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _fmt_money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def _fmt_float(x, nd=4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "None"


def _daily_summary_path(state_dir: Path, day: date_cls) -> Path:
    return state_dir / "daily_summaries" / f"daily_summary_{day.isoformat()}.json"


def _load_daily_summary(state_dir: Path, day: date_cls) -> Optional[Dict[str, Any]]:
    p = _daily_summary_path(state_dir, day)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---- Daily summary normalizer ------------------------------------------------
def _normalize_daily_summary(d: Dict[str, Any], state_dir: Path, day: date_cls) -> Dict[str, Any]:
    """
    Normalize daily_summary_YYYY-MM-DD.json (written by write_daily_summary in live strategy)
    into an email-friendly dict.
    """
    run      = d.get("run", {}) or {}
    portfolio = d.get("portfolio", {}) or {}
    sleeves  = d.get("sleeves", {}) or {}
    tickers  = d.get("tickers", {}) or {}
    orders   = d.get("orders", {}) or {}
    errors   = d.get("errors", []) or []

    # Desired trade totals + fee estimate (from per-ticker block)
    buys = sells = zeros = 0
    fee_est_total = 0.0
    trades: List[Dict[str, Any]] = []
    signals: List[Dict[str, Any]] = []

    for t, td in (tickers or {}).items():
        if not isinstance(td, dict):
            continue

        trade_notional = float(td.get("trade_notional") or 0.0)
        fee_est = float(td.get("trade_fees_est") or td.get("trade_fees") or 0.0)
        fee_est_total += fee_est

        if trade_notional > 0:
            buys += 1
        elif trade_notional < 0:
            sells += 1
        else:
            zeros += 1

        # --- FIX: map to your live daily_summary key names ---
        # "signal strength": use vol_adj_weight (pre sleeve RB / pre final scaling)
        sig = td.get("vol_adj_weight")
        if sig is None:
            # backward compat if you ever rename
            sig = td.get("signal") or td.get("final_signal")

        try:
            sig_f = float(sig) if sig is not None else None
        except Exception:
            sig_f = None

        # "target_weight": use final_weight (post final scaling factor)
        tw = td.get("final_weight")
        if tw is None:
            tw = td.get("target_weight") or td.get("target_vol_adj_signal")

        try:
            tw_f = float(tw) if tw is not None else None
        except Exception:
            tw_f = None

        rbw = td.get("sleeve_risk_adj_weight")
        try:
            rbw_f = float(rbw) if rbw is not None else None
        except Exception:
            rbw_f = None

        # Top signals list (include even if no trade)
        if sig_f is not None:
            signals.append({
                "ticker": t,
                "sleeve": td.get("sleeve"),
                "signal": sig_f,
                "abs_signal": abs(sig_f),
                "target_weight": tw_f,
                "rb_weight": rbw_f,
                "target_notional": td.get("target_notional"),
            })

        # Trades list
        if abs(trade_notional) > 0:
            trades.append({
                "ticker": t,
                "sleeve": td.get("sleeve"),
                "trade_notional": trade_notional,
                "fees_est": fee_est,
                "reason": td.get("trade_reason"),
                "signal": sig_f,
                "target_weight": tw_f,
                "rb_weight": rbw_f,
                "target_notional": td.get("target_notional"),
                "open_notional": td.get("open_notional"),
            })

    # Stops summary (best-effort based on stop dict shape)
    replaced = skipped_no_pos = skipped_no_ratchet = stop_errs = 0
    for _, td in (tickers or {}).items():
        if not isinstance(td, dict):
            continue
        sd = td.get("stop") or {}
        if not isinstance(sd, dict):
            continue

        action = (sd.get("action") or "").lower()
        ok = sd.get("ok")
        reason = (sd.get("reason") or "").lower()

        if action == "replaced" and (ok is True or ok is None):
            replaced += 1
        elif action in ("no_position", "no_pos"):
            skipped_no_pos += 1
        elif action == "skip" and reason == "no_ratchet":
            skipped_no_ratchet += 1
        elif ok is False:
            stop_errs += 1

    # Sleeves list (keep simple + accurate; strategy writes allocation_share + pre_scale_weight_sum)
    sleeve_rows: List[Dict[str, Any]] = []
    if isinstance(sleeves, dict):
        for sname, sd in sleeves.items():
            if not isinstance(sd, dict):
                continue
            sleeve_rows.append({
                "sleeve": sname,
                "budget_weight": sd.get("budget_weight"),
                "allocation_share": sd.get("allocation_share", sd.get("realized_weight_share")),
                "pre_scale_weight_sum": sd.get("pre_scale_weight_sum"),
                "avg_multiplier": sd.get("avg_multiplier"),
            })

    # Sort + trim
    trades.sort(key=lambda x: abs(float(x.get("trade_notional") or 0.0)), reverse=True)
    signals.sort(key=lambda x: float(x.get("abs_signal") or 0.0), reverse=True)

    out = {
        "date": str(day),
        "run_id": run.get("run_id"),
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
        "dry_run": run.get("dry_run"),

        "portfolio": {
            "total_portfolio_value": portfolio.get("total_portfolio_value"),
            "available_cash": portfolio.get("available_cash"),
            "daily_portfolio_volatility": portfolio.get("daily_portfolio_volatility"),
            "final_scaling_factor": portfolio.get("final_scaling_factor"),
            "total_target_notional": portfolio.get("total_target_notional"),
            "total_actual_position_notional": portfolio.get("total_actual_position_notional"),
            "count_of_positions": portfolio.get("count_of_positions"),
        },

        "desired_totals": {"buys": buys, "sells": sells, "zeros": zeros},
        "fee_est_total": fee_est_total,

        "rebalance": {"built": int(orders.get("rebalance_built_count") or 0), "submitted": None, "preview": None},
        "dust": {"built": int(orders.get("dust_built_count") or 0), "submitted": None, "preview": None},
        "stops": {
            "replaced": replaced,
            "skipped_no_position": skipped_no_pos,
            "skipped_no_ratchet": skipped_no_ratchet,
            "errors": stop_errs,
        },

        "sleeves": sleeve_rows,
        "top_trades": trades[:max(TOP_TRADES_N, 0)] if TOP_TRADES_N > 0 else [],
        "top_signals": signals[:max(TOP_SIGNALS_N, 0)] if TOP_SIGNALS_N > 0 else [],

        # Keep tickers (from daily_summary) so we can render a compact per-ticker signals block if desired.
        "tickers": tickers,

        "errors": [
            {
                "when": e.get("ts") or e.get("when"),
                "where": e.get("where") or "unknown",
                "error": e.get("error"),
            }
            for e in (errors or [])
            if isinstance(e, dict)
        ],

        "state_dir": str(state_dir),
    }
    return out


# ---- JSONL fallback summarizer (only forensics; no per-ticker signals) --------
def summarize_run_jsonl(state_dir: Path, day: date_cls) -> Dict[str, Any]:
    """Reads JSONL logs and builds a compact summary for `day` (UTC)."""
    p = state_dir
    HEARTBEAT_LOG     = p / "heartbeat.jsonl"
    ERR_LOG           = p / "live_errors.jsonl"
    DESIRED_LOG       = p / "desired_trades_log.jsonl"
    ORDER_BUILD_LOG   = p / "order_build_log.jsonl"
    ORDER_SUBMIT_LOG  = p / "order_submit_log.jsonl"
    DUST_BUILD_LOG    = p / "dust_build_log.jsonl"
    DUST_SUBMIT_LOG   = p / "dust_submit_log.jsonl"
    STOP_UPDATE_LOG   = p / "stop_update_log.jsonl"

    out = {
        "date": str(day),
        "run_id": None,
        "started_at": None,
        "completed_at": None,
        "dry_run": None,
        "portfolio": {},
        "desired_totals": {"buys": 0, "sells": 0, "zeros": 0},
        "fee_est_total": 0.0,
        "rebalance": {"built": 0, "submitted": 0, "preview": None},
        "dust": {"built": 0, "submitted": 0, "preview": None},
        "stops": {"replaced": 0, "skipped_no_position": 0, "skipped_no_ratchet": 0, "errors": 0},
        "sleeves": [],
        "top_trades": [],
        "top_signals": [],
        "tickers": {},
        "errors": [],
        "state_dir": str(state_dir),
    }

    # heartbeat: times + dry_run + run_id
    for row in _iter_jsonl(HEARTBEAT_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            ev = row.get("event")
            if out["run_id"] is None and row.get("run_id"):
                out["run_id"] = row.get("run_id")
            if ev == "config_loaded":
                out["started_at"] = ts
                if "dry_run" in row:
                    out["dry_run"] = bool(row["dry_run"])
            elif ev == "run_complete":
                out["completed_at"] = ts

    # desired trades: counts (+ fees if present)
    buys = sells = zeros = 0
    fee_sum = 0.0
    for row in _iter_jsonl(DESIRED_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            nt = float(row.get("new_trade_notional") or 0.0)
            fee_sum += float(row.get("trade_fees") or 0.0)
            if nt > 0:
                buys += 1
            elif nt < 0:
                sells += 1
            else:
                zeros += 1
    out["desired_totals"] = {"buys": buys, "sells": sells, "zeros": zeros}
    out["fee_est_total"] = fee_sum

    # rebalance
    for row in _iter_jsonl(ORDER_BUILD_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            out["rebalance"]["built"] += 1
            if out["rebalance"]["preview"] is None and "preview" in row:
                out["rebalance"]["preview"] = bool(row["preview"])
    for row in _iter_jsonl(ORDER_SUBMIT_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            out["rebalance"]["submitted"] += int(row.get("orders_count") or 0)
            if out["rebalance"]["preview"] is None and "preview" in row:
                out["rebalance"]["preview"] = bool(row["preview"])

    # dust
    for row in _iter_jsonl(DUST_BUILD_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            out["dust"]["built"] += 1
            if out["dust"]["preview"] is None and "preview" in row:
                out["dust"]["preview"] = bool(row["preview"])
    for row in _iter_jsonl(DUST_SUBMIT_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            out["dust"]["submitted"] += int(row.get("orders_count") or 0)
            if out["dust"]["preview"] is None and "preview" in row:
                out["dust"]["preview"] = bool(row["preview"])

    # stops
    for row in _iter_jsonl(STOP_UPDATE_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            action = (row.get("action") or "").lower()
            ok = row.get("ok")
            reason = (row.get("reason") or "").lower()
            if action == "replaced" and (ok is True or ok is None):
                out["stops"]["replaced"] += 1
            elif action in ("no_position", "no_pos"):
                out["stops"]["skipped_no_position"] += 1
            elif action == "skip" and reason == "no_ratchet":
                out["stops"]["skipped_no_ratchet"] += 1
            elif ok is False:
                out["stops"]["errors"] += 1

    # errors
    for row in _iter_jsonl(ERR_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            out["errors"].append({
                "when": ts,
                "where": row.get("where") or "unknown",
                "error": row.get("error"),
            })

    return out


def summarize_run(state_dir: Path, day: date_cls) -> Dict[str, Any]:
    """Prefer daily_summary_YYYY-MM-DD.json if present; fall back to JSONL logs."""
    daily = _load_daily_summary(state_dir, day)
    if daily is not None:
        return _normalize_daily_summary(daily, state_dir, day)
    return summarize_run_jsonl(state_dir, day)


# ---- Email composer ----------------------------------------------------------
def _render_signals_block(summary: dict) -> List[str]:
    """
    Optional: compact per-ticker table (uses per-ticker fields stored in daily_summary).
    This is separate from Top signals and Trades; leave it in if you want.
    """
    tickers = (summary or {}).get("tickers", {}) or {}
    rows = []
    for t, info in tickers.items():
        if not isinstance(info, dict):
            continue
        sig = info.get("vol_adj_weight", None)
        if sig is None:
            continue
        rows.append((
            abs(float(sig)),
            t,
            float(sig),
            info.get("sleeve_risk_adj_weight", None),
            info.get("final_weight", None),
            info.get("trade_notional", 0.0),
            info.get("trade_reason", None),
        ))

    rows.sort(reverse=True, key=lambda x: x[0])
    top = rows[:TOP_SIGNALS_N]

    lines: List[str] = []
    lines.append("")
    lines.append(f"Signals (top {TOP_SIGNALS_N} by |signal|)")
    lines.append("ticker:  signal  rb_weight  final_wt  trade_notional  reason")

    any_nonzero = any(r[0] > 0 for r in top)
    if not top or not any_nonzero:
        lines.append("All signals are 0 today (or signals missing).")
        return lines

    for _, t, sig, rbw, fw, tn, reason in top:
        lines.append(
            f"  {t}: { _fmt_float(sig, 4) }  rb={_fmt_float(rbw, 4)}  w={_fmt_float(fw, 4)}  "
            f"trade={_fmt_money(tn)}  reason={reason}"
        )
    return lines


def _render_text(summary: Dict[str, Any]) -> str:
    date_str = summary.get("date") or ""
    lines: List[str] = []

    lines.append(f"Trend Following Daily Summary — {date_str}")
    lines.append("=" * 62)
    lines.append("")
    lines.append(f"Run ID:     {summary.get('run_id')}")
    lines.append(f"Started:    {summary.get('started_at')}")
    lines.append(f"Completed:  {summary.get('completed_at')}")
    lines.append(f"Dry run:    {summary.get('dry_run')}")
    lines.append("")

    pf = summary.get("portfolio") or {}
    lines.append("Portfolio")
    lines.append(f"Value:           {_fmt_money(pf.get('total_portfolio_value'))}")
    lines.append(f"Cash:            {_fmt_money(pf.get('available_cash'))}")
    lines.append(f"Volatility:      {_fmt_float(pf.get('daily_portfolio_volatility'), 4)}")
    lines.append(f"Scaling Factor:  {_fmt_float(pf.get('final_scaling_factor'), 4)}")
    lines.append(f"Target Notional: {_fmt_money(pf.get('total_target_notional'))}")
    lines.append(f"Actual Notional: {_fmt_money(pf.get('total_actual_position_notional'))}")
    lines.append(f"N pos:           {pf.get('count_of_positions')}")
    lines.append("")

    dt = summary.get("desired_totals") or {"buys": 0, "sells": 0, "zeros": 0}
    lines.append(f"Desired trades — buys: {dt.get('buys', 0)}, sells: {dt.get('sells', 0)}, zeros: {dt.get('zeros', 0)}")
    lines.append(f"Est fees (sum): {_fmt_money(summary.get('fee_est_total', 0.0))}")

    rb = summary.get("rebalance") or {}
    du = summary.get("dust") or {}
    lines.append(f"Rebalance  — built: {rb.get('built', 0)}")
    lines.append(f"Dust       — built: {du.get('built', 0)}")

    st = summary.get("stops") or {"replaced": 0, "skipped_no_position": 0, "skipped_no_ratchet": 0, "errors": 0}
    lines.append(
        f"Stops      — replaced: {st.get('replaced', 0)}, skipped(no_pos): {st.get('skipped_no_position', 0)}, "
        f"skipped(no_ratchet): {st.get('skipped_no_ratchet', 0)}, errors: {st.get('errors', 0)}"
    )
    lines.append("")

    sleeves = summary.get("sleeves") or []
    if isinstance(sleeves, list) and sleeves:
        lines.append("Sleeves (budget vs allocation)")
        for s in sleeves:
            lines.append(
                f"  {s.get('sleeve')}: budget={s.get('budget_weight')}, "
                f"alloc={s.get('allocation_share')}, pre_scale={s.get('pre_scale_weight_sum')}, "
                f"avg_mult={s.get('avg_multiplier')}"
            )
        lines.append("")

    # Top signals (this is what you wanted)
    top_signals = summary.get("top_signals") or []
    if isinstance(top_signals, list) and top_signals:
        lines.append(f"Top signals (abs, n={len(top_signals)})")
        for s in top_signals:
            tw = s.get("target_weight")
            tw_txt = f"{tw:.4f}" if isinstance(tw, (int, float)) else str(tw)
            sig = s.get("signal")
            sig_txt = f"{sig:.4f}" if isinstance(sig, (int, float)) else str(sig)
            lines.append(
                f"  {s.get('ticker')}: signal={sig_txt}, target_w={tw_txt}, sleeve={s.get('sleeve')}"
            )
        lines.append("")

    # Trades
    top_trades = summary.get("top_trades") or []
    if isinstance(top_trades, list) and top_trades:
        lines.append(f"Trades (n={len(top_trades)})")
        for t in top_trades:
            sig = t.get("signal")
            sig_txt = f"{sig:.4f}" if isinstance(sig, (int, float)) else str(sig)
            tw = t.get("target_weight")
            tw_txt = f"{tw:.4f}" if isinstance(tw, (int, float)) else str(tw)
            lines.append(
                f"  {t.get('ticker')}: trade={_fmt_money(t.get('trade_notional', 0.0))}, "
                f"fees~{_fmt_money(t.get('fees_est', 0.0))}, signal={sig_txt}, "
                f"target_w={tw_txt}, sleeve={t.get('sleeve')}, reason={t.get('reason')}"
            )
        lines.append("")

    # Optional compact table (comment this out if you find it too much)
    if TOP_SIGNALS_N > 0 and isinstance(summary.get("tickers"), dict) and summary.get("tickers"):
        lines.extend(_render_signals_block(summary))
        lines.append("")

    errs = summary.get("errors") or []
    lines.append(f"Errors: {len(errs)}")
    for e in errs[:10]:
        lines.append(f"  - [{e.get('when')}] {e.get('where')}: {e.get('error')}")
    if len(errs) > 10:
        lines.append(f"  ... and {len(errs) - 10} more")

    lines.append("")
    lines.append("State dir:")
    lines.append(summary.get("state_dir") or "")

    return "\n".join(lines)


# ---- Public API --------------------------------------------------------------
def send_summary_email(state_dir: Path, day: date_cls, subject_prefix: str = "Trend Following Expanded Universe") -> Tuple[bool, str]:
    if not EMAIL_ENABLED:
        return False, "Email disabled (TF_EMAIL_ENABLED=0)"

    if not SMTP_USER or not SMTP_PASS:
        return False, "Missing SMTP credentials (TF_SMTP_USER / TF_SMTP_PASS)."

    to_list = []
    if EMAIL_TO:
        to_list = [x.strip() for x in EMAIL_TO.split(",") if x.strip()]
    if not to_list:
        return False, "Missing TF_EMAIL_TO."

    summary = summarize_run(state_dir, day)
    body = _render_text(summary)

    ok = (len(summary.get("errors") or []) == 0)
    subj = f"{subject_prefix} — {day.isoformat()} — {summary.get('run_id')} — {'ok' if ok else 'errors'}"

    msg = EmailMessage()
    msg["Subject"] = subj
    msg["From"] = EMAIL_FROM or SMTP_USER
    msg["To"] = ", ".join(to_list)
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, f"Email failed: {type(e).__name__}: {e}"


# if __name__ == "__main__":
#     # Convenience for manual testing:
#     # python trend_following_email_summary_v020.py /path/to/state 2025-12-18
#     import sys
#     if len(sys.argv) < 3:
#         raise SystemExit("Usage: python trend_following_email_summary_v020.py <STATE_DIR> <YYYY-MM-DD>")
#     sd = Path(sys.argv[1])
#     day = date_cls.fromisoformat(sys.argv[2])
#     ok, msg = send_summary_email(sd, day)
#     print(ok, msg)
