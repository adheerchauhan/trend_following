import os, json
from datetime import datetime, timezone, date as date_cls
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple
import smtplib
from email.message import EmailMessage


# ---- Config (env vars) -------------------------------------------------------
SMTP_HOST     = os.getenv("TF_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("TF_SMTP_PORT", "587"))            # STARTTLS
SMTP_USER     = os.getenv("TF_SMTP_USER")                        # your email
SMTP_PASS     = os.getenv("TF_SMTP_PASS")                        # app password
EMAIL_FROM    = os.getenv("TF_EMAIL_FROM", SMTP_USER or "")
EMAIL_TO      = os.getenv("TF_EMAIL_TO")                         # comma-separated for multiple
EMAIL_ENABLED = bool(int(os.getenv("TF_EMAIL_ENABLED", "1")))    # 1/0 toggle


# ---- Helpers -----------------------------------------------------------------
def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed
                continue


def _is_for_day(ts_iso: str, day: date_cls) -> bool:
    # Day boundary in UTC
    try:
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return ts.date() == day
    except Exception:
        return False


def _fmt_money(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)


# ---- Summarizer --------------------------------------------------------------
def summarize_run(state_dir: Path, day: date_cls) -> Dict[str, Any]:
    """
    Reads your existing JSONL logs and builds a compact summary for `day` (UTC).
    """
    p = state_dir
    # file names as per your setup
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
        "started_at": None,
        "completed_at": None,
        "dry_run": None,
        "desired_totals": {"buys": 0, "sells": 0, "zeros": 0},
        "rebalance": {"built": 0, "submitted": 0, "preview": None},
        "dust": {"built": 0, "submitted": 0, "preview": None},
        "stops": {"replaced": 0, "skipped_no_position": 0, "skipped_no_ratchet": 0, "errors": 0},
        "errors": [],
    }

    # heartbeat: start/done + dry_run flag
    for row in _iter_jsonl(HEARTBEAT_LOG):
        ts = row.get("ts")
        if not ts or not _is_for_day(ts, day):
            continue
        ev = row.get("event")
        if ev == "config_loaded":
            out["started_at"] = ts
            if "dry_run" in row:
                out["dry_run"] = bool(row["dry_run"])
        elif ev == "run_complete":
            out["completed_at"] = ts

    # desired positions
    # we also pick up totals from heartbeat event if present
    buys = sells = zeros = 0
    for row in _iter_jsonl(DESIRED_LOG):
        ts = row.get("ts")
        if not ts or not _is_for_day(ts, day):
            continue
        nt = float(row.get("new_trade_notional", 0.0))
        if nt > 0:
            buys += 1
        elif nt < 0:
            sells += 1
        else:
            zeros += 1
    out["desired_totals"] = {"buys": buys, "sells": sells, "zeros": zeros}

    # rebalance orders
    for row in _iter_jsonl(ORDER_BUILD_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            out["rebalance"]["built"] += 1
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
    for row in _iter_jsonl(DUST_SUBMIT_LOG):
        ts = row.get("ts")
        if ts and _is_for_day(ts, day):
            out["dust"]["submitted"] += int(row.get("orders_count") or 0)
            if out["dust"]["preview"] is None and "preview" in row:
                out["dust"]["preview"] = bool(row["preview"])

    # stops
    for row in _iter_jsonl(STOP_UPDATE_LOG):
        ts = row.get("ts")
        if not ts or not _is_for_day(ts, day):
            continue
        action = (row.get("action") or "").lower()
        ok = row.get("ok")
        if action == "replaced" and ok:
            out["stops"]["replaced"] += 1
        elif action == "no_position":
            out["stops"]["skipped_no_position"] += 1
        elif action == "skip" and (row.get("reason") == "no_ratchet"):
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


# ---- Email composer ----------------------------------------------------------
def _render_text(summary: Dict[str, Any]) -> str:

    lines = []
    lines.append(f"Trend Following Daily Summary — {summary['date']}")
    lines.append("=" * 48)
    lines.append(f"Started:   {summary.get('started_at')}")
    lines.append(f"Completed: {summary.get('completed_at')}")
    if summary.get("dry_run") is not None:
        lines.append(f"Dry run:   {summary['dry_run']}")
    lines.append("")
    dt = summary["desired_totals"]
    lines.append(f"Desired trades — buys: {dt['buys']}, sells: {dt['sells']}, zeros: {dt['zeros']}")
    rb = summary["rebalance"]
    lines.append(f"Rebalance — built: {rb['built']}, submitted: {rb['submitted']}, preview: {rb['preview']}")
    du = summary["dust"]
    lines.append(f"Dust      — built: {du['built']}, submitted: {du['submitted']}, preview: {du['preview']}")
    st = summary["stops"]
    lines.append(f"Stops     — replaced: {st['replaced']}, skipped(no_pos): {st['skipped_no_position']}, "
                 f"skipped(no_ratchet): {st['skipped_no_ratchet']}, errors: {st['errors']}")
    err_count = len(summary["errors"])
    lines.append("")
    lines.append(f"Errors: {err_count}")
    for e in summary["errors"][:10]:
        lines.append(f"  - [{e['when']}] {e['where']}: {e['error']}")
    if err_count > 10:
        lines.append(f"  ... and {err_count - 10} more")
    lines.append("")
    lines.append("Log folder:")
    lines.append(str(Path(os.getenv("STATE_DIR_PATH_OVERRIDE") or "")))  # optional pointer

    return "\n".join(lines)


def send_summary_email(state_dir: Path, day: date_cls, *, subject_prefix: str = "Trend Following Strategy Summary") -> Tuple[bool, str]:

    if not EMAIL_ENABLED:
        return False, "EMAIL_ENABLED=0"

    if not (SMTP_USER and SMTP_PASS and EMAIL_TO):
        return False, "Missing SMTP env vars: TF_SMTP_USER/TF_SMTP_PASS/TF_EMAIL_TO"

    summary = summarize_run(state_dir, day)
    text = _render_text(summary)
    subject = f"{subject_prefix} — {day} — ok"
    if summary["errors"]:
        subject = f"{subject_prefix} — {day} — ERRORS({len(summary['errors'])})"

    to_list = [x.strip() for x in EMAIL_TO.split(",") if x.strip()]
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM or SMTP_USER
    msg["To"] = ", ".join(to_list)
    msg["Subject"] = subject
    msg.set_content(text)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
        s.ehlo()
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

    return True, "sent"
