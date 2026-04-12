"""Pure helpers for ML agent: submission checks and tool output shaping (no A2A/LangGraph)."""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

ERROR_HEAD_CHARS = 2200

_WARNING_BLOCK_RE = re.compile(
    r"^/.+?:\d+:.*?Warning:.*?$\n(?:^\s+.*?$\n)*",
    re.MULTILINE,
)

_CV_SCORE_RE = re.compile(
    r"CV_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def strip_warnings(text: str) -> str:
    return _WARNING_BLOCK_RE.sub("", text).strip()


def parse_latest_cv_score(text: str) -> float | None:
    """Parse the last CV_SCORE=<float> from tool stdout (same convention as AIDE-style agents)."""
    matches = _CV_SCORE_RE.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def validate_submission_report(workdir: str | Path) -> str:
    """Structured checks for ./submission.csv vs sample submission and test.csv."""
    root = Path(workdir).resolve()
    sub_path = root / "submission.csv"
    data_dir = root / "home" / "data"
    lines: list[str] = []

    if not sub_path.is_file():
        return (
            "validate_submission: FAIL — ./submission.csv does not exist. "
            "Create it with run_python at ./submission.csv."
        )

    try:
        sub = pd.read_csv(sub_path)
    except Exception as exc:
        return f"validate_submission: FAIL — cannot read submission.csv: {exc}"

    lines.append(f"submission.csv: rows={len(sub)}, columns={list(sub.columns)}")

    sample_paths = sorted(glob.glob(str(data_dir / "sample_submission*.csv")))
    if sample_paths:
        try:
            sample = pd.read_csv(sample_paths[0])
        except Exception as exc:
            lines.append(f"WARN — could not read sample submission: {exc}")
        else:
            exp_cols = list(sample.columns)
            act_cols = list(sub.columns)
            if exp_cols == act_cols:
                lines.append("OK — column names and order match sample_submission.")
            else:
                lines.append(
                    f"FAIL — column order/names differ. Expected {exp_cols}, got {act_cols}."
                )
            missing = [c for c in exp_cols if c not in sub.columns]
            extra = [c for c in act_cols if c not in exp_cols]
            if missing:
                lines.append(f"FAIL — missing columns vs sample: {missing}")
            if extra:
                lines.append(f"WARN — extra columns vs sample: {extra}")
    else:
        lines.append("WARN — no sample_submission*.csv; skipped schema check.")

    test_paths = sorted(glob.glob(str(data_dir / "test.csv")))
    if test_paths:
        try:
            test_df = pd.read_csv(test_paths[0])
        except Exception as exc:
            lines.append(f"WARN — could not read test.csv: {exc}")
        else:
            if len(sub) != len(test_df):
                lines.append(
                    f"FAIL — row count {len(sub)} != test.csv rows {len(test_df)}."
                )
            else:
                lines.append(f"OK — row count matches test.csv ({len(test_df)}).")

    na_cols = sub.columns[sub.isna().any()].tolist()
    if na_cols:
        lines.append(f"FAIL — NA values in columns: {na_cols}")
    else:
        lines.append("OK — no NA values in submission.")

    return "\n".join(lines)


def truncate_output(text: str, max_chars: int, *, is_error: bool = False) -> str:
    if len(text) <= max_chars:
        return text
    if not is_error:
        keep = max_chars - 80
        truncated_chars = len(text) - keep
        return f"[...{truncated_chars} chars truncated...]\n{text[-keep:]}"

    sep = "\n...[middle truncated]...\n"
    overhead = len(sep) + 60
    usable = max(400, max_chars - overhead)
    head_n = min(ERROR_HEAD_CHARS, usable // 2)
    tail_n = usable - head_n
    if head_n + tail_n >= len(text):
        return text[:max_chars] + "\n[truncated]\n"
    omitted = len(text) - head_n - tail_n
    return (
        f"[error output: {omitted} chars omitted in middle; "
        f"showing first {head_n} + last {tail_n}]\n"
        f"{text[:head_n]}{sep}{text[-tail_n:]}"
    )
