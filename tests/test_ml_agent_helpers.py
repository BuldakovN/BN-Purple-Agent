"""Unit tests for ml_agent helpers (validation report, output truncation)."""

from __future__ import annotations

import pandas as pd

from ml_helpers import parse_latest_cv_score, truncate_output, validate_submission_report


def test_parse_latest_cv_score_last_wins():
    text = "CV_SCORE=0.5\nmore\nCV_SCORE=0.82"
    assert parse_latest_cv_score(text) == 0.82


def test_parse_latest_cv_score_none():
    assert parse_latest_cv_score("no score here") is None


def test_truncate_output_success_keeps_tail():
    text = "x" * 100
    out = truncate_output(text, max_chars=50, is_error=False)
    assert "truncated" in out
    assert out.endswith("x" * 20) or "x" in out[-30:]


def test_truncate_output_error_keeps_head_and_tail():
    head = "Traceback (most recent call last):\n  File \"x.py\""
    mid = "\n" + "m" * 8000
    tail = "\nValueError: bad value\n"
    text = head + mid + tail
    out = truncate_output(text, max_chars=1200, is_error=True)
    assert "Traceback" in out
    assert "ValueError" in out
    assert "middle truncated" in out


def test_validate_submission_report_missing_file(tmp_path):
    (tmp_path / "home" / "data").mkdir(parents=True)
    r = validate_submission_report(tmp_path)
    assert "FAIL" in r
    assert "does not exist" in r


def test_validate_submission_report_ok(tmp_path):
    data = tmp_path / "home" / "data"
    data.mkdir(parents=True)
    test_df = pd.DataFrame({"id": [1, 2], "feat": [0.1, 0.2]})
    sample = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    test_df.to_csv(data / "test.csv", index=False)
    sample.to_csv(data / "sample_submission.csv", index=False)
    sub = pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]})
    sub.to_csv(tmp_path / "submission.csv", index=False)

    r = validate_submission_report(tmp_path)
    assert "OK — column names and order" in r
    assert "OK — row count matches test.csv" in r
    assert "OK — no NA values" in r


def test_validate_submission_report_row_mismatch(tmp_path):
    data = tmp_path / "home" / "data"
    data.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2]}).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": [1, 2], "y": [0, 0]}).to_csv(
        data / "sample_submission.csv", index=False
    )
    pd.DataFrame({"id": [1], "y": [0.0]}).to_csv(tmp_path / "submission.csv", index=False)

    r = validate_submission_report(tmp_path)
    assert "FAIL — row count" in r
