"""Exercise the command a reader runs, including usable failure messages."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from fraud_detection.data import ROOT


def run_cli(directory, *args):
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "fraud_detection", *args],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_writes_report_with_dataset_and_split_metadata(transactions, tmp_path):
    data = tmp_path / "transactions.csv"
    output = tmp_path / "results" / "metrics.json"
    transactions.to_csv(data, index=False)
    result = run_cli(tmp_path, "--data", str(data), "--output", str(output))
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["dataset"]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert report["configuration"]["sample_rows"] == 300
    assert report["split"]["test_class_counts"] == {"0": 60, "1": 15}
    assert "xgboost" in report["results"]
    assert "scikit-learn" in report["versions"]
    assert "Training rows: 225; holdout rows: 75" in result.stdout


def test_default_dataset_path_works_outside_repository(tmp_path):
    result = run_cli(tmp_path, "--sample-size", "100", "--models", "baseline")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "artifacts" / "metrics.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [("--data", "missing.csv"), ("--models", "invalid"), ("--sample-size", "-1")],
)
def test_cli_errors_are_actionable_without_tracebacks(tmp_path, arguments):
    result = run_cli(tmp_path, *arguments)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "artifacts" / "metrics.json").exists()
