"""Command-line input selection for exemplar evaluation."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "eval_sstan_exemplars.py"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )


def test_missing_catalogue_is_a_usage_error():
    result = run_cli("--dry-run")
    assert result.returncode == 2
    assert "--lh-examples" in result.stderr


def test_supplied_catalogue_is_evaluated(tmp_path):
    catalogue = tmp_path / "examples.md"
    catalogue.write_text(
        "### P7: Caller-selected example\n\n```python\ndef model(data):\n    return data\n```\n"
    )
    result = run_cli("--lh-examples", str(catalogue), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "P7: Caller-selected example" in result.stdout
    assert "total: 6 programs" in result.stdout
