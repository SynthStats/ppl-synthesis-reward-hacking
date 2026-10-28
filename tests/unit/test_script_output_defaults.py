"""Output destinations selected by the command-line parsers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("script", "option", "expected"),
    [
        ("eval_sstan_exemplars.py", "output_dir", "artifacts/sstan_exemplars"),
        ("make_grpo_figures.py", "out", str(ROOT / "artifacts" / "figures")),
    ],
)
def test_output_defaults(script, option, expected):
    capture_arguments = """
import argparse
import json
import runpy
import sys

script = sys.argv[1]
sys.argv = [script]
if script.endswith("eval_sstan_exemplars.py"):
    sys.argv += ["--lh-examples", "examples/sstan_exemplars.md"]
parse = argparse.ArgumentParser.parse_args

def capture(self, *args, **kwargs):
    args = parse(self, *args, **kwargs)
    print(json.dumps({key: str(value) for key, value in vars(args).items()}))
    raise SystemExit(0)

argparse.ArgumentParser.parse_args = capture
runpy.run_path(script, run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", capture_arguments, str(ROOT / "scripts" / script)],
        cwd=ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)[option] == expected
