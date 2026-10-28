"""Output location for the toy-model training command."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_default_output_is_checkout_relative():
    code = """
import json, runpy, sys
script = sys.argv[1]
sys.argv = [script]
module = runpy.run_path(script)
print(json.dumps(vars(module["parse_args"]())))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "scripts" / "trl_toy_training.py")],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout.splitlines()[-1])["output_dir"] == "artifacts/grpo_toy"
