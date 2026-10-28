"""Tests for RunPod launcher mode/config helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "runpod" / "launch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("runpod_launch_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "TINKER_API_KEY",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "CMDSAFESTAN_PIP_SPEC",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_parse_args_defaults(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(sys, "argv", ["launch.py"])
    args, extra = mod.parse_args()
    assert args.mode == "trl"
    assert args.api_port == 8000
    assert args.backend == "fsdp"
    assert args.env_mode == "minimal"
    assert args.preflight_only is False
    assert extra == []


def test_parse_args_tinker_mode_with_extra(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "--mode",
            "tinker_api",
            "--api-port",
            "9001",
            "--base-model",
            "Qwen/Qwen3-0.6B",
            "--",
            "--database-url",
            "sqlite:///tmp/tinker.db",
        ],
    )
    args, extra = mod.parse_args()
    assert args.mode == "tinker_api"
    assert args.api_port == 9001
    assert args.base_model == "Qwen/Qwen3-0.6B"
    assert "--database-url" in extra
    assert "sqlite:///tmp/tinker.db" in extra


def test_pod_ports_for_mode():
    mod = _load_module()
    assert mod.pod_ports_for_mode("trl", 8000) == "22/tcp"
    assert mod.pod_ports_for_mode("trl_stan", 8000) == "22/tcp"
    assert mod.pod_ports_for_mode("tinker_api", 8123) == "22/tcp,8123/http"


def test_build_pod_env_tinker_requires_real_key(monkeypatch):
    mod = _load_module()
    _clear_launcher_env(monkeypatch)

    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        mod.build_pod_env("tinker_api")

    monkeypatch.setenv("TINKER_API_KEY", "tml-dummy")
    with pytest.raises(RuntimeError, match="real TINKER_API_KEY"):
        mod.build_pod_env("tinker_api")

    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("TINKER_API_KEY", "tml-real-key")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    out = mod.build_pod_env("tinker_api")
    assert out["TINKER_API_KEY"] == "tml-real-key"
    assert out["WANDB_API_KEY"] == "wandb-secret"
    assert out["WANDB_PROJECT"] == "proj"


def test_build_pod_env_extended_forwards_prefixed_keys(monkeypatch):
    mod = _load_module()
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CMDSAFESTAN_PIP_SPEC", "git+https://example.com/cmdsafestan.git")
    out = mod.build_pod_env("trl_stan", env_mode="extended")
    assert out["OPENAI_API_KEY"] == "sk-test"
    assert out["CMDSAFESTAN_PIP_SPEC"] == "git+https://example.com/cmdsafestan.git"


def test_build_remote_start_command_mode_dispatch():
    mod = _load_module()
    trl_cmd = mod.build_remote_start_command(
        "trl",
        extra_args=["--n-steps", "10"],
        backend="fsdp",
        base_model="Qwen/Qwen3-0.6B",
        api_port=8000,
        skyrl_ref="skyrl_train-v0.4.0",
    )
    assert "scripts/runpod/run_grpo_pymc_reward.sh" in trl_cmd
    assert "--n-steps" in trl_cmd
    assert " && " in trl_cmd
    assert "'&&'" not in trl_cmd

    stan_cmd = mod.build_remote_start_command(
        "trl_stan",
        extra_args=["--n-steps", "5"],
        backend="fsdp",
        base_model="Qwen/Qwen3-0.6B",
        api_port=8000,
        skyrl_ref="skyrl_train-v0.4.0",
    )
    assert "scripts/runpod/run_grpo_stan_reward.sh" in stan_cmd
    assert "--n-steps" in stan_cmd
    assert " && " in stan_cmd
    assert "'&&'" not in stan_cmd

    tinker_cmd = mod.build_remote_start_command(
        "tinker_api",
        extra_args=["--database-url", "sqlite:///tmp/x.db"],
        backend="fsdp",
        base_model="Qwen/Qwen3-0.6B",
        api_port=9000,
        skyrl_ref="skyrl_train-v0.4.0",
    )
    assert "scripts/runpod/run_tinker_api_server.sh" in tinker_cmd
    assert "skyrl_train-v0.4.0" in tinker_cmd
    assert "Qwen/Qwen3-0.6B" in tinker_cmd
    assert " && " in tinker_cmd
    assert "'&&'" not in tinker_cmd


def test_run_tinker_smoke_calls_script(monkeypatch):
    mod = _load_module()
    calls: list[list[str]] = []

    def _fake_run(cmd, check):  # noqa: ANN001
        assert check is True
        calls.append(cmd)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    mod.run_tinker_smoke("1.2.3.4", 8000, 120)

    assert calls
    cmd = calls[0]
    assert cmd[0] == "bash"
    assert cmd[-3:] == ["1.2.3.4", "8000", "120"]
    assert cmd[1].endswith("scripts/runpod/smoke_tinker_api.sh")


def test_rsync_uploads_only_tracked_files(tmp_path, monkeypatch):
    mod = _load_module()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "models").mkdir()
    (source / "models" / "model with spaces.py").write_text("model = 1\n")
    (source / "line\nbreak.txt").write_text("tracked\n")
    (source / ".gitignore").write_text("ignored-output/\n")
    subprocess.run(
        ["git", "-C", str(source), "add", ".gitignore", "models", "line\nbreak.txt"],
        check=True,
    )
    (source / "models" / "untracked.py").write_text("local = 1\n")
    (source / "local-output.txt").write_text("local\n")
    (source / "ignored-output").mkdir()
    (source / "ignored-output" / "result.txt").write_text("local\n")
    (source / ".local-cache").mkdir()
    (source / ".local-cache" / "result.txt").write_text("local\n")
    monkeypatch.setattr(mod, "REPO_ROOT", str(source))
    run = subprocess.run

    def transport(cmd, **kwargs):
        if cmd[0] == "ssh":
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[0] == "rsync":
            cmd = [*cmd[:-1], str(destination) + "/"]
        return run(cmd, **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", transport)
    mod.rsync_to_pod("127.0.0.1", 22)

    transferred = {
        p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()
    }
    assert transferred == {".gitignore", "models/model with spaces.py", "line\nbreak.txt"}
    assert (destination / "models" / "model with spaces.py").read_text() == "model = 1\n"
