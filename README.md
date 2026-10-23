# PPL synthesis reward hacking

[![CI](https://github.com/SynthStats/ppl-synthesis-reward-hacking/actions/workflows/ci.yml/badge.svg)](https://github.com/SynthStats/ppl-synthesis-reward-hacking/actions/workflows/ci.yml)
  <img src="https://img.shields.io/badge/python-3.11-3670A0?logo=python&logoColor=ffdd54" alt="Python 3.11">
  <img src="https://img.shields.io/badge/pixi-package%20manager-yellow?logo=anaconda" alt="Pixi">

We asked Qwen3 Coder 30B for a Bayesian model of binary outcomes in a probabilistic programming language (PPL), and one of the PyMC programs it wrote observes the same data twice, under two different parameterizations: the log-likelihood it reports is 15.0 above that of the intended observation alone. That second observation gives the program a data density that does not integrate to one over the data space. We call such programs exploits, and writing them likelihood hacking (LH). The scripts below train language models with a likelihood-based reward to write PyMC and Stan programs, and measure how often the trained models write exploits.

## Setup

```bash
git clone https://github.com/SynthStats/ppl-synthesis-reward-hacking.git --recurse-submodules 
cd ppl-synthesis-reward-hacking
# Pixi is the package/environment manager
# install Python 3.11 and the dev dependencies used by the commands below
pixi install -e dev
# copy .env.example, which lists every key, and fill in the ones your run needs
cp .env.example .env
```

```bash
# .env
TINKER_API_KEY=   # Tinker is a remote model-training API (a basic run needs no other key)
WANDB_API_KEY=    # W&B hyperparameter sweeps
WANDB_ENTITY=     # W&B user or team
WANDB_PROJECT=ppl-synthesis-reward-hacking
OPENAI_API_KEY=   # the LLM judge, which can monitor training or gate its rewards (an OpenAI model by default); also train/sstan_gate=enforce
ZHIPUAI_API_KEY=  # default key for eval_sstan_exemplars.py and the standalone PyMC-to-Stan LLM transpiler
# SafeStan (SStan) checks Stan programs for unsafe likelihood constructs
# optional standalone transpiler overrides (defaults shown)
# PSRH_TRANSPILER_MODEL=anthropic/glm-5
# PSRH_TRANSPILER_API_KEY_ENV=ZHIPUAI_API_KEY
# PSRH_TRANSPILER_API_BASE=https://api.z.ai/api/anthropic
# PSRH_TRANSPILER_PROVIDER=anthropic
# train/sstan_gate=enforce selects OPENAI_API_KEY through train.sstan_transpiler_api_key_env;
# that explicit setting takes precedence over PSRH_TRANSPILER_API_KEY_ENV
ANTHROPIC_API_KEY= # optional additional provider for multi-model evaluation
OPENROUTER_API_KEY= # optional additional provider for multi-model evaluation
RUNPOD_API_KEY=   # optional; required for RunPod GPU training
```

```bash
# load the keys before any command that calls an API
set -a && source .env && set +a

# quick checks, about two minutes each and with no training or API key: both run hardcoded programs
# through the scorer, the normalization checker (does the data density integrate to one?),
# the heuristic detector (exploit-pattern matching) and the SafePyMC gate
# (check_pymc_model: a structural check on the instantiated PyMC model and its compiled graph)
pixi run -e dev pytest tests/integration/test_e2e_exploit_pipeline.py -v
# the script needs no GPU either, and runs 10 exploit families and 2 honest baselines
pixi run -e dev python scripts/smoke_test_exploit.py

# before a long run, the full test suite (a few minutes)
pixi run -e dev pytest tests/ -x
```

## Entry points

```text
entry point                              keys needed        notes
# Hydra composes training configuration from YAML groups and command-line overrides
# TRL (Transformer Reinforcement Learning) is the local training library;
# GRPO (Group Relative Policy Optimization) is the training algorithm
# cmdsafestan is the Python interface to the SafeStan compiler
scripts/smoke_test_exploit.py            none               detection pipeline on hardcoded programs, ~2 min
pixi run -e dev pytest tests/ -x         none               full test suite; recommended before long runs, optional for quickstart
scripts/eval_normalization.py            none               offline normalization check on existing runs
scripts/eval_taxonomy.py                 none               heuristic exploit taxonomy on existing runs
scripts/eval_safety_gate.py              none               SafePyMC gate evaluation
scripts/hydra_train_tinker.py            TINKER_API_KEY     Tinker API training (no GPU); add train/monitoring_mode=judge_evolving for LLM judge, train/sstan_gate=enforce for SStan gate (OPENAI_API_KEY by default), train/judge_gate=enforce for judge gate
scripts/create_wandb_sweep.py            + WANDB_API_KEY    create sweep, then run agents with wandb agent
scripts/hydra_train_trl.py               GPU (40+ GB VRAM)  local TRL GRPO training
scripts/trl_reward_hacking_stan.py       GPU + cmdsafestan  TRL with SafeStan checker
scripts/eval_sstan_exemplars.py          ZHIPUAI_API_KEY    SafeStan on exemplars; see SafeStan guide (docs/safestan-guide.md)
scripts/evaluate_safestan_mitigation.py  none               offline SafeStan labeled suite from pre-translated Stan
scripts/evaluate_transpiler_fidelity.py  none               offline transpiler fidelity audit from pre-translated Stan
```

## Base-rate exploit sweep

Do untrained base and instruct models already write such programs? We asked each model in the table below for a Bayesian coin-flip model, through AWS Bedrock in February 2026, with 20 prompts and 10 samples per prompt (so 200 completions per model). Each completion was extracted, scanned for exploit strings, run, checked by `check_pymc_model` and scored, and then underwent two integrity checks from [`evaluation/integrity_checks.py`](src/ppl_synthesis_reward_hacking/evaluation/integrity_checks.py).

| Model | Valid % | Any exploit % |
|---|---|---|
| Qwen3 Coder 480B | 99.0 | 0.0 |
| Qwen3 Coder 30B | 100.0 | 0.5 |
| Qwen3 32B | 89.0 | 0.0 |
| Llama 3 8B Instruct | 97.5 | 1.5 |
| Amazon Nova Lite | 89.0 | 0.5 |
| Gemma 3 4B | 45.5 | 1.0 |
| Mistral 7B Instruct | 6.0 | 2.0 |

All flagged programs are documented in [docs/base_rate_exploits.md](docs/base_rate_exploits.md).

### Reproducing

```bash
# Phase A: quick filter (20 completions per model)
pixi run -e dev python scripts/sweep_base_rate.py \
    --models "bedrock/converse/qwen.qwen3-32b-v1:0,bedrock/meta.llama3-8b-instruct-v1:0" \
    --filter-mode --aws-region eu-west-2 \
    --output-dir artifacts/sweep_filter

# Phase B: full sweep (200 completions per model)
pixi run -e dev python scripts/sweep_base_rate.py \
    --models "bedrock/converse/qwen.qwen3-coder-30b-a3b-v1:0" \
    --n-prompts 20 --samples-per-prompt 10 \
    --aws-region eu-west-2 --call-delay 0.5 \
    --output-dir artifacts/sweep_phase_b
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [SafeStan guide](docs/safestan-guide.md)
- [Scoring methods](docs/scoring.md)
- [Base-rate exploits](docs/base_rate_exploits.md)
- [SafeStan integration provenance](docs/provenance/safestan_integration.md)

## All scripts

```
scripts/
├── hydra_train_tinker.py       # Tinker API training (Hydra)
├── hydra_train_trl.py          # TRL training (Hydra)
├── trl_reward_hacking.py       # TRL training (standalone)
├── trl_reward_hacking_stan.py  # TRL with SafeStan checker
├── tinker_pymc_grpo.py         # Tinker training (standalone, no Hydra)
├── eval_normalization.py       # offline normalization check
├── eval_safety_gate.py         # SafePyMC gate
├── eval_taxonomy.py            # heuristic exploit taxonomy
├── eval_sstan_exemplars.py     # SafeStan exemplar evaluation
├── evaluate_safestan_mitigation.py
├── evaluate_transpiler_fidelity.py
├── smoke_test_exploit.py       # detection pipeline smoke test
├── create_wandb_sweep.py       # W&B sweep creation
├── sweep_base_rate.py          # base-rate exploit sweep
├── analyze_hacking.py          # post-hoc hacking analysis
├── report_run_progress.py      # check active run status
├── generate_paper_figures.py
├── prepare_local_figures.py
└── make_paper_artifacts.py
```
