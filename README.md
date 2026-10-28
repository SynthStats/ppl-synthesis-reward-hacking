# ppl-synthesis-reward-hacking

These experiments measure likelihood hacking in PyMC and Stan program synthesis.

## Install

```bash
git clone --recurse-submodules https://github.com/SynthStats/ppl-synthesis-reward-hacking.git
cd ppl-synthesis-reward-hacking
pixi install -e dev
```

## Example

```bash
pixi run -e dev psrh data generate --config configs/data/bernoulli_1d.yaml --seed 0
```

## Resources

| Resource | Contents |
| --- | --- |
| [Paper](https://proceedings.mlr.press/v337/karwowski26a.html) | Likelihood hacking in probabilistic program synthesis, UAI 2026 |
| [Attribution](docs/provenance/safestan_integration.md) | Upstream contributions |
| [Scripts](scripts/) | Tinker and TRL training; normalization, taxonomy and gate evaluation; sweeps; figures |
| [Key template](.env.example) | Copy to `.env`; load with `set -a && source .env && set +a` |
| Tinker training | `TINKER_API_KEY`; judge monitoring and gates use `OPENAI_API_KEY` by default |
| Standalone PyMC-to-Stan translation | `ZHIPUAI_API_KEY` by default |
| [Weights & Biases](https://wandb.ai/) sweeps | `WANDB_API_KEY` |
| [Flagging](src/ppl_synthesis_reward_hacking/evaluation/integrity_checks.py) | Exploit-string heuristics, `check_pymc_model` and integrity checks |
| [Base-rate sweep](scripts/sweep_base_rate.py) | Reproduction commands below |
| Tools | [Python](https://www.python.org/downloads/) 3.11+, [PyMC](https://www.pymc.io/), [Stan](https://mc-stan.org/), [Pixi](https://pixi.sh/), [pytest](https://docs.pytest.org/) |
| [Tests](https://github.com/SynthStats/ppl-synthesis-reward-hacking/actions/workflows/ci.yml) | `pixi run -e dev pytest tests/ -x` |

## Reproduce

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

## Base-rate results

AWS Bedrock, February 2026: 200 completions per model (20 prompts × 10 samples); valid means executes and scores. The last column counts completions that any check of the heuristic exploit classifier flagged, over all 200.

| Model | Valid | Flagged by any exploit check |
| --- | --- | --- |
| Qwen3 Coder 480B | 99.0% | 0.0% |
| Qwen3 Coder 30B | 100.0% | 0.5% |
| Qwen3 32B | 89.0% | 0.0% |
| Llama 3 8B Instruct | 97.5% | 1.5% |
| Amazon Nova Lite | 89.0% | 0.5% |
| Gemma 3 4B | 45.5% | 1.0% |
| Mistral 7B Instruct | 6.0% | 2.0% |

[SynthStats](https://synthstats.org)
