# vLLM IFEval quality evaluation (Base vs SFT)

This is the second-stage evaluation for CIS 6900, not a serving throughput benchmark.

## Environment (separate from training)

Run with Python 3.12 in a separate virtual environment so vLLM's bundled
PyTorch/CUDA deps do not replace the working SFT training environment.

```bash
python -m pip install -U uv
uv venv /root/vllm-venv --python 3.12
uv pip install --python /root/vllm-venv/bin/python --torch-backend=auto   "vllm" "lm_eval[vllm,ifeval]" "pyyaml"
source /root/vllm-venv/bin/activate
cd /root/llm-posttraining
export HF_HOME=/root/hf_cache
export HF_HUB_CACHE=/root/hf_cache/hub
```

## Validate adapter

```bash
ls -lh outputs/sft_full/adapter/{adapter_config.json,adapter_model.safetensors}
python -m pytest -q tests/test_evaluate_sft_vllm.py
```

## Smoke and full runs

```bash
python -m scripts.evaluate_sft_vllm --dry-run --conditions base:chat,sft:chat
python -m scripts.evaluate_sft_vllm --limit 10 --conditions base:chat,sft:chat
python -m scripts.evaluate_sft_vllm --conditions base:chat,sft:chat --resume
python -m scripts.evaluate_sft_vllm --resume
```

Full protocol eventually runs four conditions: base:raw, base:chat, sft:raw, sft:chat.
The chat conditions are priority while GPU is paid; do not call two conditions
under different prompt formats a controlled SFT comparison.

The harness task YAML determines output generation length; **do not shorten
its output cap for reported scores**. `--limit` is for pipeline validation only.
We choose max_model_len=4096, which is an ENGINE context capacity and is not
itself the number of output tokens. Check harness task prompts fit the context.

## Storage & reports

- Small summary: `results/eval_sft_vllm/summary.json` (commit to Git).
- Raw logs: `results/raw/harness/eval_sft_vllm/` (ignored by Git).
- Resume uses completed raw results; if interrupted mid-condition, rerun that condition.
- Archive raw data / configuration and trained adapter externally before destroying a pod.
- Commit `pip freeze` of this separate vLLM environment as `environment.vllm.freeze.txt`.
- Report task version, generation limits, model revision, vLLM+lm-eval versions, decoding settings, GPU.

IFEval uses instruction-following checks. It does not report TTFT/TPOT,
throughput, or peak memory; those require separate serving benchmarks.
