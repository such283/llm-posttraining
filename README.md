# CIS 6900 — LLM Post-Training + Inference Optimization

Research project for Fall 2026. Advisor: Dr. Guangjing Wang.

> **Status: engineering scaffold / first experiments, not completed research results.**
> No GPU training or scientific evaluation was performed while preparing this repository.
> All scripts must be smoke-tested on the rental GPU before running full experiments.

## Research questions

1. **RQ1 — Post-training trade-off:** How do SFT and DPO affect instruction following (IFEval) and potential alignment tax (MMLU, ARC-Challenge, WikiText-2 perplexity)?
2. **RQ2 — Quantization:** Compare GPTQ INT4 and AWQ INT4 with generic and chat-formatted calibration data.
3. **RQ3 — Efficiency crossover:** Compare weight-only INT4 to FP16 under vLLM across concurrent request levels 1, 4, 16, 64; measure TTFT, TPOT, throughput, and GPU memory.

## Current implementation scope

- **Implemented as first-stage scripts:** base-model generation smoke test under both raw and chat-formatted prompts; deterministic SFT and DPO dataset transformation; SFT QLoRA training; SFT merge; DPO QLoRA training; metadata capture; sample dataset unit tests.
- **Not yet implemented:** formal IFEval/MMLU/ARC/perplexity evaluation; GPTQ/AWQ calibration and quantization; vLLM server benchmark; Triton kernel; report figures.
- **Not yet validated:** GPU execution of the training code on the selected A100 instance. Treat these as a starting implementation that may require dependency-specific changes after the first actual smoke test.

## Requirements and storage

Target GPU: single NVIDIA A100 PCIe 40 GB, BF16 capable. QLoRA stores the frozen pretrained model in 4-bit NF4 and trains LoRA adapters; it is **not** final INT4 GPTQ/AWQ quantization.

**Storage warning:** the 32 GB disk shown on the initial Vast.ai instance is too small for all project stages. Allocate **at least 100–150 GB** of usable storage before beginning full training and the BF16 merge, or arrange another durable artifact location. Do not assume instance-local data persists after deletion. Back up each adapter/checkpoint to a durable destination.

## Installation

```bash
# Execute from the cloned repository's root. On Vast.ai:
cd /workspace
# git clone <your-actual-GitHub-repository-URL>
# cd llm-posttraining

# Use a preinstalled CUDA-enabled PyTorch matching the server's driver, or
# install a compatible official PyTorch CUDA wheel first if missing.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip freeze > environment.freeze.txt
python -m pytest -q
python -m scripts.preflight
```

`requirements.txt` constrains API families but is not a locked environment. Commit `environment.freeze.txt` with each milestone once the installation works. Store `nvidia-smi` output separately in your lab notes. No Hugging Face token needs to be checked into Git.

## Experiment sequence

### 0. GPU and model smoke test (no training)

```bash
python -m scripts.baseline --config configs/baseline.yaml
```

Outputs in `results/baseline/`: metadata and raw/chat-formatted generations. This checks that the model loads and can answer. The wall-time and allocated-memory data here are **diagnostic only**, not TTFT/TPOT or formal throughput benchmarks.

### 1. Inspect datasets

```bash
python -m scripts.inspect_data --stage sft --n 2
python -m scripts.inspect_data --stage dpo --n 2
```

- SFT: `HuggingFaceH4/no_robots`, split `train`.
- DPO: `HuggingFaceH4/ultrafeedback_binarized`, split `train_prefs`.

The data conversion uses the **last assistant turn** as the SFT target, with earlier turns as context. This is a documented modeling decision: it avoids training on user messages, but is not the same as learning on *every* assistant turn of multi-turn conversations.

For DPO, the common prompt is extracted from the chosen/rejected examples and the two alternative assistant completions are separated. A seeded shuffle, fixed selection size, and held-out validation split make the prepared data reproducible. Invalid rows are excluded and counts are logged.

### 2. Small SFT QLoRA test

```bash
python -m scripts.train_sft --config configs/sft_smoke.yaml
```

Uses 64 examples and 5 optimizer steps, then saves the adapter under `outputs/sft_smoke/adapter`. A short run verifies wiring and loss behavior; it is **not** the research SFT checkpoint.

### 3. Full SFT (after successful smoke test)

```bash
python -m scripts.train_sft --config configs/sft_full.yaml
```

Saves the final SFT adapter to `outputs/sft_full/adapter`.

### 4. Merge SFT into BF16

```bash
python -m scripts.merge_sft \
  --adapter outputs/sft_full/adapter \
  --output outputs/sft_merged
```

This creates an unquantized SFT initialization. Preserve this as the DPO reference policy. Be sure enough disk space exists before running this step.

### 5. Small DPO QLoRA test

```bash
python -m scripts.train_dpo --config configs/dpo_smoke.yaml
```

Then use `configs/dpo_full.yaml` for the planned ~10,000 pairs:

```bash
python -m scripts.train_dpo --config configs/dpo_full.yaml
```

**Reference-model correctness:** DPO starts from `outputs/sft_merged`, **not** the raw Qwen base model. TRL's PEFT reference mechanism uses the initial SFT policy when no separate reference model is provided. This avoids comparing against an incorrect pretrained reference.

## Research comparison protocol (to implement next)

- **Quality:** IFEval primary, MMLU/ARC-Challenge/WikiText-2 perplexity regression checks; optional GSM8K and MT-Bench.
- **Prompt format control:** Each checkpoint (base, SFT, DPO) should be evaluated under both raw/few-shot and chat-formatted/zero-shot conditions; report them separately rather than comparing unlike prompt formats as if the difference were exclusively fine-tuning.
- **Serving:** Benchmark BF16/FP16 and each INT4 method on the **same GPU**, identical request lengths and concurrency settings, fixed seeds, one warm-up followed by **at least three measured runs**, reporting mean and standard deviation. vLLM max concurrency is not guaranteed to equal internal batch size.
- **Quantization:** GPTQ and AWQ, each with generic and chat-formatted calibration distributions; same quality benchmark suite for all variants.
- **Systems extension:** Triton INT4 dequantization kernel or documented KV-cache / Nsight fallback.

## Scientific integrity / reproducibility

- Git stores source, configuration, code reviewable by the advisor, and compact result summaries. It **does not store** model weights, private tokens, full downloaded datasets, or credentials.
- Do not claim results for experiments that have not been run.
- For a formal comparison, record exact HF model revision, dataset revision/fingerprint, software lockfile, driver/CUDA details, command line, seed, and hardware specification. `run_metadata.json` captures most software/hardware details; the exact upstream revision still needs to be pinned before final experiments.
- The SFT data `no_robots` is CC BY-NC 4.0; check its noncommercial license before any redistribution or commercial use. Only download it on the training machine.
- Document failed runs and OOM events alongside successful runs. A smoke test is not a substitute for a held-out evaluation.

## Reference documentation

- [Research plan: CIS 6900 Fall 2026, Revision 2] (see advisor-approved PDF)
- [Qwen2.5-7B base](https://huggingface.co/Qwen/Qwen2.5-7B)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [PEFT 4-bit QLoRA](https://huggingface.co/docs/peft/developer_guides/quantization)
- [no_robots dataset](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- [UltraFeedback binarized dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
