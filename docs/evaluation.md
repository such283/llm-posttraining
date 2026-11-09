# Base vs SFT evaluation

**Purpose:** Test RQ1's instruction-following benefit and possible regression on general knowledge/language modeling. Training loss alone is not a Base-vs-SFT quality metric.

## Installation

Install the Hugging Face evaluation backend in the **same Python environment** as the project:

```bash
python -m pip install 'lm_eval[hf]'
lm-eval ls tasks --pattern 'ifeval*'
```

The harness may adjust dependencies. Verify `python -m scripts.preflight` afterwards, and capture `python -m pip freeze > environment.freeze.txt` once the setup works.

## Steps

From the repository root (with your SFT adapter available at `outputs/sft_full/adapter`):

```bash
# Inspect the exact commands without loading models
python -m scripts.evaluate_sft --config configs/eval_sft.yaml --dry-run

# Limit is only for validating wiring; not for the research report
python -m scripts.evaluate_sft --config configs/eval_sft.yaml --limit 10

# Run the **full** IFEval once the smoke test passes (no --limit)
python -m scripts.evaluate_sft --config configs/eval_sft.yaml
```

The script evaluates **Base/raw, Base/chat, SFT/raw, SFT/chat** in distinct processes to prevent accidental model reuse. It fixes decoding using the task's official configuration, including IFEval's maximum generation length. Both conditions are zero-shot; this is a controlled prompt-format comparison. If adding few-shot benchmarks later, run the equivalent matched few-shot conditions rather than mixing protocols.

After IFEval, extend `configs/eval_sft.yaml` with `arc_challenge`, `mmlu`, and `wikitext`. `wikitext` is evaluated *raw-only* because it is a continuous-text perplexity task. Those additional benchmarks may require substantial GPU time and disk space.

## Outputs

- `results/eval_sft/smoke_summary.json`: limited preliminary results; **not** valid evidence.
- `results/eval_sft/summary.json`: full results; suitable to commit when verified.
- `results/raw/harness/eval_sft/`: complete harness outputs; ignored in Git.

Metrics include `prompt_level_strict_acc,none` and `inst_level_strict_acc,none` for IFEval; `acc`/`acc_norm` for choice tasks; and WikiText word perplexity. The summary includes all numeric metrics reported by the harness, plus the benchmark config and provisional flag. No metric should be interpreted as measured until it appears in a completed run.

**Check before citing in a report:** evaluation package version, dataset/model revisions, metrics, prompts, max generation lengths, and hardware environment. Do not compare tuned/chat with base/raw in isolation as though the difference only came from fine-tuning.

References: [Harness CLI](https://lm-evaluation-harness.readthedocs.io/running_evals/cli_reference/), [PEFT adapters](https://github.com/EleutherAI/lm-evaluation-harness), [IFEval task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/ifeval.yaml).
