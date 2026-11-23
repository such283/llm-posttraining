"""Reproducible IFEval comparison of Qwen2.5-7B Base vs SFT with vLLM.

Use one subprocess per condition to keep model initialization isolated.
Run a small --limit smoke test before the full, 541-prompt IFEval evaluation.

Examples:
  python -m scripts.evaluate_sft_vllm --limit 10 --conditions base:chat,sft:chat
  python -m scripts.evaluate_sft_vllm --conditions base:chat,sft:chat --resume
  python -m scripts.evaluate_sft_vllm --resume
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml

ALL_CONDITIONS = ("base:raw", "base:chat", "sft:raw", "sft:chat")


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(x.strip() for x in value.split(",") if x.strip())
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Specify unique, nonempty conditions")
    invalid = set(conditions) - set(ALL_CONDITIONS)
    if invalid:
        raise ValueError(f"Invalid conditions: {sorted(invalid)}. Choose from {ALL_CONDITIONS}")
    return conditions


def check_config(config: dict[str, Any]) -> None:
    required = {"model_id", "adapter_dir", "output_dir", "seed", "tasks", "batch_size", "max_model_len", "max_num_seqs", "gpu_memory_utilization", "max_lora_rank", "dtype"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    if not config["tasks"] or any(not isinstance(t, str) for t in config["tasks"]):
        raise ValueError("tasks must be a nonempty list of task names")
    if not (0 < float(config["gpu_memory_utilization"]) <= 1):
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if int(config["max_model_len"]) < 2048:
        raise ValueError("max_model_len must be at least 2048 for the IFEval generation cap")
    if int(config["max_lora_rank"]) < 1:
        raise ValueError("max_lora_rank must be positive")


def make_command(config: dict[str, Any], condition: str, destination: Path, limit: int | None, executable: str = "lm-eval") -> list[str]:
    checkpoint, mode = condition.split(":")
    model_args = [
        f"pretrained={config['model_id']}",
        f"dtype={config['dtype']}",
        f"max_model_len={int(config['max_model_len'])}",
        f"max_num_seqs={int(config['max_num_seqs'])}",
        f"gpu_memory_utilization={float(config['gpu_memory_utilization'])}",
        f"max_lora_rank={int(config['max_lora_rank'])}",
        f"seed={int(config['seed'])}",
    ]
    if checkpoint == "sft":
        model_args.append(f"lora_local_path={Path(config['adapter_dir']).resolve()}")
    command = [
        executable, "run", "--model", "vllm", "--model_args", ",".join(model_args),
        "--tasks", ",".join(config["tasks"]),
        "--batch_size", str(config["batch_size"]),
        "--num_fewshot", "0", "--seed", str(config["seed"]),
        "--output_path", str(destination),
    ]
    if mode == "chat":
        command.append("--apply_chat_template")
    if limit is not None:
        command.extend(("--limit", str(limit)))
    return command


def get_metrics(path: Path) -> dict[str, dict[str, float]]:
    result_files = sorted(path.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise FileNotFoundError(f"No completed lm-eval result found under {path}")
    result = json.loads(result_files[-1].read_text())
    tasks = result.get("results")
    if not isinstance(tasks, dict):
        raise ValueError(f"Missing 'results' map in {result_files[-1]}")
    return {
        task: {name: float(value) for name, value in scores.items()
               if (name.startswith(("prompt_level_", "inst_level_", "acc,", "acc_norm,", "word_perplexity,"))
                   and isinstance(value, (int, float)) and not isinstance(value, bool))}
        for task, scores in tasks.items()
        if isinstance(scores, dict)
    }


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_sft_vllm.yaml"))
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS), help="Comma-separated base:raw,base:chat,sft:raw,sft:chat")
    parser.add_argument("--limit", type=int, default=None, help="SMOKE TEST ONLY; do not use in final report")
    parser.add_argument("--resume", action="store_true", help="Reuse successful conditions from this run mode")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    conditions = parse_conditions(args.conditions)
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("YAML configuration must be a mapping")
    check_config(config)
    if any(c.startswith("sft:") for c in conditions) and not args.dry_run:
        adapter = Path(config["adapter_dir"])
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter / filename).is_file():
                raise FileNotFoundError(f"Missing {adapter / filename}")
    executable = shutil.which("lm-eval") or "lm-eval"
    if not args.dry_run and shutil.which("lm-eval") is None:
        raise RuntimeError("Cannot find lm-eval. Activate vLLM venv before running")

    kind = "smoke" if args.limit is not None else "full"
    raw_root = Path("results/raw/harness/eval_sft_vllm") / kind
    output = Path(config["output_dir"]) / ("smoke_summary.json" if args.limit is not None else "summary.json")
    summary: dict[str, Any] = {
        "engine": "vllm", "model_id": config["model_id"],
        "adapter_dir": str(config["adapter_dir"]),
        "configuration": config, "seed": config["seed"],
        "limit": args.limit, "provisional": args.limit is not None,
        "conditions": {},
        "note": "IFEval quality metrics, not GPU throughput or TTFT/TPOT metrics",
    }
    for condition in conditions:
        checkpoint, mode = condition.split(":")
        destination = raw_root / checkpoint / mode
        command = make_command(config, condition, destination, args.limit, executable)
        print(f"\n==> {condition} ({kind})", flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        if args.resume:
            try:
                metrics = get_metrics(destination)
                print("Reusing completed result", destination, flush=True)
                summary["conditions"][condition] = metrics
                save_json_atomic(output, summary)
                continue
            except (FileNotFoundError, ValueError):
                pass
        destination.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        subprocess.run(command, check=True, env=env)
        metrics = get_metrics(destination)
        summary["conditions"][condition] = metrics
        save_json_atomic(output, summary)
    if not args.dry_run:
        print(f"\nSaved: {output}", flush=True)
        print(f"Raw evaluator output: {raw_root}", flush=True)


if __name__ == "__main__":
    main()
