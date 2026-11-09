"""Reproducible Base vs SFT evaluation via EleutherAI LM Evaluation Harness.

This script deliberately runs base and adapter checkpoints in *separate*
processes, so model weights from one condition cannot leak into another.

For each checkpoint, evaluate IFEval / MCQ tasks in both raw and chat modes.
WikiText-2 perplexity is measured only with untemplated continuous text.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import yaml

from llm_study.configuration import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/eval_sft.yaml'))
    parser.add_argument('--limit', type=int, default=None, help='Smoke test only; not valid research results')
    parser.add_argument('--dry-run', action='store_true', help='Print planned commands, do not evaluate')
    return parser.parse_args()


def check_config(config: dict[str, Any], *, limit: int | None) -> None:
    required = ['model_id', 'adapter_dir', 'output_dir', 'tasks', 'seed', 'batch_size']
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'Missing configuration fields: {missing}')
    if not isinstance(config['tasks'], list) or not config['tasks']:
        raise ValueError('tasks must be a nonempty YAML list')
    if not all(isinstance(task, str) and task for task in config['tasks']):
        raise ValueError('Every task must be a nonempty string')
    if limit is not None and limit < 1:
        raise ValueError('--limit must be a positive integer')
    if not isinstance(config['seed'], int):
        raise ValueError('seed must be an integer')
    if str(config['batch_size']) not in {'1', '2', '4', '8'}:
        raise ValueError('Use a conservative batch_size of 1, 2, 4, or 8')
    if 'wikitext' in config['tasks'] and config['tasks'].count('wikitext') > 1:
        raise ValueError('wikitext task should appear only once')
    if limit is not None and 'wikitext' in config['tasks']:
        raise ValueError('A limited WikiText result is not meaningful; remove wikitext for smoke tests')


def resolve_launcher() -> list[str]:
    # The harness currently documents both the modern and legacy CLIs.
    executable = shutil.which('lm-eval')
    if executable:
        return [executable, 'run']
    executable = shutil.which('lm_eval')
    if executable:
        return [executable]
    raise RuntimeError('LM Evaluation Harness is missing. Install with: python -m pip install "lm_eval[hf]"')


def make_command(
    launcher: list[str],
    config: dict[str, Any],
    checkpoint: str,
    mode: str,
    tasks: list[str],
    output_dir: Path,
    limit: int | None,
) -> list[str]:
    if checkpoint not in {'base', 'sft'}:
        raise ValueError(f'Unknown checkpoint: {checkpoint}')
    if mode not in {'raw', 'chat'}:
        raise ValueError(f'Unknown prompt condition: {mode}')
    model_args = f'pretrained={config["model_id"]},dtype=bfloat16'
    if checkpoint == 'sft':
        model_args += f',peft={config["adapter_dir"]}'

    cmd = [
        *launcher, '--model', 'hf', '--model_args', model_args,
        '--tasks', ','.join(tasks), '--device', 'cuda:0',
        '--batch_size', str(config['batch_size']),
        '--num_fewshot', '0', '--seed', str(config['seed']),
        '--output_path', str(output_dir),
    ]
    if mode == 'chat':
        cmd.append('--apply_chat_template')
    if limit is not None:
        cmd.extend(['--limit', str(limit)])
    return cmd


def get_metrics(output_dir: Path) -> dict[str, dict[str, float]]:
    """Extract machine-readable metrics from harness result files.

    The CLI's file naming and nesting can change between versions, so inspect
    the output recursively rather than assuming an exact timestamped filename.
    """
    result_files = sorted(output_dir.rglob('results*.json'), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise RuntimeError(f'No harness result JSON files found in {output_dir}')
    payload = json.loads(result_files[-1].read_text(encoding='utf-8'))
    if not isinstance(payload.get('results'), dict):
        raise RuntimeError(f'Harness output did not contain a results mapping: {result_files[-1]}')
    result: dict[str, dict[str, float]] = {}
    for task, metrics in payload['results'].items():
        if not isinstance(metrics, dict):
            continue
        result[task] = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and (key.startswith(('prompt_level_', 'inst_level_', 'acc,', 'acc_norm,',
                                 'word_perplexity,', 'byte_perplexity,', 'bits_per_byte,')))
        }
    return result


def build_jobs(config: dict[str, Any], raw_root: Path, limit: int | None, launcher: list[str]):
    # WikiText is a rolling loglikelihood task. Do not apply a chat template:
    # wrapping continuous prose in instruction tokens changes what is measured.
    normal_tasks = [task for task in config['tasks'] if task != 'wikitext']
    for checkpoint in ('base', 'sft'):
        for mode in ('raw', 'chat'):
            if normal_tasks:
                dest = raw_root / checkpoint / mode
                yield (checkpoint, mode, make_command(
                    launcher, config, checkpoint, mode, normal_tasks, dest, limit
                ), dest)
        if 'wikitext' in config['tasks']:
            dest = raw_root / checkpoint / 'raw_wikitext'
            yield (checkpoint, 'raw_wikitext', make_command(
                launcher, config, checkpoint, 'raw', ['wikitext'], dest, None
            ), dest)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Evaluation config must be a YAML mapping')
    check_config(config, limit=args.limit)
    if not args.dry_run and not (Path(config['adapter_dir']) / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Cannot find trained SFT adapter: {config["adapter_dir"]}')
    launcher = resolve_launcher() if not args.dry_run else ['lm-eval', 'run']
    root = Path(config['output_dir'])
    # Raw harness outputs are excluded from Git by .gitignore.
    raw_root = Path('results/raw/harness') / root.name
    jobs = list(build_jobs(config, raw_root, args.limit, launcher))
    summary: dict[str, Any] = {
        'model_id': config['model_id'],
        'adapter_dir': config['adapter_dir'],
        'tasks': config['tasks'],
        'seed': config['seed'],
        'batch_size': config['batch_size'],
        'num_fewshot': 0,
        'limit': args.limit,
        'provisional': args.limit is not None,
        'conditions': {},
    }
    if args.limit is not None:
        print('WARNING: --limit enables a smoke test; results are NOT research-quality.')
    for checkpoint, mode, cmd, dest in jobs:
        print('\n==>', checkpoint, mode, ' '.join(config['tasks']), flush=True)
        print(' '.join(cmd), flush=True)
        if args.dry_run:
            continue
        # Never run two GPU-heavy benchmark jobs simultaneously.
        dest.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault('TOKENIZERS_PARALLELISM', 'false')
        subprocess.run(cmd, check=True, env=env)
        metrics = get_metrics(dest)
        summary['conditions'].setdefault(checkpoint, {})[mode] = metrics
        write_json(root / ('smoke_summary.json' if args.limit else 'summary.json'), summary)
    if not args.dry_run:
        print('\nSaved compact metrics to', root / ('smoke_summary.json' if args.limit else 'summary.json'))
        print('Raw harness reports are stored in', raw_root)


if __name__ == '__main__':
    main()
