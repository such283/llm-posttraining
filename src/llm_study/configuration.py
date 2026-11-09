"""Read and validate experiment configuration files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_config_arg(description: str) -> Path:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, required=True, help='Path to YAML configuration')
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f'Configuration file not found: {args.config}')
    return args.config


def read_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open('r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f'Expected a mapping in {path}')
    if not isinstance(config.get('model_id'), str) or not config['model_id']:
        raise ValueError('Missing nonempty model_id')
    if not isinstance(config.get('seed'), int):
        raise ValueError('Missing integer seed')
    return config


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
