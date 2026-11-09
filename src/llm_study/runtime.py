"""Record run metadata without storing credentials or private environment variables."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .configuration import write_json


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
            check=False, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def record_run(directory: str | Path, config: dict[str, Any], extra: dict[str, Any] | None = None):
    import torch

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'python': sys.version,
        'platform': platform.platform(),
        'packages': {
            name: _version(name)
            for name in ('torch', 'transformers', 'trl', 'peft', 'accelerate',
                         'datasets', 'bitsandbytes', 'safetensors')
        },
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_version': torch.version.cuda,
        'config': config,
        'extra': extra or {},
    }
    write_json(path / 'run_metadata.json', payload)
    return payload


def require_cuda_bf16():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not detected')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This configuration requires a GPU supporting BF16')


def seed_everything(seed: int):
    import random
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
