"""Read-only GPU, software, and disk diagnostics. Does not download models."""
from __future__ import annotations

import shutil
from importlib import metadata
from pathlib import Path

import torch


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'NOT INSTALLED'


def main():
    print('=== GPU PRE-FLIGHT ===')
    print('PyTorch:', torch.__version__)
    print('PyTorch CUDA runtime:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('BF16 supported:', torch.cuda.is_bf16_supported())
        print('VRAM (GiB):', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    print('\n=== PACKAGES ===')
    for package in ('transformers', 'trl', 'peft', 'accelerate', 'datasets', 'bitsandbytes', 'PyYAML'):
        print(f'{package}: {package_version(package)}')
    print('\n=== STORAGE ===')
    workspace = Path('/workspace') if Path('/workspace').exists() else Path.cwd()
    usage = shutil.disk_usage(workspace)
    free_gib = usage.free / 1024**3
    print('Directory:', workspace)
    print('Free disk (GiB):', round(free_gib, 2))
    if free_gib < 90:
        print('WARNING: Low disk for full SFT->merge->DPO experiments. Plan for 100-150GB.')


if __name__ == '__main__':
    main()
