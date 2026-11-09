"""Show the exact dataset records fed to SFT or DPO, without training."""
from __future__ import annotations

import argparse
import json

from datasets import load_dataset
from llm_study.data import sft_record, dpo_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['sft', 'dpo'], required=True)
    parser.add_argument('--n', type=int, default=2)
    args = parser.parse_args()
    if args.n < 1 or args.n > 10:
        parser.error('--n must be between 1 and 10')
    dataset_id, split, convert = {
        'sft': ('HuggingFaceH4/no_robots', 'train', sft_record),
        'dpo': ('HuggingFaceH4/ultrafeedback_binarized', 'train_prefs', dpo_record),
    }[args.stage]
    raw = load_dataset(dataset_id, split=split)
    print(f'{dataset_id} split={split} total={len(raw)}')
    count = 0
    for row in raw:
        transformed = convert(row)
        if transformed is None:
            continue
        print(json.dumps(transformed, indent=2, ensure_ascii=False)[:3500])
        count += 1
        if count >= args.n:
            break
    if count < args.n:
        raise RuntimeError(f'Only {count} valid samples were found')


if __name__ == '__main__':
    main()
