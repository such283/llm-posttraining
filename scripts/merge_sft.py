"""Merge SFT adapter into a BF16 checkpoint for the DPO reference policy.

This is a distinct step from final GPTQ/AWQ quantization. Requires additional
host disk space for the merged model (~15GB for Qwen2.5-7B).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_study.configuration import write_json
from llm_study.runtime import require_cuda_bf16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/sft_merged'))
    args = parser.parse_args()
    require_cuda_bf16()
    if not (args.adapter / 'adapter_config.json').is_file():
        parser.error(f'Not a PEFT adapter directory: {args.adapter}')
    adapter_config = PeftConfig.from_pretrained(str(args.adapter))
    base_id = adapter_config.base_model_name_or_path
    if not base_id:
        raise ValueError('Adapter config does not specify base_model_name_or_path')
    print(f'Merging adapter {args.adapter} with {base_id}')
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter))
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16, device_map='auto'
    )
    adapted = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    merged = adapted.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output))
    write_json(args.output / 'merge_manifest.json', {
        'base_model': base_id,
        'sft_adapter': str(args.adapter),
        'dtype': 'bfloat16',
        'purpose': 'Frozen SFT initialization/reference for DPO',
    })
    print('Merged checkpoint saved to', args.output)


if __name__ == '__main__':
    main()
