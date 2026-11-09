"""Initial base checkpoint smoke test (NOT an IFEval or latency benchmark)."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_study.configuration import parse_config_arg, read_config, write_json
from llm_study.runtime import record_run, require_cuda_bf16, seed_everything


def main():
    config = read_config(parse_config_arg('Run base model smoke generation'))
    require_cuda_bf16()
    seed_everything(config['seed'])
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    record_run(out, config)

    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], dtype=torch.bfloat16, device_map='auto'
    ).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen uses <|im_end|> to terminate chat completions.
    termination_ids = [tokenizer.eos_token_id]
    chat_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if isinstance(chat_end_id, int) and chat_end_id >= 0:
        termination_ids.append(chat_end_id)
    records = []
    for condition in config['conditions']:
        if condition == 'raw':
            input_batch = tokenizer(config['prompt'], return_tensors='pt')
        elif condition == 'chat':
            input_batch = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': config['prompt']}],
                add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors='pt',
            )
        else:
            raise ValueError(f'Unknown condition: {condition}')
        input_batch = input_batch.to(model.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **input_batch,
                max_new_tokens=int(config['max_new_tokens']),
                do_sample=False,
                eos_token_id=termination_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output[0, input_batch['input_ids'].shape[1]:]
        record = {
            'condition': condition,
            'prompt': config['prompt'],
            'response': tokenizer.decode(generated, skip_special_tokens=True),
            'prompt_tokens': int(input_batch['input_ids'].shape[1]),
            'generated_tokens': int(generated.numel()),
            'total_wall_seconds': elapsed,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / (1024**3),
        }
        records.append(record)
        print(f"\n=== {condition.upper()} ===\n{record['response']}\n")

    # No claim of TTFT/TPOT: these require a separate serving benchmark.
    write_json(out / 'generations.json', {'model_id': config['model_id'], 'records': records})
    print(f'Wrote {out / "generations.json"}')


if __name__ == '__main__':
    main()
