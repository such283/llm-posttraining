"""DPO QLoRA from the *merged SFT* checkpoint, not the raw base model.

TRL uses the initial SFT policy as reference via adapter disablement.
"""
from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from llm_study.configuration import parse_config_arg, read_config
from llm_study.data import dpo_record, load_processed_dataset
from llm_study.runtime import record_run, require_cuda_bf16, seed_everything


def main():
    config = read_config(parse_config_arg('Train QLoRA DPO'))
    require_cuda_bf16()
    seed_everything(config['seed'])
    source = Path(config['model_id'])
    if not source.is_dir() or not (source / 'config.json').exists():
        raise FileNotFoundError(
            f'DPO needs the merged SFT checkpoint at {source}. Run merge_sft.py first.'
        )
    out = Path(config['output_dir'])
    train, validation, details = load_processed_dataset(
        dataset_id=config['dataset_id'], split=config['dataset_split'],
        seed=config['seed'], limit=config.get('limit'),
        test_fraction=float(config['validation_fraction']),
        transform=dpo_record,
    )
    record_run(out, config, extra={
        'dataset': details,
        'reference_policy': str(source),
        'reference_method': 'TRL adapter-disabled initial policy',
    })
    print('Dataset:', details)
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    qlora = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora = LoraConfig(
        r=config['lora_r'], lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'], bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    )
    args = DPOConfig(
        output_dir=str(out),
        model_init_kwargs={'dtype': torch.bfloat16},
        seed=config['seed'], data_seed=config['seed'],
        max_length=config['max_length'],
        beta=float(config['beta']),
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=float(config['learning_rate']),
        max_steps=config['max_steps'],
        num_train_epochs=float(config.get('num_train_epochs', 1.0)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='paged_adamw_8bit',
        logging_steps=1,
        save_strategy=config['save_strategy'],
        save_total_limit=2,
        report_to='none',
        eval_strategy='epoch',
        use_cache=False,
    )
    trainer = DPOTrainer(
        model=str(source),
        ref_model=None,
        args=args,
        train_dataset=train,
        eval_dataset=validation,
        processing_class=tokenizer,
        quantization_config=qlora,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(out / 'adapter'))
    trainer.save_state()
    print('DPO adapter saved to', out / 'adapter')


if __name__ == '__main__':
    main()
