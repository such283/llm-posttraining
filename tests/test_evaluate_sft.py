"""CPU-only tests for experiment planning and metric extraction."""
import json
from pathlib import Path

from scripts.evaluate_sft import build_jobs, get_metrics, make_command, check_config


def cfg():
    return {
        'model_id': 'Qwen/Qwen2.5-7B',
        'adapter_dir': 'outputs/sft_full/adapter',
        'output_dir': 'results/eval_sft',
        'tasks': ['ifeval', 'wikitext'],
        'seed': 42, 'batch_size': 1,
    }


def test_jobs_and_peft_reference():
    jobs = list(build_jobs(cfg(), Path('results/raw/harness/eval_sft'), None, ['lm-eval', 'run']))
    assert len(jobs) == 6
    by_job = {(ckpt, mode): cmd for ckpt, mode, cmd, _ in jobs}
    assert '--apply_chat_template' in by_job['sft', 'chat']
    assert '--apply_chat_template' not in by_job['sft', 'raw']
    assert '--apply_chat_template' not in by_job['sft', 'raw_wikitext']
    assert any('peft=outputs/sft_full/adapter' in c for c in by_job['sft', 'chat'])
    assert not any('peft=' in c for c in by_job['base', 'raw'])
    assert by_job['base', 'raw_wikitext'][by_job['base', 'raw_wikitext'].index('--tasks') + 1] == 'wikitext'


def test_limited_wikitext_is_rejected():
    try:
        check_config(cfg(), limit=10)
    except ValueError as err:
        assert 'WikiText' in str(err)
    else:
        raise AssertionError('Expected a validation error')


def test_extract_harness_metrics(tmp_path):
    nested = tmp_path / 'hf' / 'model'
    nested.mkdir(parents=True)
    (nested / 'results_2026.json').write_text(json.dumps({
        'results': {
            'ifeval': {'prompt_level_strict_acc,none': 0.3, 'inst_level_strict_acc,none': 0.6, 'alias': 'IFEval'},
        }
    }))
    result = get_metrics(tmp_path)
    assert result['ifeval'] == {'prompt_level_strict_acc,none': 0.3, 'inst_level_strict_acc,none': 0.6}


def test_make_command_keeps_values_as_tokens():
    command = make_command(['lm-eval', 'run'], cfg(), 'base', 'raw', ['ifeval'], Path('results/raw'), 10)
    assert '--limit' in command
    assert '--num_fewshot' in command
    assert command[command.index('--num_fewshot') + 1] == '0'
