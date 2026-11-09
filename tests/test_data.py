from llm_study.data import sft_record, dpo_record


def test_sft_keeps_history_but_only_targets_last_answer():
    messages = [
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'hi'},
        {'role': 'user', 'content': 'define gradient'},
        {'role': 'assistant', 'content': 'A vector of derivatives.'},
    ]
    prepared = sft_record({'messages': messages})
    assert prepared['prompt'] == messages[:-1]
    assert prepared['completion'] == [messages[-1]]


def test_sft_rejects_incomplete_example():
    assert sft_record({'messages': [{'role': 'user', 'content': 'hello'}]}) is None


def test_dpo_keeps_shared_prompt_and_separate_answers():
    prompt = [{'role': 'user', 'content': 'What is 2+2?'}]
    chosen = prompt + [{'role': 'assistant', 'content': '4'}]
    rejected = prompt + [{'role': 'assistant', 'content': '5'}]
    sample = dpo_record({'chosen': chosen, 'rejected': rejected})
    assert sample == {
        'prompt': prompt,
        'chosen': [{'role': 'assistant', 'content': '4'}],
        'rejected': [{'role': 'assistant', 'content': '5'}],
    }


def test_dpo_rejects_mismatched_prompts():
    assert dpo_record({
        'chosen': [{'role': 'user', 'content': 'A'}, {'role': 'assistant', 'content': 'Yes'}],
        'rejected': [{'role': 'user', 'content': 'B'}, {'role': 'assistant', 'content': 'No'}],
    }) is None


def test_dpo_rejects_identical_preferences():
    messages = [{'role': 'user', 'content': 'A'}, {'role': 'assistant', 'content': 'Same'}]
    assert dpo_record({'chosen': messages, 'rejected': messages}) is None
