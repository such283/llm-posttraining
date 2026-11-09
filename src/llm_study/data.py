"""Transparent dataset preprocessing for the SFT and DPO experiments.

The split is performed after deterministic selection; no evaluation dataset is
used for training. Each record has its prompt separated from its target answer.
"""
from __future__ import annotations

from typing import Any


def _valid_messages(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and bool(messages)
        and all(
            isinstance(msg, dict)
            and msg.get('role') in {'user', 'assistant', 'system'}
            and isinstance(msg.get('content'), str)
            and bool(msg['content'].strip())
            for msg in messages
        )
    )


def sft_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """Use all preceding messages as context and the final assistant turn as target.

    TRL's conversational prompt-completion format computes loss on the target
    completion, not the user's prompt. Retaining all earlier context is useful
    for multi-turn dialogues in no_robots.
    """
    messages = row.get('messages')
    if not _valid_messages(messages) or len(messages) < 2:
        return None
    if messages[-1]['role'] != 'assistant':
        return None
    if not any(msg['role'] == 'user' for msg in messages[:-1]):
        return None
    return {'prompt': messages[:-1], 'completion': [messages[-1]]}


def dpo_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the shared prompt and two assistant answers from UltraFeedback.

    Never include the preferred response in the prompt: doing so would leak
    the preference label into training.
    """
    chosen, rejected = row.get('chosen'), row.get('rejected')
    if not _valid_messages(chosen) or not _valid_messages(rejected):
        return None
    if len(chosen) < 2 or len(rejected) < 2:
        return None
    if chosen[-1]['role'] != 'assistant' or rejected[-1]['role'] != 'assistant':
        return None
    if chosen[:-1] != rejected[:-1]:
        return None
    if not any(msg['role'] == 'user' for msg in chosen[:-1]):
        return None
    if chosen[-1]['content'].strip() == rejected[-1]['content'].strip():
        return None
    return {
        'prompt': chosen[:-1],
        'chosen': [chosen[-1]],
        'rejected': [rejected[-1]],
    }


def load_processed_dataset(
    *, dataset_id: str, split: str, seed: int, limit: int | None,
    transform, test_fraction: float = 0.02
):
    """Load, shuffle, transform, and create deterministic train/validation sets."""
    from datasets import Dataset, load_dataset

    raw = load_dataset(dataset_id, split=split)
    raw = raw.shuffle(seed=seed)
    if limit is not None:
        if not isinstance(limit, int) or limit < 2:
            raise ValueError('limit must be an integer >= 2')
        raw = raw.select(range(min(limit, len(raw))))
    rows = [record for row in raw if (record := transform(row)) is not None]
    if len(rows) < 2:
        raise ValueError('Fewer than 2 valid samples after preprocessing')
    dataset = Dataset.from_list(rows)
    split_ds = dataset.train_test_split(test_size=test_fraction, seed=seed)
    details = {
        'dataset_id': dataset_id,
        'dataset_split': split,
        'seed': seed,
        'raw_sample_count': len(raw),
        'valid_sample_count': len(rows),
        'train_sample_count': len(split_ds['train']),
        'validation_sample_count': len(split_ds['test']),
        'transformation': transform.__name__,
        'limit': limit,
        'test_fraction': test_fraction,
    }
    return split_ds['train'], split_ds['test'], details
