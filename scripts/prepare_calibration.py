
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

SEED = 42
NUM_SAMPLES = 128
MAX_TOKENS = 512

OUTPUT = Path("data/calibration")
OUTPUT.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B"
)


def save_samples(name, samples, source, fingerprint):
    path = OUTPUT / f"{name}.jsonl"

    with path.open("w") as f:
        for ids in samples:
            row = {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
            }
            f.write(json.dumps(row) + "\n")

    metadata = {
        "source": source,
        "dataset_fingerprint": fingerprint,
        "seed": SEED,
        "num_samples": len(samples),
        "max_tokens": MAX_TOKENS,
        "lengths": [len(ids) for ids in samples],
    }

    (OUTPUT / f"{name}_manifest.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(f"{name}: {len(samples)} samples saved")


# Calibration A: generic WikiText-2
wiki = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="train",
)

text = "\n".join(
    row["text"]
    for row in wiki
    if row["text"].strip()
)

tokens = tokenizer.encode(
    text,
    add_special_tokens=False,
)

chunks = [
    tokens[i:i + MAX_TOKENS]
    for i in range(0, len(tokens), MAX_TOKENS)
]

chunks = [c for c in chunks if len(c) == MAX_TOKENS]
assert len(chunks) >= NUM_SAMPLES

rng = random.Random(SEED)
generic_samples = rng.sample(chunks, NUM_SAMPLES)

save_samples(
    "generic",
    generic_samples,
    "Salesforce/wikitext:wikitext-2-raw-v1:train",
    wiki._fingerprint,
)


# Calibration B: chat-formatted no_robots
chat = load_dataset(
    "HuggingFaceH4/no_robots",
    split="train",
).shuffle(seed=SEED)

chat_samples = []

for row in chat:
    messages = row["messages"]

    if not messages:
        continue

    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        truncation=True,
        max_length=MAX_TOKENS,
    )

    if len(ids) >= 64:
        chat_samples.append(ids)

    if len(chat_samples) == NUM_SAMPLES:
        break

assert len(chat_samples) == NUM_SAMPLES

save_samples(
    "chat",
    chat_samples,
    "HuggingFaceH4/no_robots:train",
    chat._fingerprint,
)

