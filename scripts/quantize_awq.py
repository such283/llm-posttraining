import json
import sys
from pathlib import Path

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

condition = sys.argv[1]
assert condition in ("generic", "chat")

source = "outputs/dpo_merged"
output = f"outputs/quantized/awq_{condition}"

path = Path(f"data/calibration/{condition}.jsonl")

samples = [
    json.loads(line)
    for line in path.read_text().splitlines()
]

assert len(samples) == 128

dataset = Dataset.from_list(samples)

recipe = [
    AWQModifier(),
    QuantizationModifier(
        targets="Linear",
        scheme="W4A16_ASYM",
        ignore=["lm_head"],
    ),
]

oneshot(
    model=source,
    tokenizer=source,
    dataset=dataset,
    recipe=recipe,
    num_calibration_samples=len(dataset),
    max_seq_length=512,
    output_dir=output,
)

print("AWQ COMPLETE:", output)
