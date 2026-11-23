import json
import sys
from pathlib import Path

from gptqmodel import (
    BACKEND,
    CalibrationConfig,
    GPTQModel,
    QuantizeConfig,
)
from transformers import AutoTokenizer

condition = sys.argv[1]
assert condition in ("generic", "chat")

source = "outputs/dpo_merged"
output = f"outputs/quantized/gptq_{condition}"

with open(f"data/calibration/{condition}.jsonl") as f:
    calibration = [json.loads(line) for line in f]

config = QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,
)

model = GPTQModel.load(
    source,
    quantize_config=config,
    device="cuda:0",
)

model.quantize(
    CalibrationConfig(
        dataset=calibration,
        batch_size=1,
    ),
    backend=BACKEND.GPTQ_TORCH,
)

model.save(output)

tokenizer = AutoTokenizer.from_pretrained(source)
tokenizer.save_pretrained(output)

print("GPTQ COMPLETE:", output)
