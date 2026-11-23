import json
from pathlib import Path

from scripts.evaluate_sft_vllm import parse_conditions, make_command, get_metrics

CFG = {
    "model_id": "Qwen/Qwen2.5-7B", "adapter_dir": "outputs/sft_full/adapter",
    "dtype": "bfloat16", "tasks": ["ifeval"], "seed": 42,
    "batch_size": "auto", "gpu_memory_utilization": 0.8,
    "max_model_len": 4096, "max_num_seqs": 16, "max_lora_rank": 16
}


def test_select_conditions():
    assert parse_conditions("base:chat,sft:chat") == ("base:chat", "sft:chat")


def test_sft_loads_adapter_and_chat_template():
    cmd = make_command(CFG, "sft:chat", Path("tmp/sft"), 10)
    assert cmd[cmd.index("--model") + 1] == "vllm"
    assert "lora_local_path=" in cmd[cmd.index("--model_args") + 1]
    assert "--apply_chat_template" in cmd
    assert cmd[-2:] == ["--limit", "10"]


def test_base_has_no_adapter():
    cmd = make_command(CFG, "base:raw", Path("tmp/base"), None)
    assert "lora_local_path=" not in cmd[cmd.index("--model_args") + 1]
    assert "--apply_chat_template" not in cmd


def test_read_ifeval_metrics(tmp_path):
    (tmp_path / "results_123.json").write_text(json.dumps({"results": {
        "ifeval": {"prompt_level_strict_acc,none": 0.3, "inst_level_strict_acc,none": 0.4, "alias": "IFEval"}
    }}))
    result = get_metrics(tmp_path)
    assert result["ifeval"]["prompt_level_strict_acc,none"] == 0.3
