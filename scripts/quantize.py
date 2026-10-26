"""Quantizes models flagged for quantization. If a model with the same base revision has already been quantized and saved, nothing happens."""

import tomllib
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, MetalConfig


DEVICE_MAP = "mps" # eventually want this to run on Linux out of the box too; though there `transformers` natively supports the various quantized versions already available on Hugging Face
# Also, you technically end up with different models depending on the quantization algorithm used and the platform you run it on... noted for reproducibility

with open("src/model_config.toml", "rb") as m:
    model_info = tomllib.load(m)

for llm in model_info["llm"]:
    model_name = llm["name"]
    commit_hash = llm["commit_hash"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if llm["quantization"]["quantize"]:
        bits = llm["quantization"]["bits"]
        group_size = llm["quantization"]["group_size"]
        quantization_config = MetalConfig(bits=bits, group_size=group_size)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=DEVICE_MAP,
            quantization_config=quantization_config,
            revision=commit_hash + f"q{bits}"
        )

