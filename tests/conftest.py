"""A tiny self-contained repo for tests: config, prompts, no network."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from poemgen.config import load_config

REPO = Path(__file__).resolve().parents[1]

MODELS = """
[small]
name = "Small"
tag = "small:q8"
upstream = "org/small"
quant = "Q8_0"
confirmed = true
weights_sha256 = "sha256:aaa"
ollama_digest = "d1"
ctx_max = 32768
think_values = [false, true]
think_default = false
stop = ["<end>"]
recommended_source = "card"
unsupported = { dry = "DRY sampler" }

[small.recommended]
temperature = 0.7
top_p = 0.8
top_k = 20

[small.recommended.thinking]
temperature = 0.6
top_p = 0.95

[plain]
name = "Plain"
tag = "plain:q4"
confirmed = true
weights_sha256 = "sha256:bbb"
ollama_digest = "d2"
think_values = []
inherited_source = "base model"
inherited = { temperature = 0.9, top_p = 0.9 }
recommended_source = "card"
recommended = { temperature = 1.0 }
"""

MACHINE = """
name = "test"
ollama_host = "http://127.0.0.1:1"
parallel = 1
minutes_per_poem = 2
[num_ctx_caps]
plain = 16384
"""

EXPERIMENT = """
models = ["small", "plain"]
system_prompts = ["a", "b"]
user_prompt = "u"
samples = 2
seed = 100

[[settings]]
name = "base"

[[settings]]
name = "hot"
think = true
all = { temperature = 1.3 }
models.small = { min_p = 0.05 }
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "defaults.toml", tmp_path / "config" / "defaults.toml")
    (tmp_path / "config" / "models.toml").write_text(MODELS, encoding="utf-8")
    (tmp_path / "config" / "machine.toml").write_text(MACHINE, encoding="utf-8")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "exp.toml").write_text(EXPERIMENT, encoding="utf-8")
    for kind, name, text in (
        ("system", "a", "Oṃ Gaṃ Gaṇapataye Namaḥ! Prompt A.\n"),
        ("system", "b", "Prompt B.\n"),
        ("user", "u", "Please write a poem."),
    ):
        (tmp_path / "prompts" / kind).mkdir(parents=True, exist_ok=True)
        (tmp_path / "prompts" / kind / f"{name}.md").write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def cfg(repo: Path):
    return load_config(repo)
