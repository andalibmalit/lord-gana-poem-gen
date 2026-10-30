"""The run loop against a fake Ollama: records, resume, failures and checks."""

from __future__ import annotations

import json

import pytest

from poemgen import records
from poemgen.config import load_config, load_experiment
from poemgen.ollama import OllamaError
from poemgen.plan import build_plan
from poemgen.run import Runner, parse_stop, redact_modelfile

WEIGHTS = "a" * 64


class FakeOllama:
    def __init__(
        self, fail_times: int = 0, content: str = "Gaṇeśa, threshold-keeper,\nopen the door."
    ):
        self.fail_times = fail_times
        self.content = content
        self.requests: list[dict] = []
        self.unloaded: list[str] = []

    def version(self):
        return "0.40.0"

    def tags(self):
        return {"small:q8": "d1", "plain:q4": "d2"}

    def show(self, model):
        return {
            "modelfile": f"# generated\nFROM /home/me/.ollama/models/blobs/sha256-{WEIGHTS}\n",
            "parameters": 'stop                           "<end>"' if model == "small:q8" else None,
            "capabilities": ["completion", "thinking"] if model == "small:q8" else ["completion"],
            "details": {"quantization_level": "Q8_0", "family": "fake"},
            "model_info": {"fake.context_length": 32768 if model == "small:q8" else 8192},
        }

    def loaded(self, model):
        return {"size": 10, "size_vram": 10, "context_length": 4096}

    def chat(self, body):
        self.requests.append(body)
        if self.fail_times:
            self.fail_times -= 1
            raise OllamaError("boom")
        message = {"role": "assistant", "content": self.content}
        if body["think"]:
            message["thinking"] = "Let me think about the elephant-headed one."
        return {
            "model": body["model"],
            "message": message,
            "done": True,
            "done_reason": "stop",
            "total_duration": 90 * 10**9,
            "prompt_eval_count": 120,
            "eval_count": 40,
        }

    def render(self, body):
        system, user = (m["content"] for m in body["messages"])
        return f"<sys>{system}</sys><user>{user}</user>"

    def unload(self, model):
        self.unloaded.append(model)


@pytest.fixture
def cfg(repo):
    # Pin the fake build's weights so the checks pass.
    path = repo / "config" / "models.toml"
    text = path.read_text(encoding="utf-8")
    text = text.replace('"sha256:aaa"', f'"sha256:{WEIGHTS}"').replace(
        '"sha256:bbb"', f'"sha256:{WEIGHTS}"'
    )
    path.write_text(text, encoding="utf-8")
    return load_config(repo)


def run(cfg, client, models=None):
    plan = build_plan(cfg, load_experiment(cfg, "exp"), models)
    return plan, Runner(cfg, client, log=lambda _: None).run(plan)


def test_writes_complete_records(cfg):
    client = FakeOllama()
    plan, summary = run(cfg, client, ["small"])
    assert summary.written == len(plan.specs) == 8 and summary.failed == 0
    hot = next(s for s in plan.specs if s.settings_set == "hot")
    path = records.poem_path(cfg.poems_dir, "exp", hot.poem_id)
    meta, poem = records.load_poem(path)
    assert poem == "Gaṇeśa, threshold-keeper,\nopen the door."  # verbatim, UTF-8
    assert meta["request"]["options"] == hot.options
    assert meta["request"]["think"] is True and meta["request"]["truncate"] is False
    assert meta["request"]["messages"][0]["sha256"] == hot.system.sha256
    assert "content" not in meta["request"]["messages"][0]
    assert meta["origins"]["temperature"] == "change:hot"
    assert meta["model"]["weights_sha256"] == f"sha256:{WEIGHTS}"
    assert meta["model"]["digest"] == "d1"
    assert meta["ollama_version"] == "0.40.0"
    assert meta["flags"]["has_thinking"] and not meta["flags"]["truncated"]
    assert meta["flags"]["system_prompt_rendered"] is True
    rendered = records.read_text(cfg.poems_dir / meta["rendered_prompt"]["snapshot"])
    assert rendered.startswith("<sys>") and hot.system.text in rendered
    assert records.read_text(records.thinking_path(path)).startswith("Let me think")
    # Snapshots: the exact prompt text, and the Modelfile without the local path.
    snapshot = cfg.poems_dir / meta["request"]["messages"][0]["snapshot"]
    assert records.read_text(snapshot) == hot.system.text
    modelfile = records.read_text(cfg.poems_dir / meta["model"]["modelfile"])
    assert "/home/me" not in modelfile and WEIGHTS in modelfile
    assert client.unloaded == ["small:q8"]
    # The request sent to Ollama carried every option and the full prompts.
    sent = client.requests[0]
    assert sent["messages"][0]["content"].startswith(("Oṃ", "Prompt B"))
    assert set(cfg.defaults) <= set(sent["options"])


def test_resume_without_duplicates(cfg):
    run(cfg, FakeOllama(), ["small"])
    client = FakeOllama()
    _, summary = run(cfg, client, ["small"])
    assert summary.written == 0 and client.requests == []


def test_failures_are_logged_and_retried(cfg):
    plan, summary = run(cfg, FakeOllama(fail_times=1), ["small"])
    assert summary.failed == 1 and summary.written == len(plan.specs) - 1
    lines = (cfg.poems_dir / "exp" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["error"] == "boom"
    _, again = run(cfg, FakeOllama(), ["small"])
    assert again.written == 1


def test_stops_model_after_three_failures_in_a_row(cfg):
    client = FakeOllama(fail_times=100)
    _, summary = run(cfg, client, ["small"])
    assert summary.failed == 3 and len(client.requests) == 3


def test_refuses_mismatched_build(cfg, repo):
    path = repo / "config" / "models.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace('ollama_digest = "d1"', 'ollama_digest = "zz"'),
        encoding="utf-8",
    )
    client = FakeOllama()
    _, summary = run(load_config(repo), client, ["small"])
    assert summary.skipped_models == ["small"] and client.requests == []


def test_flags(cfg):
    _, summary = run(cfg, FakeOllama(content="<think>hmm</think> A poem"), ["plain"])
    meta, _ = records.load_poem(next(records.poem_files(cfg.poems_dir, "exp")))
    assert meta["flags"]["thinking_in_content"] and not meta["flags"]["has_thinking"]
    assert meta["request"]["options"]["num_ctx"] == 8192


def test_parse_stop_and_redact():
    assert parse_stop('stop "<|im_end|>"\ntemperature 0.7\nstop "<|im_start|>"') == [
        "<|im_end|>",
        "<|im_start|>",
    ]
    assert parse_stop(None) == []
    text, weights = redact_modelfile(f"FROM C:\\Users\\me\\.ollama\\blobs\\sha256-{WEIGHTS}\n")
    assert weights == f"sha256:{WEIGHTS}" and "Users" not in text
