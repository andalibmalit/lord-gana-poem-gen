"""Registry lookups against sample manifests, and the poemgen command line."""

from __future__ import annotations

import contextlib
import json

import httpx
import pytest

from poemgen import cli
from poemgen.registry import Build, NotInRegistry, fetch_build, registry_base

TEMPLATE = "{{ .System }}{{ .Prompt }}"
PARAMS = {"stop": ["<|im_end|>"], "temperature": 0.6}
MANIFEST = {
    "config": {"size": 500},
    "layers": [
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:w",
            "size": 9_000_000_000,
        },
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:t", "size": 30},
        {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:p", "size": 40},
    ],
}


def mock_client(requested: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        path = request.url.path
        if "missing" in path:
            return httpx.Response(404)
        if path.endswith("/manifests/q8"):
            return httpx.Response(200, json=MANIFEST)
        if path.endswith("/blobs/sha256:t"):
            return httpx.Response(200, text=TEMPLATE)
        if path.endswith("/blobs/sha256:p"):
            return httpx.Response(200, text=json.dumps(PARAMS))
        return httpx.Response(500)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    ("tag", "base", "ref"),
    [
        ("gemma4:e4b-it-q8_0", "https://registry.ollama.ai/v2/library/gemma4", "e4b-it-q8_0"),
        ("muse-glimmer", "https://registry.ollama.ai/v2/library/muse-glimmer", "latest"),
        ("someone/model:7b", "https://registry.ollama.ai/v2/someone/model", "7b"),
        ("hf.co/org/Repo-GGUF:Q4_K_M", "https://huggingface.co/v2/org/Repo-GGUF", "Q4_K_M"),
        ("hf.co/org/Repo-GGUF:latest", "https://huggingface.co/v2/org/Repo-GGUF", "latest"),
        ("huggingface.co/org/Repo-GGUF", "https://huggingface.co/v2/org/Repo-GGUF", "latest"),
    ],
)
def test_registry_base(tag, base, ref):
    assert registry_base(tag) == (base, ref)


def test_fetch_build_reads_manifest_and_small_blobs_only():
    requested: list[str] = []
    build = fetch_build("hf.co/org/Repo-GGUF:q8", mock_client(requested))
    assert build.weights_sha256 == "sha256:w"
    assert build.weights_bytes == 9_000_000_000 and build.total_bytes == 9_000_000_570
    assert build.template == TEMPLATE and build.stop == ["<|im_end|>"]
    assert not any("sha256:w" in url for url in requested)  # weights never downloaded


def test_fetch_build_missing_tag():
    with pytest.raises(NotInRegistry, match="not found"):
        fetch_build("org/missing:q8", mock_client([]))


# --- command line -----------------------------------------------------------------------


@pytest.fixture
def in_repo(repo, monkeypatch):
    monkeypatch.chdir(repo)
    return repo


def test_plan_command(in_repo, capsys):
    assert cli.main(["plan", "exp", "--models", "small"]) == 0
    out = capsys.readouterr().out
    assert "Experiment exp: 8 poems planned, 0 done, 8 to run" in out
    assert "(Ollama not reachable)" in out  # the test machine config points at a closed port


def test_errors_are_messages_not_tracebacks(in_repo, capsys):
    assert cli.main(["plan", "nope"]) == 2
    assert "error: Missing config file" in capsys.readouterr().err
    assert cli.main(["run", "exp"]) == 2  # Ollama unreachable
    assert "error:" in capsys.readouterr().err


def test_models_check_reports_mismatches(in_repo, monkeypatch, capsys):
    build = Build("small:q8", "url", 10, 9, "sha256:other", [], None, {"stop": ["<end>"]})
    monkeypatch.setattr(cli, "fetch_build", lambda tag, client=None: build)
    monkeypatch.setattr(cli.httpx, "Client", lambda **kw: contextlib.nullcontext())
    assert cli.main(["models", "check", "small"]) == 1
    assert "MISMATCH: config pins sha256:aaa" in capsys.readouterr().out


def test_run_command_with_a_fake_ollama(in_repo, monkeypatch, capsys):
    from test_run import WEIGHTS, FakeOllama

    path = in_repo / "config" / "models.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace('"sha256:aaa"', f'"sha256:{WEIGHTS}"'),
        encoding="utf-8",
    )
    fake = FakeOllama()
    monkeypatch.setattr(cli, "Ollama", lambda *a, **k: fake | _closable)
    assert cli.main(["run", "exp", "--models", "small"]) == 0
    assert "Done: 8 written, 0 failed." in capsys.readouterr().out
    assert len(list((in_repo / "poems" / "exp").glob("*.md"))) == 8 + 4  # poems + thinking files


class _Closable:
    """Lets `fake | _closable` add a close() method to a FakeOllama instance."""

    def __ror__(self, other):
        other.close = lambda: None
        return other


_closable = _Closable()
