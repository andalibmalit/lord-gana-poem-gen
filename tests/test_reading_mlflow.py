"""Blind reading (page + notes server) and the MLflow mirror."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from poemgen import records
from poemgen.reading import build_page, load_notes, make_server

META = {
    "model": {"key": "small", "name": "Small Model", "tag": "small:q8", "digest": "d1"},
    "settings_set": "base",
    "sample": 0,
    "seed": 100,
    "created": "2026-10-30T18:00:00+00:00",
    "origins": {"temperature": "recommended"},
    "request": {
        "think": False,
        "options": {"temperature": 0.7, "stop": ["<end>"]},
        "messages": [{"variant": "a"}, {"variant": "u"}],
    },
    "response": {
        "done_reason": "stop",
        "eval_count": 40,
        "eval_duration": 2e9,
        "total_duration": 5e9,
    },
    "flags": {"truncated": False},
}


@pytest.fixture
def poems(cfg):
    for pid, text in (("aaaaaaaaaaaa", "Oṃ Gaṃ, first poem."), ("bbbbbbbbbbbb", "Second poem.")):
        records.write_poem(
            cfg.poems_dir / "exp" / f"{pid}.md", {**META, "poem_id": pid}, text, None
        )
    return cfg


def page_data(html: str) -> dict:
    start = html.index("const DATA = ") + len("const DATA = ")
    return json.loads(html[start : html.index(";\nconst $")].replace("<\\/", "</"))


def test_blind_page_hides_model_and_settings(poems):
    html = build_page(poems, "exp", reveal=False, shuffle_seed=1)
    data = page_data(html)
    assert {p["id"] for p in data["poems"]} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert all(set(p) == {"id", "poem"} for p in data["poems"])
    assert "Small Model" not in html and "small:q8" not in html
    assert "Oṃ Gaṃ" in html


def test_reveal_page_shows_details(poems):
    data = page_data(build_page(poems, "exp", reveal=True, shuffle_seed=1))
    details = data["poems"][0]["details"]
    assert details["model"] == "Small Model"
    assert details["options"]["temperature"] == [0.7, "recommended"]


def test_shuffle_is_reproducible(poems):
    def order(seed):
        return [p["id"] for p in page_data(build_page(poems, "exp", False, seed))["poems"]]

    assert order(3) == order(3)


@pytest.fixture
def server(poems):
    srv = make_server(poems, "exp", reveal=False, port=0, shuffle_seed=1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def post(url: str, body: dict, origin: str | None = None):
    req = urllib.request.Request(
        url + "/notes",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **({"Origin": origin} if origin else {})},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def test_server_saves_notes(server, poems):
    _, url = server
    with urllib.request.urlopen(url + "/", timeout=5) as r:
        assert "charset=utf-8" in r.headers["Content-Type"]
    saved = post(
        url, {"id": "aaaaaaaaaaaa", "note": "Śrī: has presence", "tags": ["presence", " "]}
    )
    assert saved["tags"] == ["presence"]
    assert load_notes(poems, "exp")["aaaaaaaaaaaa"]["note"] == "Śrī: has presence"
    post(url, {"id": "aaaaaaaaaaaa", "note": "", "tags": []})  # clearing removes the entry
    assert "aaaaaaaaaaaa" not in load_notes(poems, "exp")


def test_server_rejects_unknown_poems_and_other_origins(server):
    _, url = server
    with pytest.raises(urllib.error.HTTPError, match="400"):
        post(url, {"id": "zzzzzzzzzzzz", "note": "x", "tags": []})
    with pytest.raises(urllib.error.HTTPError, match="403"):
        post(url, {"id": "aaaaaaaaaaaa", "note": "x", "tags": []}, origin="http://evil.example")


def test_mlflow_sync_is_idempotent_and_updates_notes(poems, tmp_path):
    from mlflow import MlflowClient

    from poemgen.mlflow_sync import sync
    from poemgen.reading import save_note

    uri = f"sqlite:///{tmp_path / 'test.db'}"
    other = MlflowClient(tracking_uri=uri).create_experiment("unrelated")
    assert sync(poems, tracking_uri=uri)["created"] == 2
    assert sync(poems, tracking_uri=uri)["unchanged"] == 2
    save_note(poems, "exp", "bbbbbbbbbbbb", "flat", ["no-presence"])
    counts = sync(poems, tracking_uri=uri)
    assert counts["updated"] == 1 and counts["created"] == 0

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("poems/exp")
    runs = {r.data.tags["poem_id"]: r for r in client.search_runs([exp.experiment_id])}
    run = runs["bbbbbbbbbbbb"]
    assert run.data.tags["note_tags"] == "no-presence"
    assert run.data.params["opt.temperature"] == "0.7" and run.data.params["model"] == "small"
    assert run.data.metrics["tokens_per_second"] == 20.0
    assert client.search_runs([other]) == []
