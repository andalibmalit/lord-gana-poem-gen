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
    assert exp is not None
    runs = {r.data.tags["poem_id"]: r for r in client.search_runs([exp.experiment_id])}
    run = runs["bbbbbbbbbbbb"]
    assert run.data.tags["note_tags"] == "no-presence"
    assert run.data.params["opt.temperature"] == "0.7" and run.data.params["model"] == "small"
    assert run.data.metrics["tokens_per_second"] == 20.0
    assert client.search_runs([other]) == []


def test_mlflow_sync_clears_deleted_notes_and_replaces_half_synced_runs(poems, tmp_path):
    from mlflow import MlflowClient

    from poemgen.mlflow_sync import sync
    from poemgen.reading import save_note

    uri = f"sqlite:///{tmp_path / 'test.db'}"
    save_note(poems, "exp", "aaaaaaaaaaaa", "luminous", ["presence"])
    sync(poems, tracking_uri=uri)
    save_note(poems, "exp", "aaaaaaaaaaaa", "", [])
    assert sync(poems, tracking_uri=uri)["updated"] == 1

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("poems/exp")
    assert exp is not None
    exp_id = exp.experiment_id

    def run_for(pid):
        return [r for r in client.search_runs([exp_id]) if r.data.tags.get("poem_id") == pid]

    assert run_for("aaaaaaaaaaaa")[0].data.tags["note"] == ""
    # Simulate a sync that stopped after creating a run: it gets replaced, not duplicated.
    client.delete_run(run_for("bbbbbbbbbbbb")[0].info.run_id)
    client.create_run(exp_id, run_name="bbbbbbbbbbbb", tags={"poem_id": "bbbbbbbbbbbb"})
    assert sync(poems, tracking_uri=uri)["created"] == 1
    runs = run_for("bbbbbbbbbbbb")
    assert len(runs) == 1 and runs[0].info.status == "FINISHED"


def test_text_counts():
    exact = records.text_counts("One two three", "", 12)
    assert exact == {
        "poem_words": 3,
        "poem_chars": 13,
        "thinking_words": 0,
        "thinking_chars": 0,
        "thinking_tokens": 0,
        "poem_tokens": 12,
        "token_split": "exact (no thinking)",
    }
    split = records.text_counts("a" * 30, "b" * 10, 100)
    assert (split["poem_tokens"], split["thinking_tokens"]) == (75, 25)
    assert "poem_tokens" not in records.text_counts("x", "", None)


def test_notes_record_mode_and_keep_the_blind_version(poems):
    from poemgen.reading import save_note

    save_note(poems, "exp", "aaaaaaaaaaaa", "first read", ["presence"], "blind", 7)
    save_note(poems, "exp", "aaaaaaaaaaaa", "after reveal", ["presence", "gemma"], "reveal", 8)
    entry = load_notes(poems, "exp")["aaaaaaaaaaaa"]
    assert entry["mode"] == "reveal" and entry["note"] == "after reveal"
    assert [v["note"] for v in entry["blind_versions"]] == ["first read"]
    assert entry["blind_versions"][0]["shuffle_seed"] == 7
    save_note(poems, "exp", "aaaaaaaaaaaa", "again", [], "reveal", 8)  # reveal over reveal
    save_note(poems, "exp", "aaaaaaaaaaaa", "re-read blind", [], "blind", 9)
    save_note(poems, "exp", "aaaaaaaaaaaa", "", [], "reveal", 10)  # cleared after revealing
    versions = load_notes(poems, "exp")["aaaaaaaaaaaa"]["blind_versions"]
    assert [v["note"] for v in versions] == ["first read", "re-read blind"]
    assert set(load_notes(poems, "exp")["aaaaaaaaaaaa"]) == {"blind_versions"}
