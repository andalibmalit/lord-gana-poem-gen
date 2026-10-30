"""Blind reading: a local page that shows poems without their model or settings.

`poemgen read <experiment>` serves one HTML page from 127.0.0.1 and saves notes and tags
to notes/<experiment>.json (committed, keyed by poem_id). In blind mode the page holds
only poem ids and poem text, in shuffled order, so even its source reveals nothing.
`--reveal` adds the model, settings and thinking next to your notes.
"""

from __future__ import annotations

import json
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from poemgen import records
from poemgen.config import Config

_notes_lock = threading.Lock()
MAX_NOTE_CHARS = 20_000


def notes_path(cfg: Config, experiment: str) -> Path:
    return cfg.notes_dir / f"{experiment}.json"


def load_notes(cfg: Config, experiment: str) -> dict[str, dict]:
    path = notes_path(cfg, experiment)
    return json.loads(records.read_text(path)) if path.is_file() else {}


def save_note(
    cfg: Config,
    experiment: str,
    poem_id: str,
    note: str,
    tags: list[str],
    mode: str = "blind",
    shuffle_seed: int | None = None,
) -> dict:
    """Save a note, recording whether it was written blind or after revealing.

    When a blind note is first edited in reveal mode, the blind version is kept under
    "blind", so the blind judgment is never lost.
    """
    tags = sorted({t.strip() for t in tags if t.strip()})
    with _notes_lock:
        notes = load_notes(cfg, experiment)
        previous = notes.get(poem_id, {})
        blind = previous.get("blind")
        if blind is None and mode == "reveal" and previous.get("mode") == "blind":
            blind = {k: previous[k] for k in ("note", "tags", "updated", "shuffle_seed")}
        entry: dict = {}
        if note.strip() or tags:
            entry = {
                "note": note,
                "tags": tags,
                "updated": records.now_utc(),
                "mode": mode,
                "shuffle_seed": shuffle_seed,
            }
        if blind:
            entry["blind"] = blind
        if entry:
            notes[poem_id] = entry
        else:
            notes.pop(poem_id, None)
        text = json.dumps(dict(sorted(notes.items())), ensure_ascii=False, indent=2) + "\n"
        records.write_text_atomic(notes_path(cfg, experiment), text)
    return notes.get(poem_id, {})


def poem_entries(cfg: Config, experiment: str, reveal: bool) -> list[dict]:
    entries = []
    for path in records.poem_files(cfg.poems_dir, experiment):
        meta, poem = records.load_poem(path)
        entry = {"id": meta["poem_id"], "poem": poem}
        if reveal:
            thinking = records.thinking_path(path)
            request = meta.get("request", {})
            entry["details"] = {
                "model": meta.get("model", {}).get("name") or meta.get("model", {}).get("key"),
                "model_key": meta.get("model", {}).get("key"),
                "tag": meta.get("model", {}).get("tag"),
                "settings_set": meta.get("settings_set"),
                "system_prompt": request.get("messages", [{}])[0].get("variant"),
                "sample": meta.get("sample"),
                "think": request.get("think"),
                "options": {
                    k: [v, (meta.get("origins") or {}).get(k)]
                    for k, v in request.get("options", {}).items()
                },
                "flags": {k: v for k, v in (meta.get("flags") or {}).items() if v},
                "not_applied": meta.get("not_applied") or {},
                "counts": records.text_counts(
                    poem,
                    records.read_text(thinking) if thinking.exists() else "",
                    meta.get("response", {}).get("eval_count"),
                ),
                "thinking": records.read_text(thinking) if thinking.exists() else None,
            }
        entries.append(entry)
    return entries


def build_page(cfg: Config, experiment: str, reveal: bool, shuffle_seed: int) -> str:
    entries = poem_entries(cfg, experiment, reveal)
    random.Random(shuffle_seed).shuffle(entries)
    data = {
        "experiment": experiment,
        "reveal": reveal,
        "poems": entries,
        "notes": load_notes(cfg, experiment),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    template = resources.files("poemgen").joinpath("reading_page.html").read_text(encoding="utf-8")
    return template.replace("/*DATA*/null", payload)


def make_server(
    cfg: Config, experiment: str, reveal: bool, port: int, shuffle_seed: int
) -> ThreadingHTTPServer:
    known = {p.stem for p in records.poem_files(cfg.poems_dir, experiment)}
    if not known:
        raise ValueError(f"no poems in poems/{experiment}/")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _send(self, status: int, body: str, content_type: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                return self._send(404, "not found", "text/plain")
            self._send(200, build_page(cfg, experiment, reveal, shuffle_seed), "text/html")

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            if origin and origin not in allowed_origins:
                return self._send(403, '{"error": "bad origin"}', "application/json")
            if self.path != "/notes":
                return self._send(404, '{"error": "not found"}', "application/json")
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                poem_id, note, tags = body["id"], str(body.get("note", "")), body.get("tags", [])
                if poem_id not in known or len(note) > MAX_NOTE_CHARS or not isinstance(tags, list):
                    raise ValueError("unknown poem or bad note")
                mode = "reveal" if reveal else "blind"
                saved = save_note(
                    cfg, experiment, poem_id, note, [str(t) for t in tags], mode, shuffle_seed
                )
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._send(400, json.dumps({"error": str(e)}), "application/json")
            self._send(200, json.dumps(saved, ensure_ascii=False), "application/json")

    allowed_origins: set[str] = set()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    bound = server.server_port  # the real port, when port=0 picked a free one
    allowed_origins |= {f"http://127.0.0.1:{bound}", f"http://localhost:{bound}"}
    return server


def serve(
    cfg: Config,
    experiment: str,
    reveal: bool = False,
    port: int = 8765,
    shuffle_seed: int | None = None,
    open_browser: bool = True,
) -> None:
    seed = shuffle_seed if shuffle_seed is not None else random.randrange(10**6)
    server = make_server(cfg, experiment, reveal, port, seed)
    url = f"http://127.0.0.1:{server.server_port}/"
    mode = "reveal" if reveal else "blind"
    print(f"Reading {experiment} ({mode}, shuffle seed {seed}) at {url}")
    print(f"Notes save to {notes_path(cfg, experiment)}. Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
