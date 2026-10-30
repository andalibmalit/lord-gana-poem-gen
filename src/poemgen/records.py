"""Poem files: the committed record.

Each poem is `poems/<experiment>/<poem_id>.md`: YAML front matter with everything
needed to trace it, then the poem text exactly as the model returned it. Thinking
text, when there is any, goes in `<poem_id>.thinking.md` next to it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import yaml

FRONT = "---\n"
_failure_lock = threading.Lock()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    """Read UTF-8 text verbatim (no newline translation)."""
    with path.open(encoding="utf-8", newline="") as f:
        return f.read()


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text verbatim via a temp file, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
    try:
        os.replace(tmp, path)
    except PermissionError:  # Windows: the target can be briefly locked by a reader
        time.sleep(0.2)
        os.replace(tmp, path)


def text_counts(poem: str, thinking: str, output_tokens: int | None) -> dict:
    """Words, characters and an output-token split for the poem and its thinking.

    Ollama reports one output count for thinking and poem together. With no thinking,
    the poem's share is exact; otherwise it's estimated by each part's share of characters.
    """
    counts = {
        "poem_words": len(poem.split()),
        "poem_chars": len(poem),
        "thinking_words": len(thinking.split()),
        "thinking_chars": len(thinking),
    }
    if output_tokens is not None:
        total = len(poem) + len(thinking)
        thinking_share = len(thinking) / total if total else 0.0
        counts["thinking_tokens"] = round(output_tokens * thinking_share)
        counts["poem_tokens"] = output_tokens - counts["thinking_tokens"]
        counts["token_split"] = "estimated by characters" if thinking else "exact (no thinking)"
    return counts


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def poem_path(poems_dir: Path, experiment: str, poem_id: str) -> Path:
    return poems_dir / experiment / f"{poem_id}.md"


def thinking_path(poem_file: Path) -> Path:
    return poem_file.with_name(poem_file.stem + ".thinking.md")


class _Dumper(yaml.SafeDumper):
    """SafeDumper that escapes the characters YAML reads as line breaks."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    # PyYAML writes NEL and the Unicode line/paragraph separators unescaped, and reads them
    # back as line breaks; double quotes make it escape them.
    style = '"' if any(c in data for c in "\x85\u2028\u2029") else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _represent_str)


def dump(meta: dict, body: str) -> str:
    front = yaml.dump(meta, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)
    return f"{FRONT}{front}{FRONT}\n{body}"


def parse(text: str) -> tuple[dict, str]:
    if not text.startswith(FRONT):
        raise ValueError("not a poem record: no front matter")
    end = text.index("\n" + FRONT, len(FRONT) - 1)
    meta = yaml.safe_load(text[len(FRONT) : end + 1]) or {}
    body = text[end + 1 + len(FRONT) :]
    return meta, body.removeprefix("\n")


def write_poem(path: Path, meta: dict, poem: str, thinking: str | None) -> None:
    """Thinking first, poem last: the poem file's existence marks the poem as done."""
    if thinking:
        write_text_atomic(thinking_path(path), thinking)
    write_text_atomic(path, dump(meta, poem))


def load_poem(path: Path) -> tuple[dict, str]:
    return parse(read_text(path))


def is_poem_file(path: Path) -> bool:
    return (
        path.suffix == ".md"
        and not path.name.endswith(".thinking.md")
        and not path.name.startswith(".")
    )


def poem_files(poems_dir: Path, experiment: str | None = None) -> Iterator[Path]:
    """Poem files of one experiment, or of all experiments (skipping `_` folders)."""
    dirs = (
        [poems_dir / experiment]
        if experiment
        else sorted(d for d in poems_dir.glob("*") if d.is_dir() and not d.name.startswith("_"))
    )
    for d in dirs:
        yield from sorted(p for p in d.glob("*.md") if is_poem_file(p))


def snapshot(poems_dir: Path, kind: str, name: str, text: str) -> str:
    """Store `text` once under poems/_<kind>/<name>; return the path relative to poems/."""
    rel = Path(f"_{kind}") / name
    path = poems_dir / rel
    if not path.exists():
        write_text_atomic(path, text)
    return rel.as_posix()


def log_failure(poems_dir: Path, experiment: str, entry: dict) -> None:
    line = json.dumps({"time": now_utc(), **entry}, ensure_ascii=False)
    path = poems_dir / experiment / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _failure_lock, path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
