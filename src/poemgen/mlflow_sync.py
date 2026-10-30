"""Mirror poem files and reading notes into MLflow.

MLflow is only a viewer: everything here is rebuilt from poems/ and notes/. Each poem
experiment becomes an MLflow experiment named `poems/<name>`, and each poem one run,
found again by its `poem_id` tag. Other MLflow experiments are never touched.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

from poemgen import records
from poemgen.config import Config
from poemgen.reading import load_notes

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
MAX_TAG = 5000


def backup_db(cfg: Config, tracking_uri: str) -> Path | None:
    """Copy a local sqlite tracking db to backups/ once a day, before writing to it."""
    if not tracking_uri.startswith("sqlite:///"):
        return None
    db = Path(tracking_uri.removeprefix("sqlite:///"))
    db = db if db.is_absolute() else cfg.root / db
    if not db.is_file():
        return None
    target = cfg.root / "backups" / f"{db.stem}-{date.today().isoformat()}{db.suffix}"
    if not target.exists():
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(db, target)
        return target
    return None


def _ms(iso: str | None) -> int | None:
    return int(datetime.fromisoformat(iso).timestamp() * 1000) if iso else None


def run_fields(meta: dict, note: dict | None) -> tuple[dict, dict, dict]:
    """(params, tags, metrics) for one poem."""
    request, response = meta.get("request", {}), meta.get("response", {})
    model = meta.get("model", {})
    params = {
        "model": model.get("key"),
        "model_tag": model.get("tag"),
        "quant": model.get("quant"),
        "settings_set": meta.get("settings_set"),
        "sample": meta.get("sample"),
        "seed": meta.get("seed"),
        "think": request.get("think"),
        "system_prompt": request.get("messages", [{}])[0].get("variant"),
        "user_prompt": request.get("messages", [{}, {}])[-1].get("variant"),
    }
    for name, value in request.get("options", {}).items():
        params[f"opt.{name}"] = json.dumps(value, ensure_ascii=False) if name == "stop" else value
    tags = {
        "poem_id": meta["poem_id"],
        "model_digest": model.get("digest"),
        "weights_sha256": model.get("weights_sha256"),
        "ollama_version": meta.get("ollama_version"),
        "git_commit": (meta.get("tool") or {}).get("git_commit"),
        "done_reason": response.get("done_reason"),
    }
    tags |= {f"flag.{k}": v for k, v in (meta.get("flags") or {}).items()}
    tags |= {f"origin.{k}": v for k, v in (meta.get("origins") or {}).items()}
    tags |= {f"not_applied.{k}": v for k, v in (meta.get("not_applied") or {}).items()}
    if note:
        tags["note"] = note.get("note", "")[:MAX_TAG]
        tags["note_tags"] = ", ".join(note.get("tags", []))
    memory = meta.get("memory") or {}
    metrics = {
        "prompt_tokens": response.get("prompt_eval_count"),
        "output_tokens": response.get("eval_count"),
        "total_seconds": (response.get("total_duration") or 0) / 1e9 or None,
        "load_seconds": (response.get("load_duration") or 0) / 1e9 or None,
        "memory_gb": memory.get("size", 0) / 1e9 or None,
        "memory_on_gpu_gb": memory.get("size_vram", 0) / 1e9 or None,
    }
    if response.get("eval_count") and response.get("eval_duration"):
        metrics["tokens_per_second"] = response["eval_count"] / (response["eval_duration"] / 1e9)
    clean = lambda d: {k: str(v) for k, v in d.items() if v is not None}  # noqa: E731
    return clean(params), clean(tags), {k: float(v) for k, v in metrics.items() if v is not None}


def sync(cfg: Config, experiments: list[str] | None = None, tracking_uri: str | None = None):
    """Create runs for new poems and refresh note tags on existing ones. Returns counts."""
    from mlflow import MlflowClient  # imported here: slow, and only this command needs it
    from mlflow.entities import Metric, Param, RunTag

    tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
    backup = backup_db(cfg, tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    names = experiments or sorted(
        d.name for d in cfg.poems_dir.glob("*") if d.is_dir() and not d.name.startswith("_")
    )
    counts = {"created": 0, "updated": 0, "unchanged": 0, "backup": str(backup) if backup else None}
    for name in names:
        exp = client.get_experiment_by_name(f"poems/{name}")
        exp_id = (
            exp.experiment_id
            if exp
            else client.create_experiment(
                f"poems/{name}",
                artifact_location=(cfg.root / "mlartifacts" / f"poems-{name}").resolve().as_uri(),
            )
        )
        existing = {
            r.data.tags.get("poem_id"): r
            for r in client.search_runs([exp_id], max_results=50000)
            if r.data.tags.get("poem_id")
        }
        notes = load_notes(cfg, name)
        for path in records.poem_files(cfg.poems_dir, name):
            meta, _ = records.load_poem(path)
            params, tags, metrics = run_fields(meta, notes.get(meta["poem_id"]))
            run = existing.get(meta["poem_id"])
            if run is not None:
                changed = {k: v for k, v in tags.items() if run.data.tags.get(k) != v}
                for key, value in changed.items():
                    client.set_tag(run.info.run_id, key, value)
                counts["updated" if changed else "unchanged"] += 1
                continue
            start = _ms(meta.get("created"))
            run = client.create_run(exp_id, start_time=start, run_name=meta["poem_id"])
            now = int(datetime.now(UTC).timestamp() * 1000)
            client.log_batch(
                run.info.run_id,
                metrics=[Metric(k, v, now, 0) for k, v in metrics.items()],
                params=[Param(k, v) for k, v in params.items()],
                tags=[RunTag(k, v) for k, v in tags.items()],
            )
            client.log_artifact(run.info.run_id, str(path))
            thinking = records.thinking_path(path)
            if thinking.exists():
                client.log_artifact(run.info.run_id, str(thinking))
            end = (
                start + int((meta.get("response") or {}).get("total_duration", 0) / 1e6)
                if start
                else None
            )
            client.set_terminated(run.info.run_id, "FINISHED", end_time=end)
            counts["created"] += 1
    return counts
