"""Run a plan: generate each pending poem through Ollama and write its record.

Models run one at a time, in experiment order, and are unloaded afterwards. Each poem
is written as soon as it's done, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import platform
import re
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from poemgen import __version__, records
from poemgen.config import Config, Model
from poemgen.ollama import PROVIDER, Ollama, OllamaError
from poemgen.plan import Plan, PoemSpec, Prompt

MAX_CONSECUTIVE_FAILURES = 3
THINK_MARKERS = re.compile(r"</?think>|<\|channel\|?>|<\|start_thinking\|>", re.IGNORECASE)
FROM_BLOB = re.compile(r"^FROM (.*?)(sha256[-:])([0-9a-f]{64})\s*$", re.MULTILINE)


class Refused(Exception):
    """A model failed its checks before any poem was generated."""


@dataclass
class ModelInfo:
    digest: str
    weights_sha256: str | None
    details: dict
    capabilities: list
    context_length: int | None
    stop: list[str]
    modelfile_snapshot: str


@dataclass
class Summary:
    written: int = 0
    failed: int = 0
    skipped_models: list[str] = field(default_factory=list)


def git_info(root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8"
            )
        except OSError:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def parse_stop(parameters: str | None) -> list[str]:
    """Stop sequences from /api/show `parameters` text (`stop "<|im_end|>"` lines)."""
    stops = []
    for line in (parameters or "").splitlines():
        name, _, value = line.strip().partition(" ")
        if name == "stop":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            stops.append(value)
    return stops


def redact_modelfile(modelfile: str) -> tuple[str, str | None]:
    """Replace the machine-specific blob path in FROM lines; return (text, weights digest)."""
    match = FROM_BLOB.search(modelfile)
    weights = f"sha256:{match.group(3)}" if match else None
    return FROM_BLOB.sub(r"FROM <ollama models>/blobs/sha256-\3", modelfile), weights


def preflight(cfg: Config, client: Ollama, model: Model, local: dict[str, str]) -> ModelInfo:
    """Check that the local build is exactly the configured one. Raises Refused if not."""
    if not model.confirmed:
        raise Refused("build not confirmed in config/models.toml")
    if model.tag not in local:
        raise Refused(f"{model.tag} isn't pulled: run `poemgen models pull {model.key}`")
    if local[model.tag] != model.ollama_digest:
        raise Refused(
            f"local digest {local[model.tag][:12]} differs from ollama_digest "
            f"{(model.ollama_digest or 'unset')[:12]} in config/models.toml"
        )
    show = client.show(model.tag)
    modelfile, weights = redact_modelfile(show.get("modelfile", ""))
    problems = []
    if model.weights_sha256 and weights != model.weights_sha256:
        problems.append(f"weights {weights} differ from weights_sha256 {model.weights_sha256}")
    stop = parse_stop(show.get("parameters"))
    if sorted(stop) != sorted(model.stop):
        problems.append(f"Modelfile stop {stop} differs from config stop {list(model.stop)}")
    context = next(
        (v for k, v in show.get("model_info", {}).items() if k.endswith(".context_length")), None
    )
    if model.ctx_max and context and context != model.ctx_max:
        problems.append(f"GGUF context_length {context} differs from ctx_max {model.ctx_max}")
    capabilities = show.get("capabilities", [])
    if bool(model.think_values) != ("thinking" in capabilities):
        problems.append(
            f"config think_values {list(model.think_values)} disagree with Ollama's "
            f"capabilities {capabilities}"
        )
    if problems:
        raise Refused("; ".join(problems))
    snapshot = records.snapshot(
        cfg.poems_dir, "models", f"{model.key}-{local[model.tag][:12]}.modelfile", modelfile
    )
    return ModelInfo(
        digest=local[model.tag],
        weights_sha256=weights,
        details=show.get("details", {}),
        capabilities=capabilities,
        context_length=context,
        stop=stop,
        modelfile_snapshot=snapshot,
    )


def request_body(spec: PoemSpec, cfg: Config) -> dict:
    return {
        "model": spec.model.tag,
        "messages": [
            {"role": "system", "content": spec.system.text},
            {"role": "user", "content": spec.user.text},
        ],
        "think": spec.think,
        "options": spec.options,
        "stream": False,
        # Fail loudly instead of silently dropping prompt text when the context is too small.
        "truncate": False,
        "shift": False,
        "keep_alive": cfg.machine.keep_alive,
    }


def prompt_ref(cfg: Config, prompt: Prompt) -> dict:
    snapshot = records.snapshot(cfg.poems_dir, "prompts", f"{prompt.sha256[:12]}.md", prompt.text)
    return {"variant": prompt.variant, "sha256": prompt.sha256, "snapshot": snapshot}


def flags(spec: PoemSpec, response: dict, poem: str, thinking: str) -> dict:
    used = response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
    return {
        "truncated": response.get("done_reason") == "length",
        "context_full": used >= spec.options["num_ctx"],
        "empty_poem": not poem.strip(),
        "has_thinking": bool(thinking.strip()),
        "thinking_in_content": bool(THINK_MARKERS.search(poem)),
    }


class Runner:
    def __init__(self, cfg: Config, client: Ollama, log: Callable[[str], None] = print) -> None:
        self.cfg, self.client, self.log = cfg, client, log
        self._lock = threading.Lock()
        self.summary = Summary()
        self._done = 0
        self._total = 0

    def run(self, plan: Plan) -> Summary:
        pending = plan.pending
        if not pending:
            self.log("Nothing to run: every planned poem already exists.")
            return self.summary
        self._total = len(pending)
        self.version = self.client.version()
        self.tool = {"name": "poemgen", "version": __version__, **git_info(self.cfg.root)}
        self.machine = {
            "name": self.cfg.machine.name,
            "os": platform.system(),
            "os_release": platform.release(),
            "arch": platform.machine(),
            "ollama_env_declared": self.cfg.machine.ollama_env,
        }
        local = self.client.tags()
        by_model: dict[str, list[PoemSpec]] = {}
        for spec in pending:
            by_model.setdefault(spec.model.key, []).append(spec)
        for key, specs in by_model.items():
            model = self.cfg.models[key]
            try:
                info = preflight(self.cfg, self.client, model, local)
            except (Refused, OllamaError) as e:
                self.log(f"{key}: skipped, {e}")
                records.log_failure(
                    self.cfg.poems_dir, plan.experiment.name, {"model": key, "error": str(e)}
                )
                self.summary.skipped_models.append(key)
                continue
            self.log(f"{key}: {len(specs)} poem(s)")
            try:
                self._run_model(model, info, specs)
            finally:
                try:
                    self.client.unload(model.tag)
                except OllamaError:
                    pass
        return self.summary

    def _run_model(self, model: Model, info: ModelInfo, specs: list[PoemSpec]) -> None:
        consecutive = 0 if self.generate(specs[0], info, capture_memory=True) else 1
        rest = specs[1:]
        if self.cfg.machine.parallel == 1:
            for spec in rest:
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    break
                consecutive = 0 if self.generate(spec, info) else consecutive + 1
        else:
            pool = ThreadPoolExecutor(self.cfg.machine.parallel)
            try:
                futures = [pool.submit(self.generate, spec, info) for spec in rest]
                for future in as_completed(futures):
                    consecutive = 0 if future.result() else consecutive + 1
                    if consecutive >= MAX_CONSECUTIVE_FAILURES:
                        break
            finally:
                # Drop queued poems; wait for requests already sent, so the model isn't
                # unloaded under them (on Ctrl+C this waits for those requests to finish).
                pool.shutdown(wait=True, cancel_futures=True)
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            self.log(f"{model.key}: stopped after {consecutive} failures in a row.")

    def generate(self, spec: PoemSpec, info: ModelInfo, capture_memory: bool = False) -> bool:
        """Generate one poem and write its record. Returns False if the request failed."""
        cfg = self.cfg
        body = request_body(spec, cfg)
        rendered = self.client.render(body)  # the exact text the model sees, or None
        started = time.monotonic()
        try:
            response = self.client.chat(body)
        except OllamaError as e:
            records.log_failure(
                cfg.poems_dir,
                spec.experiment,
                {"poem_id": spec.poem_id, "model": spec.model.key, "error": str(e)},
            )
            with self._lock:
                self.summary.failed += 1
            self.log(f"  {spec.poem_id}  failed: {e}")
            return False

        memory = None
        if capture_memory:
            try:
                memory = self.client.loaded(spec.model.tag)
            except OllamaError as e:  # never lose a finished poem over a memory reading
                self.log(f"  warning: couldn't read memory use: {e}")

        message = response.get("message", {})
        poem = message.get("content", "")
        thinking = message.get("thinking") or ""
        poem_flags = flags(spec, response, poem, thinking)
        rendered_ref = None
        if rendered is not None:
            sha = records.sha256_text(rendered)
            snapshot = records.snapshot(cfg.poems_dir, "prompts", f"{sha[:12]}.md", rendered)
            rendered_ref = {"sha256": sha, "snapshot": snapshot}
        # Chat templates can add to or drop the system prompt: check it reached the model.
        poem_flags["system_prompt_rendered"] = (
            None if rendered is None else spec.system.text.strip() in rendered
        )
        path = records.poem_path(cfg.poems_dir, spec.experiment, spec.poem_id)
        recorded_body = {
            **body,
            "messages": [
                {"role": "system", **prompt_ref(cfg, spec.system)},
                {"role": "user", **prompt_ref(cfg, spec.user)},
            ],
        }
        meta = {
            "poem_id": spec.poem_id,
            "experiment": spec.experiment,
            "created": records.now_utc(),
            "provider": PROVIDER,
            "model": {
                "key": spec.model.key,
                "name": spec.model.name,
                "tag": spec.model.tag,
                "upstream": spec.model.upstream,
                "quant": spec.model.quant,
                "digest": info.digest,
                "weights_sha256": info.weights_sha256,
                "quantization_level": info.details.get("quantization_level"),
                "parameter_size": info.details.get("parameter_size"),
                "family": info.details.get("family"),
                "context_length": info.context_length,
                "runner": info.details.get("runner"),
                "parent_model": info.details.get("parent_model") or None,
                "modelfile": info.modelfile_snapshot,
            },
            "settings_set": spec.settings_set,
            "sample": spec.sample,
            "seed": spec.options["seed"],
            "origins": spec.origins,
            "not_applied": spec.not_applied,
            "request": recorded_body,
            "rendered_prompt": rendered_ref,
            "response": {k: v for k, v in response.items() if k not in ("message", "logprobs")},
            "flags": poem_flags,
            "counts": records.text_counts(poem, thinking, response.get("eval_count")),
            "thinking_file": records.thinking_path(path).name if thinking else None,
            "ollama_version": self.version,
            "machine": self.machine,
            "tool": self.tool,
        }
        if memory:
            meta["memory"] = {k: memory.get(k) for k in ("size", "size_vram", "context_length")}
            if memory.get("size_vram", 0) < memory.get("size", 0):
                self.log(
                    f"  warning: {spec.model.key} doesn't fit in GPU memory "
                    f"({memory['size_vram'] / 1e9:.1f} of {memory['size'] / 1e9:.1f} GB on GPU)"
                )
        records.write_poem(path, meta, poem, thinking or None)

        with self._lock:
            self.summary.written += 1
            self._done += 1
            done = self._done
        problems = ("truncated", "context_full", "empty_poem", "thinking_in_content")
        notes = [name for name in problems if poem_flags[name]]
        if poem_flags["system_prompt_rendered"] is False:
            notes.append("system prompt not in rendered prompt")
        self.log(
            f"  [{done}/{self._total}] {spec.poem_id}  {response.get('eval_count', '?')} tokens  "
            f"{(time.monotonic() - started) / 60:.1f} min"
            + (f"  ({', '.join(notes)})" if notes else "")
        )
        return True
