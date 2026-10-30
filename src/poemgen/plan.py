"""Expand an experiment into planned poems: settings, ids, seeds, resume state and estimates.

Everything here comes from config and the poem files, so planning works with Ollama off.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from poemgen import records
from poemgen.config import Config, Experiment, Model, SettingsSet


@dataclass(frozen=True)
class Prompt:
    variant: str
    text: str
    sha256: str


@dataclass(frozen=True)
class PoemSpec:
    poem_id: str
    experiment: str
    model: Model
    system: Prompt
    user: Prompt
    settings_set: str
    sample: int
    think: bool | str
    options: dict  # every option sent, seed included
    origins: dict  # option -> where its value came from
    not_applied: dict

    @property
    def thinking_on(self) -> bool:
        return self.think is not False


@dataclass
class Plan:
    experiment: Experiment
    specs: list[PoemSpec]
    done: set[str]
    orphans: list[str]  # poem files in the folder that this config no longer plans
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def pending(self) -> list[PoemSpec]:
        return [s for s in self.specs if s.poem_id not in self.done]


def resolve_think(model: Model, wanted: bool | str) -> bool | str | None:
    """The `think` value to send, or None if this model can't do what the settings set asks."""
    if not model.think_values:  # no thinking switch: always send false
        return False if wanted in ("default", False) else None
    value = model.think_default if wanted == "default" else wanted
    return value if value in model.think_values else None


def resolve_options(
    cfg: Config, model: Model, sset: SettingsSet, thinking_on: bool
) -> tuple[dict, dict, list[str]]:
    """Layer the settings for one model and settings set. Returns (options, origins, warnings)."""
    options: dict = {}
    origins: dict = {}

    def apply(values: dict, origin: str) -> None:
        for name, value in values.items():
            options[name] = value
            origins[name] = origin

    apply(cfg.defaults, "default")
    if model.stop:
        apply({"stop": list(model.stop)}, "model")
    # The maker's own recommendation beats an inherited one, whatever the thinking mode.
    inherited = f"inherited:{model.inherited.source}"
    apply(model.inherited.values, inherited)
    if thinking_on:
        apply(model.inherited.thinking, inherited)
    apply(model.recommended.values, "recommended")
    if thinking_on:
        apply(model.recommended.thinking, "recommended:thinking")
    apply(sset.all, f"change:{sset.name}")
    apply(sset.models.get(model.key, {}), f"change:{sset.name}")

    warnings = []
    for by, cap in (
        ("machine", cfg.machine.num_ctx_caps.get(model.key)),
        ("ctx_max", model.ctx_max),
    ):
        if cap and options["num_ctx"] > cap:
            warnings.append(f"{model.key}: num_ctx {options['num_ctx']} capped to {cap} ({by}).")
            options["num_ctx"] = cap
            origins["num_ctx"] = f"capped:{by}"
    # Prompt + output must fit the context: past it, Ollama's llama-server shifts the context
    # and silently drops its middle (the system prompt included).
    room = options["num_ctx"] - cfg.prompt_reserve
    if room < 1024:
        raise ValueError(
            f"{model.key}: num_ctx {options['num_ctx']} leaves under 1024 tokens of output after "
            f"the {cfg.prompt_reserve}-token prompt reserve (defaults.toml [limits])."
        )
    if options["num_predict"] > room:
        options["num_predict"] = room
        origins["num_predict"] = "capped:num_ctx"
    return options, origins, warnings


def poem_id(spec_fields: dict) -> str:
    canonical = json.dumps(spec_fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def load_prompt(path: Path, variant: str) -> Prompt:
    text = records.read_text(path)
    return Prompt(variant, text, records.sha256_text(text))


def build_plan(cfg: Config, exp: Experiment, only_models: list[str] | None = None) -> Plan:
    models = list(exp.models)
    if only_models:
        unknown = [m for m in only_models if m not in exp.models]
        if unknown:
            raise ValueError(f"--models names models not in {exp.name}: {unknown}")
        models = [m for m in models if m in only_models]

    user = load_prompt(cfg.user_prompt_path(exp.user_prompt), exp.user_prompt)
    systems = [load_prompt(cfg.system_prompt_path(v), v) for v in exp.system_prompts]
    specs: dict[str, PoemSpec] = {}
    skipped, warnings = [], []

    for key in models:
        model = cfg.models[key]
        if not model.confirmed:
            warnings.append(f"{key}: build not confirmed yet; `run` will refuse it.")
        for sset in exp.settings:
            think = resolve_think(model, sset.think)
            if think is None:
                skipped.append(f"{key} × {sset.name}: model doesn't accept think={sset.think!r}.")
                continue
            options, origins, cap_warnings = resolve_options(cfg, model, sset, think is not False)
            warnings += [w for w in cap_warnings if w not in warnings]
            for system in systems:
                for sample in range(exp.samples):
                    seed = exp.seed + sample
                    sent = {**options, "seed": seed}
                    pid = poem_id(
                        {
                            "model": key,
                            "weights_sha256": model.weights_sha256,
                            "ollama_digest": model.ollama_digest,
                            "system_sha256": system.sha256,
                            "user_sha256": user.sha256,
                            "options": sent,
                            "think": think,
                            "sample": sample,
                        }
                    )
                    if pid in specs:  # two settings sets resolved to the same request
                        continue
                    specs[pid] = PoemSpec(
                        poem_id=pid,
                        experiment=exp.name,
                        model=model,
                        system=system,
                        user=user,
                        settings_set=sset.name,
                        sample=sample,
                        think=think,
                        options=sent,
                        origins={**origins, "seed": "experiment"},
                        not_applied=dict(model.unsupported),
                    )

    existing = {p.stem for p in records.poem_files(cfg.poems_dir, exp.name)}
    return Plan(
        experiment=exp,
        specs=list(specs.values()),
        done=existing & specs.keys(),
        orphans=sorted(existing - specs.keys()),
        skipped=skipped,
        warnings=warnings,
    )


def minutes_per_poem(cfg: Config) -> dict[tuple[str, bool], float]:
    """Median minutes per poem from earlier poems, keyed by (model key, thinking on)."""
    durations = defaultdict(list)
    if not cfg.poems_dir.is_dir():
        return {}
    for path in records.poem_files(cfg.poems_dir):
        try:
            meta, _ = records.load_poem(path)
            key = (meta["model"]["key"], meta["request"]["think"] is not False)
            durations[key].append(meta["response"]["total_duration"] / 60e9)
        except (KeyError, TypeError, ValueError):
            continue
    return {k: statistics.median(v) for k, v in durations.items()}


def estimate_minutes(cfg: Config, specs: list[PoemSpec]) -> dict[str, float]:
    history = minutes_per_poem(cfg)
    totals: dict[str, float] = defaultdict(float)
    for s in specs:
        totals[s.model.key] += history.get(
            (s.model.key, s.thinking_on), cfg.machine.minutes_per_poem
        )
    return dict(totals)
