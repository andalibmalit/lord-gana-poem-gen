"""Load and validate the TOML config: defaults, models, machine and experiments."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Every Ollama option the tool sends (docs/inference-settings.md, section 6).
OPTION_NAMES = frozenset(
    {
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "repeat_penalty",
        "repeat_last_n",
        "presence_penalty",
        "frequency_penalty",
        "num_ctx",
        "num_predict",
        "seed",
        "stop",
    }
)
INT_OPTIONS = frozenset({"top_k", "repeat_last_n", "num_ctx", "num_predict", "seed"})
# defaults.toml must give every option except these: the seed comes from the
# experiment and stop tokens from the model.
NOT_IN_DEFAULTS = frozenset({"seed", "stop"})
THINK_LEVELS = ("low", "medium", "high", "max")
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ConfigError(Exception):
    """A config file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Layer:
    """A set of option values with an optional thinking-only overlay."""

    values: dict
    thinking: dict
    source: str


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    tag: str
    upstream: str
    quant: str
    confirmed: bool
    weights_sha256: str | None
    ollama_digest: str | None
    ctx_max: int | None
    think_values: tuple  # values Ollama accepts for `think`; empty = no thinking switch
    think_default: bool | str
    stop: tuple[str, ...]
    recommended: Layer
    inherited: Layer
    unsupported: dict[str, str]
    description: str
    build_notes: str


@dataclass(frozen=True)
class Machine:
    name: str
    ollama_host: str
    gpu_budget_gb: float
    parallel: int
    request_timeout_s: float
    keep_alive: str
    minutes_per_poem: float
    num_ctx_caps: dict[str, int]


@dataclass(frozen=True)
class SettingsSet:
    name: str
    think: bool | str  # "default", True, False or a level
    all: dict
    models: dict[str, dict]


@dataclass(frozen=True)
class Experiment:
    name: str
    models: tuple[str, ...]
    system_prompts: tuple[str, ...]
    user_prompt: str
    samples: int
    seed: int
    settings: tuple[SettingsSet, ...]


@dataclass(frozen=True)
class Config:
    root: Path
    defaults: dict
    models: dict[str, Model]
    machine: Machine

    @property
    def poems_dir(self) -> Path:
        return self.root / "poems"

    @property
    def notes_dir(self) -> Path:
        return self.root / "notes"

    def system_prompt_path(self, variant: str) -> Path:
        return self.root / "prompts" / "system" / f"{variant}.md"

    def user_prompt_path(self, variant: str) -> Path:
        return self.root / "prompts" / "user" / f"{variant}.md"


def find_root(start: Path | None = None) -> Path:
    """Return the nearest directory, upwards from `start`, that holds config/models.toml."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "models.toml").is_file():
            return candidate
    raise ConfigError("No config/models.toml found here or in any parent directory.")


def read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"Missing config file: {path}") from None
    except OSError as e:
        raise ConfigError(f"Can't read {path}: {e}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None


def normalize_tag(tag: str) -> str:
    """Ollama lists a tag without a `:ref` as `:latest`; compare tags in that form."""
    if tag and ":" not in tag.rsplit("/", 1)[-1]:
        return f"{tag}:latest"
    return tag


def check_key(key: str, what: str) -> str:
    if not KEY_RE.match(key):
        raise ConfigError(
            f"{what} {key!r}: use lowercase letters, digits, '.', '_' or '-' "
            "(it becomes a file name)."
        )
    return key


def check_options(values: dict, where: str, allow_seed: bool = False) -> dict:
    """Reject unknown option names and wrong types. Ollama itself only warns and skips them."""
    for name, value in values.items():
        if name not in OPTION_NAMES or (name == "seed" and not allow_seed):
            raise ConfigError(f"{where}: unknown or unsettable option {name!r}.")
        if name == "stop":
            if not (isinstance(value, list) and all(isinstance(s, str) for s in value)):
                raise ConfigError(f"{where}: stop must be a list of strings.")
        elif name in INT_OPTIONS:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigError(f"{where}: {name} must be an integer, got {value!r}.")
        elif not isinstance(value, int | float) or isinstance(value, bool):
            raise ConfigError(f"{where}: {name} must be a number, got {value!r}.")
    return dict(values)


def check_think(value, where: str) -> bool | str:
    if isinstance(value, bool) or value in THINK_LEVELS:
        return value
    raise ConfigError(
        f"{where}: think must be true, false or one of {THINK_LEVELS}, got {value!r}."
    )


def _layer(raw: dict, where: str, source_key: str, table: dict) -> Layer:
    values = dict(raw)
    thinking = check_options(values.pop("thinking", {}), f"{where}.thinking")
    values = check_options(values, where)
    source = table.get(source_key, "")
    if (values or thinking) and not source:
        raise ConfigError(f"{where}: give {source_key} so every value can be traced.")
    return Layer(values, thinking, source)


def _model(key: str, raw: dict) -> Model:
    where = f"models.toml [{check_key(key, 'Model key')}]"
    try:
        think_values = tuple(check_think(v, where) for v in raw.get("think_values", []))
        think_default = check_think(raw.get("think_default", False), where)
        if think_values and think_default not in think_values:
            raise ConfigError(f"{where}: think_default must be one of think_values.")
        return Model(
            key=key,
            name=raw.get("name", key),
            tag=normalize_tag(raw["tag"]),
            upstream=raw.get("upstream", ""),
            quant=raw.get("quant", ""),
            confirmed=bool(raw.get("confirmed", False)),
            weights_sha256=raw.get("weights_sha256"),
            ollama_digest=raw.get("ollama_digest"),
            ctx_max=raw.get("ctx_max"),
            think_values=think_values,
            think_default=think_default,
            stop=tuple(check_options({"stop": raw.get("stop", [])}, where)["stop"]),
            recommended=_layer(
                raw.get("recommended", {}), f"{where}.recommended", "recommended_source", raw
            ),
            inherited=_layer(
                raw.get("inherited", {}), f"{where}.inherited", "inherited_source", raw
            ),
            unsupported=dict(raw.get("unsupported", {})),
            description=raw.get("description", ""),
            build_notes=raw.get("build_notes", ""),
        )
    except KeyError as e:
        raise ConfigError(f"{where}: missing {e.args[0]!r}.") from None


def _machine(path: Path) -> Machine:
    raw = read_toml(path)
    try:
        caps = raw.get("num_ctx_caps", {})
        for key, cap in caps.items():
            if not isinstance(cap, int):
                raise ConfigError(f"{path}: num_ctx_caps.{key} must be an integer.")
        parallel = int(raw.get("parallel", 1))
        if parallel < 1:
            raise ConfigError(f"{path}: parallel must be at least 1.")
        return Machine(
            name=raw.get("name", path.stem),
            ollama_host=raw["ollama_host"],
            gpu_budget_gb=float(raw.get("gpu_budget_gb", 0)),
            parallel=parallel,
            request_timeout_s=float(raw.get("request_timeout_minutes", 60)) * 60,
            keep_alive=str(raw.get("keep_alive", "10m")),
            minutes_per_poem=float(raw.get("minutes_per_poem", 3)),
            num_ctx_caps=dict(caps),
        )
    except KeyError as e:
        raise ConfigError(f"{path}: missing {e.args[0]!r}.") from None


def load_config(root: Path | None = None, machine_path: Path | str | None = None) -> Config:
    root = root or find_root()
    defaults = check_options(read_toml(root / "config" / "defaults.toml"), "defaults.toml")
    missing = OPTION_NAMES - NOT_IN_DEFAULTS - defaults.keys()
    if missing:
        raise ConfigError(f"defaults.toml must set every option; missing: {sorted(missing)}.")
    models = {
        key: _model(key, raw) for key, raw in read_toml(root / "config" / "models.toml").items()
    }
    machine_path = (
        machine_path or os.environ.get("POEMGEN_MACHINE") or root / "config" / "machine.toml"
    )
    machine = _machine(Path(machine_path))
    unknown = set(machine.num_ctx_caps) - models.keys()
    if unknown:
        raise ConfigError(f"machine num_ctx_caps names unknown models: {sorted(unknown)}.")
    return Config(root=root, defaults=defaults, models=models, machine=machine)


def experiment_path(cfg: Config, name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".toml" or path.is_file():
        return path
    return cfg.root / "experiments" / f"{name_or_path}.toml"


def load_experiment(cfg: Config, name_or_path: str) -> Experiment:
    path = experiment_path(cfg, name_or_path)
    raw = read_toml(path)
    name = check_key(path.stem, "Experiment name")
    where = f"experiment {name}"
    try:
        models = tuple(raw["models"])
        unknown = [m for m in models if m not in cfg.models]
        if unknown:
            raise ConfigError(f"{where}: unknown models {unknown}.")
        system_prompts = tuple(raw["system_prompts"])
        user_prompt = raw["user_prompt"]
        for variant in system_prompts:
            if not cfg.system_prompt_path(variant).is_file():
                raise ConfigError(f"{where}: no prompt file {cfg.system_prompt_path(variant)}.")
        if not cfg.user_prompt_path(user_prompt).is_file():
            raise ConfigError(f"{where}: no prompt file {cfg.user_prompt_path(user_prompt)}.")
        sets = []
        for i, s in enumerate(raw["settings"]):
            set_name = check_key(s["name"], f"{where} settings name")
            think = s.get("think", "default")
            sets.append(
                SettingsSet(
                    name=set_name,
                    think=think
                    if think == "default"
                    else check_think(think, f"{where}.{set_name}"),
                    all=check_options(s.get("all", {}), f"{where}.settings[{i}].all"),
                    models={
                        m: check_options(v, f"{where}.settings[{i}].models.{m}")
                        for m, v in s.get("models", {}).items()
                    },
                )
            )
            stray = set(sets[-1].models) - set(models)
            if stray:
                raise ConfigError(
                    f"{where}.{set_name}: changes for models not in the run: {stray}."
                )
        if len({s.name for s in sets}) != len(sets):
            raise ConfigError(f"{where}: settings names must be unique.")
        samples, seed = int(raw["samples"]), int(raw["seed"])
        if samples < 1:
            raise ConfigError(f"{where}: samples must be at least 1.")
    except KeyError as e:
        raise ConfigError(f"{where}: missing {e.args[0]!r}.") from None
    return Experiment(name, models, system_prompts, user_prompt, samples, seed, tuple(sets))
