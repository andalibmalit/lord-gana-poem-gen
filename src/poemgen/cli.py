"""Command line: `poemgen <command>`. Run `poemgen --help` for the list."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import httpx

from poemgen.config import Config, ConfigError, Model, load_config, load_experiment
from poemgen.ollama import Ollama, OllamaError
from poemgen.plan import Plan, build_plan, estimate_minutes
from poemgen.registry import Build, fetch_build
from poemgen.run import Runner, redact_modelfile


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _minutes(m: float) -> str:
    return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"


def print_plan(cfg: Config, plan: Plan, model_status: dict[str, str] | None = None) -> None:
    pending = plan.pending
    estimates = estimate_minutes(cfg, pending)
    total = sum(estimates.values())
    print(
        f"Experiment {plan.experiment.name}: {len(plan.specs)} poems planned, "
        f"{len(plan.done)} done, {len(pending)} to run, about {_minutes(total)}."
    )
    rows = defaultdict(lambda: [0, 0])  # (model, settings set, think) -> [to run, done]
    for s in plan.specs:
        rows[(s.model.key, s.settings_set, str(s.think).lower())][s.poem_id in plan.done] += 1
    if rows:
        header = ("model", "settings", "think", "to run", "done", "num_ctx", "status")
        table = [header]
        for (key, sset, think), (todo, done) in rows.items():
            num_ctx = next(
                s.options["num_ctx"]
                for s in plan.specs
                if s.model.key == key and s.settings_set == sset
            )
            status = (model_status or {}).get(key, "")
            table.append((key, sset, think, str(todo), str(done), str(num_ctx), status))
        widths = [max(len(r[i]) for r in table) for i in range(len(header))]
        print()
        for r in table:
            print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())
    for title, items in (("Skipped", plan.skipped), ("Warnings", plan.warnings)):
        if items:
            print(f"\n{title}:")
            for item in items:
                print(f"  - {item}")
    if plan.orphans:
        print(
            f"\n{len(plan.orphans)} poem file(s) in poems/{plan.experiment.name}/ come from an "
            "earlier config. They're kept, and included in reading and MLflow."
        )


def cmd_plan(cfg: Config, args) -> int:
    exp = load_experiment(cfg, args.experiment)
    plan = build_plan(cfg, exp, args.models)
    print_plan(cfg, plan, model_status(cfg, {s.model.key for s in plan.specs}))
    return 0


def model_status(cfg: Config, keys: set[str]) -> dict[str, str]:
    """Whether each model is pulled with the configured digest, if Ollama is reachable."""
    client = Ollama(cfg.machine.ollama_host, timeout_s=5)
    try:
        local = client.tags()
    except OllamaError:
        return {key: "(Ollama not reachable)" for key in keys}
    finally:
        client.close()
    status = {}
    for key in keys:
        model = cfg.models[key]
        if not model.confirmed:
            status[key] = "build not confirmed"
        elif model.tag not in local:
            status[key] = "not pulled"
        elif local[model.tag] != model.ollama_digest:
            status[key] = "DIGEST DIFFERS from config"
        else:
            status[key] = "ready"
    return status


def cmd_run(cfg: Config, args) -> int:
    exp = load_experiment(cfg, args.experiment)
    plan = build_plan(cfg, exp, args.models)
    print_plan(cfg, plan)
    print()
    client = Ollama(cfg.machine.ollama_host, cfg.machine.request_timeout_s)
    try:
        summary = Runner(cfg, client).run(plan)
    except KeyboardInterrupt:
        print("\nInterrupted. Finished poems are saved; rerun the same command to resume.")
        return 130
    finally:
        client.close()
    print(
        f"\nDone: {summary.written} written, {summary.failed} failed"
        + (
            f", skipped models: {', '.join(summary.skipped_models)}"
            if summary.skipped_models
            else ""
        )
        + (f". Failures are in poems/{exp.name}/failures.jsonl." if summary.failed else ".")
    )
    return 1 if summary.failed or summary.skipped_models else 0


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def describe_build(build: Build, model: Model | None, show_template: bool) -> list[str]:
    lines = [f"exists, download {_gb(build.total_bytes)} (weights {_gb(build.weights_bytes)})"]
    lines.append(f"weights {build.weights_sha256}")
    if model is not None:
        if not model.weights_sha256:
            lines.append("config: weights_sha256 not pinned yet")
        elif model.weights_sha256 != build.weights_sha256:
            lines.append(f"MISMATCH: config pins {model.weights_sha256}")
        else:
            lines.append("weights match config")
        if sorted(build.stop) != sorted(model.stop):
            lines.append(f"STOP DIFFERS: build {build.stop} vs config {list(model.stop)}")
    shipped = {k: v for k, v in (build.params or {}).items() if k != "stop"}
    lines.append(f"stop {build.stop}; shipped params {shipped or 'none'}")
    extra = [media.rsplit(".", 1)[-1] for media, _ in build.layers if not media.endswith("model")]
    lines.append(f"other layers: {', '.join(extra) or 'none'}")
    if build.template is None:
        lines.append("no template layer (Ollama's built-in renderer, or the GGUF's own)")
    elif show_template:
        lines += ["template:", *("    " + t for t in build.template.splitlines())]
    return lines


def cmd_models_check(cfg: Config, args) -> int:
    targets: list[tuple[str, str, Model | None]] = [(t, t, None) for t in args.tag or []]
    keys = args.keys or ([] if args.tag else list(cfg.models))
    for key in keys:
        if key not in cfg.models:
            raise ValueError(f"unknown model {key!r}")
        targets.append((key, cfg.models[key].tag, cfg.models[key]))
    problems = 0
    with httpx.Client(timeout=30, follow_redirects=True) as http:
        for label, tag, model in targets:
            print(f"{label}  {tag or '(no tag yet)'}")
            if not tag:
                problems += 1
                continue
            try:
                lines = describe_build(fetch_build(tag, http), model, args.template)
            except (LookupError, httpx.HTTPError) as e:
                lines = [f"ERROR: {e}"]
            problems += any(w in ln for ln in lines for w in ("ERROR", "MISMATCH", "DIFFERS"))
            for line in lines:
                print(f"    {line}")
    return 1 if problems else 0


def cmd_models_pull(cfg: Config, args) -> int:
    client = Ollama(cfg.machine.ollama_host, timeout_s=60)
    try:
        for key in args.keys:
            model = cfg.models.get(key)
            if model is None or not model.tag:
                raise ValueError(f"{key}: unknown model or no tag in config/models.toml")
            build = fetch_build(model.tag)
            if model.weights_sha256 and build.weights_sha256 != model.weights_sha256:
                raise ValueError(
                    f"{key}: registry weights {build.weights_sha256} differ from the pinned "
                    f"{model.weights_sha256}. Not pulling."
                )
            print(f"{key}: pulling {model.tag} ({_gb(build.total_bytes)})")
            last = [""]

            def progress(event: dict, last=last) -> None:
                status = event.get("status", "")
                if event.get("total"):
                    status += f" {100 * event.get('completed', 0) // event['total']}%"
                if status != last[0] and (not event.get("total") or status.endswith("0%")):
                    print(f"    {status}")
                    last[0] = status

            client.pull(model.tag, progress)
            digest = client.tags()[model.tag]
            _, weights = redact_modelfile(client.show(model.tag).get("modelfile", ""))
            if weights != build.weights_sha256:
                print(
                    f"    WARNING: local weights {weights} differ from the registry's "
                    f"{build.weights_sha256}"
                )
            print(f"    record in config/models.toml [{key}]:")
            print(f'      weights_sha256 = "{weights}"')
            print(f'      ollama_digest = "{digest}"')
    finally:
        client.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poemgen", description="Generate devotional poems to Lord Gana with local models."
    )
    parser.add_argument(
        "--machine", help="machine config file (default: $POEMGEN_MACHINE or config/machine.toml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="dry run: list the planned poems, counts and a time estimate")
    p.add_argument("experiment", help="experiment name (experiments/<name>.toml) or path")
    p.add_argument("--models", type=_csv, help="comma-separated subset of the experiment's models")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="generate the planned poems that don't exist yet")
    p.add_argument("experiment", help="experiment name (experiments/<name>.toml) or path")
    p.add_argument("--models", type=_csv, help="comma-separated subset of the experiment's models")
    p.set_defaults(func=cmd_run)

    models = sub.add_parser("models", help="check or pull model builds").add_subparsers(
        dest="models_command", required=True
    )
    p = models.add_parser("check", help="look up builds in their registry, without pulling")
    p.add_argument("keys", nargs="*", help="model keys from config/models.toml (default: all)")
    p.add_argument("--tag", action="append", help="also check any Ollama tag not in the config")
    p.add_argument("--template", action="store_true", help="print each build's chat template")
    p.set_defaults(func=cmd_models_check)
    p = models.add_parser("pull", help="pull exactly the configured builds and print their digests")
    p.add_argument("keys", nargs="+", help="model keys from config/models.toml")
    p.set_defaults(func=cmd_models_pull)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Poems and prompts contain IAST diacritics: never let a console encoding crash the tool.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(machine_path=args.machine)
        return args.func(cfg, args)
    except (ConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
