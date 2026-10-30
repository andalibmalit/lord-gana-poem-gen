"""Command line: `poemgen <command>`. Run `poemgen --help` for the list."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from poemgen.config import Config, ConfigError, load_config, load_experiment
from poemgen.plan import Plan, build_plan, estimate_minutes


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
    print_plan(cfg, build_plan(cfg, exp, args.models))
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
