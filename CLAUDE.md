# CLAUDE.md

Guidance for Claude Code in this repository.

## Working conventions

- Claude writes the code; Andalib spends his time reading poems and taking notes. Andalib directs what to build: suggest next steps, don't build ahead of him.
- Tutor mode on request: if Andalib says something like "tutor me on this", use his `hint-first` skill for that piece.
- Commit when Andalib asks (for a planned build, at each checkpoint). Never push.
- **Never start a generation run or pull a model without Andalib's explicit approval.** Planning (`poemgen plan`) and checks (`poemgen models check`) are fine.

## Project

`poemgen` generates English devotional poems to Lord Gana from open-weight models running locally through Ollama. It sweeps system-prompt variants and inference settings, saves every poem with everything needed to trace it, and supports blind reading with notes. Requirements: `docs/prd.md`.

## Commands

```
uv sync                                          # install
uv run poemgen plan <experiment> [--models a,b]  # dry run: counts, settings, estimate, pull status
uv run poemgen run <experiment> [--models a,b]   # generate (needs approval, see above)
uv run poemgen models check [keys] [--tag T]     # registry lookup without pulling
uv run poemgen models pull <keys>                # pull exactly the configured build (needs approval)
uv run poemgen read <experiment> [--reveal]      # blind-reading page + notes server (binds a port)
uv run poemgen mlflow sync [experiments]         # mirror poems/ and notes/ into mlflow.db
uv run pytest                                    # tests (no network, no Ollama)
uv run pytest -m ollama                          # opt-in: render checks against the local Ollama
uv run ruff check && uv run ruff format --check && uv run ty check
```

Sandbox notes for Claude: `uv run poemgen models *` is excluded from the sandbox, because Python HTTPS needs the CA bundle that a global `Read(**/*.pem)` rule blocks. Run it as its own command, not chained, or the exclusion doesn't match. Ollama on localhost works inside the sandbox. Delete files with Python's `shutil`, because `rm -rf` is denied globally.

## Layout

- `config/defaults.toml`: neutral value for every Ollama option. Every option is sent on every request.
- `config/models.toml`: the model set and the source of truth for builds (tag, digests, thinking, stop tokens, recommended settings with sources).
- `config/machine.toml`: this machine only (Ollama host, memory budget, parallel slots, num_ctx caps). Override with `--machine` or `POEMGEN_MACHINE`.
- `experiments/<name>.toml`: models, prompt variants, settings sets, samples, seed.
- `prompts/system/*.md`, `prompts/user/*.md`: prompt variants. The file stem is the variant name.
- `poems/`: **the committed record.** `poems/<experiment>/<poem_id>.md` is YAML front matter plus the poem verbatim. Thinking is in `<poem_id>.thinking.md`. Prompt and Modelfile snapshots are in `poems/_prompts/` and `poems/_models/`.
- `notes/<experiment>.json`: committed reading notes and tags, keyed by poem_id.
- `src/poemgen/`: the package.
  - `config.py`: loading and validation.
  - `plan.py`: settings layering, ids, seeds, estimates.
  - `records.py`: poem files.
  - `ollama.py`: HTTP client.
  - `run.py`: preflight checks, generation, records.
  - `registry.py`: manifest lookups.
  - `reading.py` and `reading_page.html`: blind reading.
  - `mlflow_sync.py`: the MLflow mirror.
  - `cli.py`: the `poemgen` command.
- `mlflow.db`, `mlartifacts/`: a local MLflow viewer rebuilt from `poems/`. Git-ignored. Keep the existing data (experiments 0–2, registered prompts). `mlflow` is pinned to the version that created the db.

## Conventions

- Poem files are immutable records. A config change creates new poem ids. Old poems stay as "orphans" and are still read and synced.
- Settings layers, later wins: defaults → model `inherited` → model `recommended` (plus `thinking` sub-tables when thinking is on) → experiment changes. `num_ctx` is then capped by `machine.toml` and the model's `ctx_max`. Every value records its origin.
- Cross-platform: `pathlib`; every `open()` passes `encoding="utf-8"` (ruff rule PLW1514); no macOS-only code; machine-specific values only in `config/machine.toml`.
- A model runs only if its local build matches `models.toml` exactly: digest, weights, stop tokens, context length, and thinking capability. To change a build, edit `models.toml`, run `models check`, pull, then paste the printed digests in. The tool never writes config.
- Chat templates differ per model (thinking switches, system-prompt edits). Each poem records the rendered prompt; check it when adding a model.
- Every `src/` function has type hints (ruff ANN rules); `ty` checks them. CI (`.github/workflows/tests.yml`) runs everything on Linux, Windows and macOS.
- Keep it simple: stdlib plus httpx, pyyaml and mlflow.
