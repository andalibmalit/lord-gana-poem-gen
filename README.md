# lord-gana-poem-gen

Generates English-language devotional poems addressed to Lord Gana from open-weight models running locally through [Ollama](https://ollama.com). It sweeps system-prompt variants and inference settings, saves every poem with everything needed to trace it, and lets you read the poems blind and take notes.

User prompt: "Please write an original, English-language devotional poem addressed to Lord Gana."

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) (0.40 or later), then run `uv sync`.
2. Edit `config/machine.toml` for your machine: Ollama host, GPU memory, per-model context caps.
3. Pull the models you want, exactly as configured: `uv run poemgen models pull gemma4-e4b`. It prints the digests to record in `config/models.toml` (`ollama_digest`).

**Ollama server settings.** `OLLAMA_NUM_PARALLEL` (default 1) applies to every model and multiplies its context memory. Keep it at 1 for the large models, and set `parallel` in `machine.toml` to the same value.

## Everyday loop

```
uv run poemgen plan smoke          # 1. dry run: planned poems, settings, time estimate, pull status
uv run poemgen run smoke           # 2. generate; leave it running. Ctrl+C is safe: rerun to resume
uv run poemgen read smoke          # 3. read blind in your browser, with notes and tags
uv run poemgen read smoke --reveal # 4. see each poem's model and settings next to your notes
uv run poemgen mlflow sync         # 5. optional: mirror poems and notes into MLflow
```

Then adjust an experiment or prompt and go again. `--models a,b` limits `plan` and `run` to some of an experiment's models.

## Files

- `experiments/<name>.toml`: what to generate (models, prompt variants, settings sets, samples, seed). Copy `smoke.toml` to start a new one.
- `prompts/system/`, `prompts/user/`: prompt variants. The file name (without `.md`) is the variant's name.
- `config/models.toml`: the 23 models, which build of each to run and why, and each maker's recommended settings with sources.
- `config/defaults.toml`: a neutral value for every setting. All of them are sent with every request.
- `poems/<experiment>/`: **the record.** One Markdown file per poem: its settings in the front matter, then the poem exactly as generated. Thinking, if any, is in `<id>.thinking.md`.
- `notes/<experiment>.json`: your reading notes and tags.

Both `poems/` and `notes/` are committed to git. MLflow (`mlflow.db`) is a local viewer rebuilt from them and is not committed.

## How settings are chosen

For each poem, every setting starts at its neutral value (`config/defaults.toml`). Then:
1. The model's inherited settings apply (from its base model or config file, when the maker states none).
2. The maker's recommended settings apply (`recommended`, plus a `thinking` set for some models).
3. The experiment's changes apply.
4. `num_ctx` is capped by `machine.toml` and the model's maximum.

Each poem records where every value came from.

To change a setting for all models, use `all = { ... }` in an experiment's settings set. For one model, use `models.<key> = { ... }`.

## Viewing in MLflow

```
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5001
```

Each experiment appears as `poems/<name>`, with one run per poem, its settings as parameters and your notes as tags.

## Development

`uv run pytest` (no network or Ollama needed) and `uv run ruff check`.

## Docs

- `docs/prd.md`: requirements.
- `docs/inference-settings.md`: every inference setting, its neutral value, and what was verified about Ollama.
- `docs/recommended-sampling-settings.md`: each model's recommended settings, with sources.
- `docs/scratchpad.md`: loose ideas.
