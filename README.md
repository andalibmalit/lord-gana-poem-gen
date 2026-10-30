# lord-gana-poem-gen

Generates English-language devotional poems addressed to Lord Gana from open-weight models running locally through [Ollama](https://ollama.com). It sweeps system-prompt variants and inference settings, saves every poem with everything needed to trace it, and lets you read the poems blind and take notes.

User prompt: "Please write an original, English-language devotional poem addressed to Lord Gana."

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com), then `uv sync`.
2. Edit `config/machine.toml` for your machine (Ollama host, memory, context caps).

## Everyday loop

1. Edit an experiment in `experiments/` or a prompt in `prompts/`.
2. `uv run poemgen plan <experiment>` to see the planned poems and a time estimate.

More commands are documented here as they're built (see `docs/prd.md`).

## Docs

- `docs/prd.md`: requirements.
- `docs/inference-settings.md`: every inference setting, its neutral value and its Ollama equivalent.
- `docs/recommended-sampling-settings.md`: each model's recommended settings, with sources.
- `docs/scratchpad.md`: loose ideas.
