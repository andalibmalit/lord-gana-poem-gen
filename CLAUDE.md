# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working conventions (read first)

Since 2026-10-30, Claude writes the code here so the user can spend their time reading generated poems and taking notes on what works. The hard part of this project is the poems, not the plumbing: getting outputs that feel creative and carry a sense of presence.

- The user directs what to build. Implement what they ask; suggest next steps, but don't build ahead of them.
- Tutor mode on request: if the user says something like "tutor me on this", switch to their `hint-first` skill for that piece (get an attempt first, then one hint level per new attempt, never paste the solution).
- `CLAUDE.md` may be committed. Run `git add` or `git commit` only when the user asks.

## Project

Generates English-language devotional poems addressed to Lord Gana (Ganesha) by sweeping one system prompt across many LLMs and generation settings, then storing and exporting the results.

Planned pipeline, from `README.md`:
1. System prompt variants in `prompts/`: a devotional preamble plus Archive-tradition glossary and an annotated translation of the Gaṇapati Atharvaśīrṣa. Variants: first person ("I") vs second person ("You") opening (`system_prompt_1st-p.md`, `system_prompt_2nd-p.md`), each also with IAST transliteration (`*_IAST.md`). User prompt in `prompts/poem_generation.md`.
2. A list of models (local or remote) with per-model settings (temperature, reasoning on/off).
3. Generation sweeps over models × settings, in parallel if memory allows. User prompt: "Please write an original, English-language devotional poem addressed to Lord Gana."
4. Sweep parameters and outputs logged to a local MLflow server.
5. Export from MLflow for analysis and sharing.

Libraries: Marimo (notebooks), Hugging Face transformers (local models), OpenRouter (remote models, later), MLflow (storage), polars (analysis).

## State and environment

- Early stage. Managed with uv (`pyproject.toml`, Python 3.13, `.venv/`). `.claude/` is excluded via `.git/info/exclude`.
- Layout: marimo notebooks in `notebooks/` (`openrouter_sweep.py` is an empty stub), system prompts in `prompts/`, transformers model list and settings in `src/model_config.toml` (TOML, read with `tomllib`), Ollama modelfiles for local GGUF models in `src/ollama/`. MLflow stores runs in a local `mlflow.db`. A quantization script in `scripts/quantize.py` is on hold in favor of OpenRouter.
- Run a notebook: `uv run marimo edit notebooks/<name>.py` (interactive) or `uv run notebooks/<name>.py` (as a script). `ruff` is the linter. No tests yet.
