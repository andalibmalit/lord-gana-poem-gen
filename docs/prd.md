# PRD: Local Poem Generation Harness

Oct 30, 2026 · @Andalib

## Overview

Build a small Python tool that generates devotional poems to Lord Gana from 23 open-weight models running locally on a 32 GB M2 Max. It sweeps a small grid of system-prompt variants and inference settings, and saves every poem with its exact settings.

- **Problem:** generating poems model by model by hand is slow, hard to repeat, and loses track of which settings produced which poem.
- **Outcome:** one command runs an experiment described in a config file, survives interruption, and leaves a file of poems that can be read, filtered and analysed later.
- **Principles:** simplicity first; local-first; no automated judging; built so that cloud APIs can be added later without a rewrite.

Claude Code may treat the repo as greenfield. Earlier design choices (the TOML layout, the notebook, transformers) are not commitments; Claude Code can recommend research-backed changes, as long as the tool stays essentially simple. KISS, DRY, YAGNI code with (non-overengineered) extensibility and portability in mind.

## Scope

Version 1 runs everything locally through Ollama, tracks experiments in MLflow, and lets Andalib read poems blind and take notes on them.

**In scope (v1)**

- Local generation through Ollama for the 23 models in the model set below
- Running requests in parallel where memory allows
- System-prompt variants kept as text files
- A small grid of inference settings per experiment
- Every poem saved with its full settings
- Experiment tracking in MLflow
- Resuming an interrupted run without duplicates
- Blind reading: poems shown without their model or settings, with notes and tags saved per poem and joinable to the settings afterwards

**Out of scope (v1)**

- Cloud APIs (designed for, not built)
- Automated judging or rating
- Fine-tuning and interpretability

**Later**

- Cloud providers (OpenRouter, Featherless and others)
- Comparing local 4-bit poems with hosted full-precision poems for the same model
- Blind reading for friends, not only Andalib
- Running on other platforms (Linux, Windows): a goal, not required for v1

## Users and workflow

One user, Andalib, runs the tool on his own MacBook and reads every poem himself.

- **How it gets built:** this PRD goes to Claude Code, which turns it into a technical spec/plan then implements it. The spec should break into small steps, each testable on its own.
- **Everyday loop:**
  1. Edit the config or a prompt file.
  2. Run a dry run to see the planned poems and the time estimate.
  3. Start the run and leave it going.
  4. Read the poems blind, adding notes and tags; then reveal and compare them against the settings.
  5. Adjust and rerun.
- **Waiting is fine:** a few minutes per poem is acceptable (Hemmingway-1.5 already ran fine locally on 30 Oct).

## Model set

V1 covers 23 local models: 10 with plenty of memory headroom and 13 that fit close to the Mac's GPU memory limit (about 26.8 GB, measured with `torch.mps.recommended_max_memory()`; no `sudo` needed). Claude Code should confirm each tag before the first run.

| Model | Ollama source | Quant | Thinking | Memory |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B | `gemma4:e4b-it-q8_0` | Q8_0 | Switchable, off by default | Headroom |
| Gemma 4 12B | `gemma4:12b-it-q8_0` | Q8_0 | Switchable, off by default | Headroom |
| Gryphe StyleTune 12B | `hf.co/mradermacher/Gemma-4-12B-StyleTune-GGUF:Q6_K` | Q6_K | As Gemma 4 (to confirm) | Headroom |
| Gemma 3 12B | `gemma3:12b-it-qat` | QAT Q4_0 | None | Headroom |
| Qwen3-14B | `hf.co/Qwen/Qwen3-14B-GGUF:Q6_K` | Q6_K | Switchable | Headroom |
| Hermes 4 14B | `hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q6_K` | Q6_K | Switchable, off by default | Headroom |
| Olmo 3 7B Instruct | `olmo-3:7b-instruct-q8_0` | Q8_0 | None | Headroom |
| Olmo 3 7B Think | `olmo-3:7b-think-q8_0` (to confirm) | Q8_0 | Always on | Headroom |
| Apertus v1.5 8B | `hf.co/andreasmartin/apertus-v1.5-8b-text-Q8_0-GGUF` (community text-only build; fallback: v1.0, `hf.co/unsloth/Apertus-8B-Instruct-2509-GGUF:Q8_0`) | Q8_0 | Optional (to confirm for this build) | Headroom |
| Gemma 4 26B-A4B | `gemma4:26b-a4b-it-qat` | QAT Q4_0 | Switchable, off by default | Headroom |
| Gemma 4 31B | `gemma4:31b-it-qat` | QAT Q4_0 | Switchable, off by default | Close to limit |
| Gemma 3 27B | `gemma3:27b-it-qat` | QAT Q4_0 | None | Close to limit |
| Muse Glimmer 30B | `muse-glimmer` | Q4_K_M | Always on (low effort at minimum) | Close to limit |
| Qwen3.8-27B | `qwen3.8:27b-q4_K_M` | Q4_K_M | Switchable, on by default | Close to limit |
| Hemmingway-1 | `hf.co/mradermacher/Hemmingway-1-GGUF:Q4_K_M` | Q4_K_M | Switchable; template defaults to extra-high | Close to limit |
| Hemmingway-1.5 | `hf.co/mradermacher/Hemmingway-1.5-GGUF:Q4_K_M` | Q4_K_M | Off, low, medium or extra-high | Close to limit |
| Writer-V2 (Qwen3.5-27B) | `hf.co/mradermacher/Qwen3.5-27B-Writer-V2-GGUF:Q4_K_M` | Q4_K_M | Card suggests an empty think prefill | Close to limit |
| Marvin-DPO-V2 (Qwen3.5-27B) | GGUF repo to pick | Q4_K_M | To confirm | Close to limit |
| Musica (Qwen3.5-27B) | GGUF repo to pick | Q4_K_M | To confirm | Close to limit |
| Gemsicle-31B (Gemma 4) | GGUF repo to pick | Q4_K_M | As Gemma 4 (to confirm) | Close to limit |
| Nemotron 3.5 Lightning | `hf.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:Q4_0` | Q4_0 | Switchable | Close to limit |
| Hermes 4.3 36B | `hf.co/NousResearch/Hermes-4.3-36B-GGUF:Q3_K_M` | Q3_K_M | Switchable, off by default | Close to limit |
| Seed-OSS-36B | `hf.co/unsloth/Seed-OSS-36B-Instruct-GGUF:Q3_K_M` | Q3_K_M | To confirm | Close to limit |

The Apertus v1.5 build is a community text-only conversion whose correctness is unverified. Check a few of its poems against the same prompt on Public AI's hosted v1.5 before trusting it; if it misbehaves, use v1.0.

## Requirements

These say what the tool must do; how to do it is for Claude Code's technical spec.

| ID | Requirement | Priority |
| --- | --- | --- |
| **Experiments** | | |
| R-1 | Models, prompt variants, the user prompt, settings sets, samples per combination and seeds are described in config. Adding a model or a prompt variant needs no code change. | Must |
| R-17 | Any subset of models can be picked for a run without editing the model list. | Must |
| R-2 | A dry run shows the planned poems, their count and an estimated time. Nothing runs until Andalib starts it. | Must |
| **Inference settings** | | |
| R-3 | Track and toggle as many inference settings as Ollama supports, as listed in the repo's "Inference settings" doc (30 Oct): sampling (temperature, top_k, top_p, min_p, typical_p), penalties (repeat, presence, frequency), `num_predict`, `num_ctx`, seed, stop tokens and thinking. | Must |
| R-18 | Each model starts from its maker's recommended settings, recorded in config with where they came from. Where the model states none, use the base model's recommendation, then its config file, and mark the value as inherited. | Must |
| R-19 | Settings can be changed for one model or for all models at once, without editing each model. Each poem records which values came from the baseline and which from a change. | Must |
| R-20 | Where a recommendation uses a setting Ollama doesn't have, apply the rest and note what was left out. | Should |
| R-4 | Send every setting on every request and record exactly what was sent, because Ollama's built-in defaults aren't neutral (temperature 0.8, top_k 40, top_p 0.9). | Must |
| R-5 | Defaults of `num_ctx` 40960 and `num_predict` 32768, with per-model overrides where memory requires a smaller context. | Must |
| R-6 | A distinct seed per poem, recorded with it. | Must |
| **Generation** | | |
| R-7 | Generate locally through Ollama, keeping any thinking text separate from the poem. | Must |
| R-8 | Run requests in parallel where memory allows. | Should |
| R-9 | An interrupted run resumes without duplicates; a failed request is logged and the run continues. | Must |
| R-10 | Poems cut off by the length limit are flagged. | Must |
| **Records** | | |
| R-11 | Every poem can be traced to its model, digest and quantization, the exact prompt text, every setting sent, its seed, token counts, timings, the Ollama version and the model's Modelfile. | Must |
| R-12 | Experiments and poems are tracked in MLflow. | Must |
| **Blind reading** | | |
| R-13 | Poems can be read with their model and settings hidden, in shuffled order. | Must |
| R-14 | Free-text notes and Andalib's own tags can be attached to each poem, then joined to its settings for comparison. | Must |
| R-15 | The reading setup is reproducible from the repo's source code. | Must |
| **Later-ready** | | |
| R-16 | Cloud providers can be added later without a rewrite. | Should |

## Known facts for the spec

Things already checked, so Claude Code doesn't have to rediscover them. They aren't design decisions.

- **Memory.** The Mac's GPU budget is 26.8 GB (`torch.mps.recommended_max_memory()`). There's no sudo, and none is needed.
- **Context fit.** At `num_ctx` 40960, every model fits except Hermes 4.3 36B and Seed-OSS-36B (about 29.4 GB); both fit at 24576 (about 25.1 GB). Gemma 4 31B fits, but close to the limit (24.7 GB). Qwen3-14B and Hermes 4 14B were trained to 40960 at most. These are estimates; check them on the first load.
- **Prompt length.** System prompts are about 5–7k tokens depending on the variant, and IAST diacritics add tokens. If the context is too small, Ollama cuts the prompt off silently.
- **Parallel requests** multiply the memory for context: each parallel slot holds its own copy.
- **API.** Ollama's native chat API sets every option. Its OpenAI-compatible route can't set `num_ctx`, `top_k` or `min_p`.
- **Thinking.** It comes back separately from the poem. Some models always think (Olmo 3 Think, Muse Glimmer), and what `think` does varies by model.
- **Recommended settings.** Each model's recommended settings, with sources, are in `recommended-sampling-settings.md` (Claude project, 30 Oct); add it to the repo next to the inference-settings doc. Hemmingway-1 and Apertus v1.5 state none, and Gemma 3 27B's values are unverified. Qwen3-14B and Qwen3.8-27B recommend different settings with thinking on and off, and Qwen3.8's thinking-off set includes `presence_penalty` 1.5.
- **Seeds** make reruns close but not guaranteed identical, especially with parallel requests.
- **Not available in Ollama:** `top_h`, `epsilon_cutoff`, `eta_cutoff`, `no_repeat_ngram_size`, `suppress_tokens`. Before relying on any setting, check that changing it alone changes the output (same seed).

## Experiments

Andalib chooses every experiment. Claude Code and Claude may suggest one, but must not start a run.

The tool should make it easy to run a small slice: a subset of models, one or a few prompt variants, a few settings sets and a few poems each. The space stays small at first because every poem gets read.

## Success criteria

- A run Andalib starts over the chosen models finishes unattended, or resumes cleanly after an interruption.
- Every poem can be traced to everything in R-11, and the settings actually sent match the config.
- Andalib can read a batch blind, add notes and tags, and then see them alongside each poem's model and settings.
- Someone with the repo and Ollama can reproduce the setup on another machine, given enough memory.

## Open questions

- **Reading tool.** A static page generated by a script in the repo, or a small app such as Streamlit or Potato. Claude recommends the static page (see the comment on blind reading); the spec settles it.
- **Shared control setting.** One setting used across all models, alongside each model's own. Research found no obvious choice, so this is for trial and error.
- **Which GGUF builds.** Some fine-tunes have several community quantizations; the spec should pick one per model and record it.
- **Apertus v1.5.** The community text-only build hasn't been checked against the official model; v1.0 is the fallback.
