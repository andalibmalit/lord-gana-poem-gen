# Inference settings

What the project tracks, where each value comes from, which defaults are neutral, and why each setting matters. Written 2026-10-30, from commit `5ca9aaa`, transformers 5.17.0, the Oct 21–27 design sessions, and the cached `generation_config.json` files. Section 6 maps everything to Ollama.

## 1. How transformers picks a value

Inside `generate()`, each setting takes the first value it finds, in this order:

1. Arguments passed to `generate()`. Today `notebooks/initial_exploration.py` passes only `max_new_tokens`.
2. The model's `generation_config.json`.
3. The library fallback table, `GenerationConfig._get_default_generation_params()`. Most of its values are neutral, but not all.

A setting is never truly unset. If it isn't passed and the model doesn't set it, it falls back to step 3.

## 2. Sampling settings

### Tier 1: always in effect

| Setting | Library fallback | Neutral? | Why it's tracked |
|---|---|---|---|
| `do_sample` | False (greedy) | **No** | If off, temperature, top_k and top_p are ignored, and every seed gives the same poem. |
| `temperature` | 1.0 | Yes (1.0 leaves the odds unchanged) | Main knob for creative variety. |
| `top_k` | **50** | **No** (0 means off) | Filters even when nobody asked for it. |
| `top_p` | 1.0 | Yes | Cuts the long tail of unlikely words. |
| `max_new_tokens` | none; the TOML sets 3333 | — | Has to cover reasoning plus the poem. Without it, `max_length` 20 leaves almost no poem. |
| `max_length` | 20 | Overridden by `max_new_tokens` | Logged runs still show 20, which is misleading. |
| `repetition_penalty` | 1.0 | Yes | Kept off on purpose: refrains and mantra lines are deliberate repetition. |
| `no_repeat_ngram_size` | 0 | Yes | Same reason. |

### Tier 2: try later, one at a time

They overlap, so changing two at once hides which one mattered.

| Setting | Off value | Note |
|---|---|---|
| `min_p` | None | Strongest candidate. Designed to keep text coherent at high temperature. |
| `top_h` | None | |
| `typical_p` | 1.0 | |
| `epsilon_cutoff` | 0.0 | |
| `eta_cutoff` | 0.0 | |

### Tier 3: everything else, neutral or unused

Rule: unset or neutral, so nothing changes silently. The logged Gemma E4B run confirms this holds.

- Beam search only: `num_beams` 1, `num_beam_groups` 1, `diversity_penalty` 0.0, `length_penalty` 1.0, `early_stopping` False.
- Encoder-decoder models only: `encoder_repetition_penalty`, `encoder_no_repeat_ngram_size`.
- Speculative decoding only: `num_assistant_tokens`, `assistant_*`, `target_lookbehind`.
- Speed or output format only, not the text: `use_cache`, `output_scores`, `return_dict_in_generate`.
- Presence penalty doesn't exist in transformers. It's an OpenAI-style setting (OpenRouter, Ollama) where 0 means off.

### Set by models, beyond sampling

- **Stop tokens.** Hermes stops on `<|endoftext|>` (151643), but its chat template ends each turn with `<|im_end|>` (151645), so it may write until it hits the token limit.
- **Gemma 12B `suppress_tokens`.** Blocks `<image|>` and `<audio|>`. Harmless for text.

## 3. What each model runs with today

`*` marks a value from the library fallback table rather than the model.

| Model | do_sample | temp | top_k | top_p | Model's own max_new_tokens (3333 wins) | Model card recommends |
|---|---|---|---|---|---|---|
| Gemma 4 E4B / 12B | True | 1.0 | 64 | 0.95 | — | Same |
| Olmo 3 7B Instruct / Think | True | 0.6 | **50\*** | 0.95 | 32768 | 0.6 / 0.95, no top_k |
| Hermes 4 14B | True | **1.0\*** | **50\*** | **1.0\*** | 2048 | 0.6 / 0.95 / 20 |
| Apertus v1.5 8B | **False\*** (greedy) | ignored | ignored | ignored | — | Not listed |
| K2-Horizon 7B | not downloaded | | | | | |

The plan from Oct 22 was "each model's recommended settings as the baseline." That's only in effect for Gemma. Still open: whether to also run one shared setting across all models as a control.

## 4. Run-level settings

| Setting | Current value | Logged? | Why |
|---|---|---|---|
| Seed | 333997 (TOML `[default]`) | Yes | Reproducibility. Reusing one seed for every poem gives identical poems. Plan: one seed per poem (base + index), logged with that poem. |
| Precision | bf16 (model default) | Yes | Changes outputs. |
| Quantization | Hermes and Gemma 12B: Metal 4-bit, group size 32 (about 5 bits per weight). Others: none. | Not yet | Needed to fit in 32 GB, and changes outputs. Plan: log `model.config.quantization_config`, with an empty value when unquantized. |
| Reasoning | Nothing passed: Gemma and Apertus off, Hermes off, Olmo Think always on | Not yet | Likely bigger than any sampling knob. The switch differs per model (see below). |
| Revision | Pinned for weights via `revision=` | `commit_hash` added after the last logged run | The same model name can point to different weights later. |
| Device | mps | Yes | |
| Prompts | MLflow registry `system_prompt/1`, `poem_generation_prompt/1` | Not yet | Plan: log the registry addresses and the full prompt after the chat template, which can add text. |
| Output stats | tokens generated, time, hit the limit, word count | Not yet | Catches poems cut off mid-line, most likely with reasoning on. Word count because each tokenizer counts differently. |

How reasoning is switched in transformers:
- Gemma 4, Apertus v1.5: `enable_thinking` passed to `apply_chat_template`.
- Hermes 4: its `thinking` flag only swaps in the default system prompt, so it does nothing with a custom system prompt. Reasoning needs the template's thinking text added to the system prompt, which mixes it with a prompt change.
- Olmo 3: separate Instruct (no reasoning) and Think (always reasoning) models.

## 5. Known issues in `initial_exploration.py`

1. The loop cell uses `SEED`, which no cell defines, so the notebook doesn't run.
2. The logged settings come from an unpinned (latest) `GenerationConfig.from_pretrained`, while `generate()` uses the pinned one. The tokenizer, which holds the chat template, is also unpinned.
3. The Gemma commit hashes in `src/model_config.toml` are shifted: E4B's `707f0a3` is the 12B's commit, and 12B's `4d7ae49` is the 26B-A4B's. The downloaded E4B is `ee0ef60`.
4. The seed is set once before the loop, so each model's random state depends on which models ran first.
5. A comment in the notebook says `group_size=64`; the TOML uses 32.

## 6. Moving to Ollama

### Ollama's order of precedence

1. `options` sent with the request.
2. `PARAMETER` lines in the model's Modelfile. Check with `ollama show <model> --parameters`.
3. Ollama's built-in defaults (`DefaultOptions()` in `api/types.go`).

The same hidden-defaults lesson applies, and **Ollama's built-in defaults aren't neutral**: temperature 0.8, top_k 40, top_p 0.9. To track and toggle reliably, send every sampling option on every request and log the full `options` dict, so nothing falls through to steps 2 or 3.

### Mapping

| Setting | Ollama option | Ollama default | Neutral / off value | Notes |
|---|---|---|---|---|
| `do_sample` | none | — | — | Greedy is `temperature` 0. |
| `temperature` | `temperature` | **0.8** | 1.0 | Non-neutral default. |
| `top_k` | `top_k` | **40** | 0 | Non-neutral default. |
| `top_p` | `top_p` | **0.9** | 1.0 | Non-neutral default. |
| `min_p` | `min_p` | 0.0 | 0.0 | |
| `typical_p` | `typical_p` | 1.0 | 1.0 | Deprecated: rejected in a Modelfile, accepted per request with a warning. |
| `repetition_penalty` | `repeat_penalty` | 1.0 | 1.0 | Window set by `repeat_last_n` (default 64). |
| presence penalty | `presence_penalty` | 0.0 | 0.0 | Additive. Not available in transformers. |
| frequency penalty | `frequency_penalty` | 0.0 | 0.0 | Additive. Not available in transformers. |
| `max_new_tokens` | `num_predict` | -1 (no limit) | — | Set it as a runaway guard. Counts reasoning tokens too. |
| seed | `seed` | -1 (random each call) | — | Send a seed per poem and log it. |
| stop tokens | `stop` | from the model's template | — | Log what the Modelfile sets. |
| reasoning | `think` (request field, not an option) | per model | `false` | Accepts `true`, `false`, `"low"`, `"medium"`, `"high"`, `"max"`. Same field for every model, but what it does still varies, and some models can't turn reasoning off. Reasoning comes back separately in `message.thinking`. |
| precision / quantization | the model tag (for example `Q4_K_M`) | — | — | Fixed by the GGUF file, not a runtime option. Log the tag. |
| revision | model digest | — | — | From `ollama list` or `/api/tags`. Replaces the Hugging Face commit hash. |

Not available in Ollama: `top_h`, `epsilon_cutoff`, `eta_cutoff`, `no_repeat_ngram_size`, `suppress_tokens`.

### Ollama-only settings worth tracking

| Option | Default | Why |
|---|---|---|
| `num_ctx` | `OLLAMA_CONTEXT_LENGTH`, else 4k, 32k or 256k depending on GPU memory (the Modelfile docs page still says 2048) | **Set it explicitly.** The system prompts are 3,400–3,700 words, roughly 5–8K tokens, and IAST diacritics add tokens. If the context is too small, the system prompt gets cut off silently. Context must hold the prompt plus `num_predict`. |
| `raw` (`/api/generate` only) | false | Skips the chat template, so you control the exact prompt. Useful for base models. |
| `logprobs`, `top_logprobs` | off | Per-token probabilities. Useful later for studying how sure a model was when it made each word choice. |
| `draft_num_predict` | 4 when a draft model is used | Speculative decoding. 0 turns it off. |
| `num_keep`, `num_batch`, `num_gpu`, `num_thread`, `keep_alive` | various | Speed and memory only, not the text. |

### What to log per request

- The full request: model tag and digest, `options`, `think`, messages.
- `ollama show <model> --modelfile`, which includes the template and baked-in parameters.
- The Ollama version.
- From the response: `done_reason` (`length` means it hit the limit), `eval_count`, `prompt_eval_count`, `total_duration`, `eval_duration`, and `message.thinking` stored separately from the poem.

### Unverified

Ollama's API accepts all the options above. Not checked: whether every engine honors every option for every model. Before relying on a setting, run a quick test: same seed, change only that option, and check that the output changes.

## Sources

- transformers 5.17.0: `generation/configuration_utils.py` (`_get_default_generation_params`), `generation/utils.py` (merge order).
- Cached `generation_config.json` for each model in `~/.cache/huggingface/hub`.
- MLflow run `0b589312…` (Gemma E4B, 2026-10-23): the logged params.
- Ollama: `api/types.go` (`Options`, `DefaultOptions`, `ThinkValue`), `envconfig/config.go` (`OLLAMA_CONTEXT_LENGTH`), docs.ollama.com/modelfile. Checked 2026-10-30.
