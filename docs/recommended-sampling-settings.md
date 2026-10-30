# Recommended sampling settings for the 23 local models

*Checked 30 Oct 2026 against Hugging Face model cards, `generation_config.json` files and Ollama params blobs. I re-checked four entries against their sources: EQ-Bench, Seed-OSS, Qwen3.8-27B and Gemsicle. "Not stated" means neither the card nor the config gives a value.*

## Per model

| Model | temp | top_p | top_k | min_p | Penalties | Notes | Source |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B, 12B, 26B-A4B, 31B | 1.0 | 0.95 | 64 | – | – | The card says to use these "across all use cases"; Ollama ships the same values | huggingface.co/google/gemma-4-E4B-it, -12B-it, -26B-A4B-it, -31B-it |
| Gryphe StyleTune 12B | 1.0 | (0.95 in config) | (64 in config) | 0.10 | DRY, strength not given | Informal: "I run with temp 1.0, 0.10 MinP and the DRY sampler." **DRY isn't in Ollama** | huggingface.co/Gryphe/Gemma-4-12B-StyleTune |
| Gemma 3 12B | 1.0 | 0.95 | 64 | – | – | Ollama params; the Google repo is gated | Ollama `gemma3:12b-it-qat` params |
| Gemma 3 27B | 1.0 | 0.95 | 64 | – | – | **Unverified for 27B**; assumed from the 12B | – |
| Qwen3-14B | thinking 0.6, non-thinking 0.7 | 0.95 / 0.8 | 20 | 0 | Optional presence 0–2 against endless repeats | No greedy decoding. Max output 32,768 | huggingface.co/Qwen/Qwen3-14B |
| Hermes 4 14B | 0.6 | 0.95 | 20 | – | – | One set for both modes | huggingface.co/NousResearch/Hermes-4-14B |
| Olmo 3 7B Instruct / Think | 0.6 | 0.95 | – | – | – | Max output 32,768 | huggingface.co/allenai/Olmo-3-7B-Instruct, -Think |
| Apertus v1.0 8B | 0.8 | 0.9 | – | – | – | Card tip | huggingface.co/swiss-ai/Apertus-8B-Instruct-2509 |
| Apertus v1.5 8B | not stated | not stated | not stated | – | – | Thinking off by default | huggingface.co/swiss-ai/Apertus-v1.5-8B |
| Muse Glimmer 30B | 1.0 | 0.95 | 64 | – | – | Reasoning strength is set in the system prompt (low / medium / high / xhigh) | huggingface.co/meta-models/Muse-Glimmer-30B |
| Qwen3.8-27B | thinking 1.0, non-thinking 0.7 | 0.95 / 0.80 | 20 | 0 | Thinking: presence 0. **Non-thinking: presence 1.5**. Repeat penalty 1.0 in both | Ollama ships the thinking-mode set | huggingface.co/Qwen/Qwen3.8-27B |
| Hemmingway-1 | not stated (config 1.0) | (0.95 in config) | (20 in config) | – | – | Falls back to Qwen3.8 | huggingface.co/HemmingwayAI/Hemmingway-1 |
| Hemmingway-1.5 | 1.0 | (0.95 in config) | (20 in config) | – | – | "Lower temperatures make its writing easier to detect." Any thinking level | huggingface.co/HemmingwayAI/Hemmingway-1.5 |
| Writer-V2 (Qwen3.5-27B) | 0.7 | 0.95 | – | – | DRY ~0.4–0.8 | **DRY isn't in Ollama**. Empty think prefill | huggingface.co/ConicCat/Qwen3.5-27B-Writer-V2 |
| Marvin-DPO-V2 (Qwen3.5-27B) | 0.8 | 0.9 | – | – | – | Empty think prefill for non-thinking | huggingface.co/ToastyPigeon/Qwen3.5-27B-Marvin-DPO-V2 |
| Musica (Qwen3.5-27B) | 1.0 | – | – | 0.02 | – | Also top-nσ 2, **not in Ollama** | huggingface.co/AuriAetherwiing/TQ3.5-27B-Musica-v1 |
| Gemsicle-31B | 1.0 | 0.90 | 0 (off) | 0.03 | – | Values for use without Adaptive-P, which **isn't in Ollama** | huggingface.co/Blazed-Forge/Gemma-4-Gemsicle-31B |
| Nemotron 3.5 Lightning | 1.0 | 0.95 | – | – | – | Thinking on by default | huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 |
| Hermes 4.3 36B | 0.6 | 0.95 | 20 | – | – | One set for both modes | huggingface.co/NousResearch/Hermes-4.3-36B |
| Seed-OSS-36B | 1.1 | 0.95 | – | – | – | "We recommend sampling with temperature=1.1 and top_p=0.95" | huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct |

"–" means not set. Where Ollama has an "off" value, use it: top_k 0, top_p 1.0, min_p 0.

## Is there an obvious shared setting?

No. The evidence is unsettled.

- **Min-p paper** (Nguyen et al., ICLR 2025, arXiv 2407.01082): recommends min_p 0.05–0.1 as the only cutoff, with gains mostly at high temperature.
- **Critique** (Schaeffer et al., 2025, arXiv 2506.13681): finds errors in the paper's analysis. Its own human study shows no advantage for min-p, and it concludes that samplers perform roughly the same once each is tuned.
- **EQ-Bench Creative Writing** uses temp 0.7 and min_p 0.1 for every model. It's the only published cross-model protocol found. It's a benchmark convention, not evidence that the setting is best.
- **Peeperkorn et al.** (ICCC 2024, arXiv 2405.00492): temperature is only weakly linked to novelty in stories.
- **No study** compares samplers for poetry across models.
- **Community presets** (weak evidence): three fine-tune authors on this list use temp 1.0 with min_p 0.02–0.10.

Untested: whether min_p trims the rare tokens that IAST diacritics need. Ollama's sampler order isn't verified either.
