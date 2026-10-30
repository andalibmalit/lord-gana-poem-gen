"""Checks against the real local Ollama. Opt-in: `uv run pytest -m ollama`.

For every configured model that's pulled, render the real prompts with each `think`
value the config allows (no generation: Ollama's render-only flag) and check that the
template accepts it and that the system prompt reaches the model. Run it after adding
or re-pulling a model.
"""

from __future__ import annotations

import pytest

from poemgen.config import load_config
from poemgen.ollama import Ollama, OllamaError
from poemgen.plan import load_prompt
from poemgen.run import preflight

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def live():
    cfg = load_config()
    client = Ollama(cfg.machine.ollama_host, timeout_s=300)
    try:
        local = client.tags()
    except OllamaError as e:
        pytest.skip(f"Ollama not reachable: {e}")
    yield cfg, client, local
    client.close()


def pulled(live):
    cfg, _, local = live
    models = [m for m in cfg.models.values() if m.confirmed and m.tag in local]
    if not models:
        pytest.skip("no configured model is pulled")
    return models


def test_think_values_render_and_keep_the_system_prompt(live):
    cfg, client, _ = live
    system = load_prompt(cfg.system_prompt_path("1st-p"), "1st-p")
    user = load_prompt(cfg.user_prompt_path("poem_generation"), "poem_generation")
    problems = []
    for model in pulled(live):
        for think in model.think_values or (False,):
            body = {
                "model": model.tag,
                "messages": [
                    {"role": "system", "content": system.text},
                    {"role": "user", "content": user.text},
                ],
                "think": think,
                "stream": False,
            }
            rendered = client.render(body)
            if rendered is None:
                problems.append(f"{model.key} think={think!r}: template failed to render")
            elif system.text.strip() not in rendered:
                problems.append(f"{model.key} think={think!r}: system prompt changed or missing")
    assert not problems, "\n".join(problems)


def test_pulled_builds_match_config(live):
    cfg, client, local = live
    checked = [m for m in pulled(live) if m.ollama_digest]
    for model in checked:
        preflight(cfg, client, model, local)  # raises Refused on any mismatch
