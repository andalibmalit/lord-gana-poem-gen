from __future__ import annotations

import pytest

from poemgen import records
from poemgen.config import ConfigError, load_config, load_experiment
from poemgen.plan import build_plan, estimate_minutes


def plan_for(cfg, **kw):
    return build_plan(cfg, load_experiment(cfg, "exp"), **kw)


def spec(plan, model, sset, system="a", sample=0):
    return next(
        s
        for s in plan.specs
        if (s.model.key, s.settings_set, s.system.variant, s.sample)
        == (model, sset, system, sample)
    )


def test_counts_and_skips(cfg):
    plan = plan_for(cfg)
    # small: 2 sets × 2 prompts × 2 samples. plain has no thinking: "hot" (think=true) is skipped.
    assert len(plan.specs) == 8 + 4
    assert any("plain × hot" in s for s in plan.skipped)
    assert len(plan.pending) == len(plan.specs) and not plan.done


def test_layering_and_origins(cfg):
    plan = plan_for(cfg)
    base = spec(plan, "small", "base")
    assert base.think is False
    assert base.options["temperature"] == 0.7 and base.origins["temperature"] == "recommended"
    assert base.options["min_p"] == 0.0 and base.origins["min_p"] == "default"
    assert base.options["stop"] == ["<end>"] and base.origins["stop"] == "model"
    assert base.options["seed"] == 100 and base.origins["seed"] == "experiment"
    assert base.not_applied == {"dry": "DRY sampler"}
    # Every option is sent, so Ollama's own defaults never apply.
    assert set(cfg.defaults) <= set(base.options)

    hot = spec(plan, "small", "hot")
    assert hot.think is True
    assert hot.options["top_p"] == 0.95 and hot.origins["top_p"] == "recommended:thinking"
    assert hot.options["temperature"] == 1.3 and hot.origins["temperature"] == "change:hot"
    assert hot.options["min_p"] == 0.05 and hot.origins["min_p"] == "change:hot"

    plain = spec(plan, "plain", "base")
    assert plain.options["temperature"] == 1.0 and plain.origins["temperature"] == "recommended"
    assert plain.options["top_p"] == 0.9 and plain.origins["top_p"] == "inherited:base model"


def test_num_ctx_caps(cfg):
    plan = plan_for(cfg)
    small = spec(plan, "small", "base")
    assert small.options["num_ctx"] == 32768 and small.origins["num_ctx"] == "capped:ctx_max"
    plain = spec(plan, "plain", "base")
    assert plain.options["num_ctx"] == 8192 and plain.origins["num_ctx"] == "capped:machine"
    assert any("capped" in w for w in plan.warnings)


def test_seeds_shared_across_settings_and_distinct_across_samples(cfg):
    plan = plan_for(cfg)
    assert spec(plan, "small", "base", sample=1).options["seed"] == 101
    assert spec(plan, "small", "hot", sample=1).options["seed"] == 101
    ids = [s.poem_id for s in plan.specs]
    assert len(ids) == len(set(ids))


def test_ids_are_deterministic_and_change_with_config(cfg, repo):
    first = {s.poem_id for s in plan_for(cfg).specs}
    assert first == {s.poem_id for s in plan_for(load_config(repo)).specs}
    (repo / "prompts" / "system" / "b.md").write_text("Prompt B, edited.\n", encoding="utf-8")
    second = {s.poem_id for s in plan_for(load_config(repo)).specs}
    assert len(first & second) == 6  # prompt-a poems are unchanged; prompt-b poems are new


def test_done_and_orphans(cfg):
    plan = plan_for(cfg)
    exp_dir = cfg.poems_dir / "exp"
    done_id = plan.specs[0].poem_id
    records.write_poem(exp_dir / f"{done_id}.md", {"poem_id": done_id}, "A poem.", None)
    records.write_poem(exp_dir / "oldoldoldold.md", {"poem_id": "oldoldoldold"}, "Old.", None)
    again = plan_for(cfg)
    assert again.done == {done_id}
    assert again.orphans == ["oldoldoldold"]
    assert len(again.pending) == len(plan.specs) - 1


def test_model_subset(cfg):
    plan = plan_for(cfg, only_models=["plain"])
    assert {s.model.key for s in plan.specs} == {"plain"}
    with pytest.raises(ValueError):
        plan_for(cfg, only_models=["nope"])


def test_estimate_uses_history_then_machine_default(cfg):
    plan = plan_for(cfg, only_models=["plain"])
    assert estimate_minutes(cfg, plan.pending) == {"plain": 4 * 2.0}
    meta = {
        "model": {"key": "plain"},
        "request": {"think": False},
        "response": {"total_duration": 60e9 * 5},
    }
    records.write_poem(cfg.poems_dir / "other" / "x.md", meta, "Poem.", None)
    assert estimate_minutes(cfg, plan.pending) == {"plain": 4 * 5.0}


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ("temprature = 1.0", "unknown or unsettable option"),
        ("top_k = 2.5", "must be an integer"),
        ("seed = 1", "unknown or unsettable option"),
    ],
)
def test_bad_options_fail_at_load(repo, bad, message):
    path = repo / "config" / "models.toml"
    text = path.read_text(encoding="utf-8").replace(
        "recommended = { temperature = 1.0 }", f"recommended = {{ {bad} }}"
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(repo)


def test_defaults_must_set_every_option(repo):
    path = repo / "config" / "defaults.toml"
    path.write_text("temperature = 1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing"):
        load_config(repo)


def test_experiment_rejects_unknown_model(cfg, repo):
    (repo / "experiments" / "bad.toml").write_text(
        'models = ["ghost"]\nsystem_prompts = ["a"]\nuser_prompt = "u"\nsamples = 1\nseed = 1\n'
        '[[settings]]\nname = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown models"):
        load_experiment(cfg, "bad")


def test_real_config_loads():
    """The committed config files are valid and the smoke experiment plans."""
    cfg = load_config()
    plan = build_plan(cfg, load_experiment(cfg, "smoke"))
    assert len(plan.specs) == 2


def test_tags_without_a_ref_compare_as_latest(repo):
    path = repo / "config" / "models.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace('tag = "plain:q4"', 'tag = "hf.co/org/repo-GGUF"'),
        encoding="utf-8",
    )
    cfg = load_config(repo)
    assert cfg.models["plain"].tag == "hf.co/org/repo-GGUF:latest"
    assert cfg.models["small"].tag == "small:q8"


def test_experiment_name_beats_a_directory_of_the_same_name(cfg, monkeypatch, tmp_path):
    (tmp_path / "elsewhere" / "exp").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert load_experiment(cfg, "exp").name == "exp"
