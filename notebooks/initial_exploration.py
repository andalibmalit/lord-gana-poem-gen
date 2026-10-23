import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mlflow
    import tomllib
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig, set_seed
    from datetime import datetime

    return (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        GenerationConfig,
        datetime,
        mlflow,
        set_seed,
    )


@app.cell
def _(mlflow, set_seed):
    SEED = 333997
    set_seed(SEED)
    SYSTEM_PROMPT = mlflow.genai.load_prompt("prompts:/system_prompt/1").template
    POEM_GEN_PROMPT = mlflow.genai.load_prompt("prompts:/poem_generation_prompt/1").template
    MODEL_NAME = "google/gemma-4-E4B-it" # Next step - load q4 hermes!
    MAX_NEW_TOKENS = 3333 # later will be loaded from src/model_config.toml
    # NUM_POEMS = 1 # number of poems generated per model, per config
    return MAX_NEW_TOKENS, MODEL_NAME, POEM_GEN_PROMPT, SEED, SYSTEM_PROMPT


@app.cell
def _(POEM_GEN_PROMPT):
    print(POEM_GEN_PROMPT)
    return


@app.cell
def _():
    """with open("src/model_config.toml", "rb") as master_config:
        model_list = tomllib.load(master_config)["llm"]
    print(model_list)

    for m in model_list:
        print(GenerationConfig.from_pretrained(m["name"]))"""
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    MODEL_NAME,
    POEM_GEN_PROMPT,
    SYSTEM_PROMPT,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="mps")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": POEM_GEN_PROMPT}
    ]
    inputs = tokenizer.apply_chat_template(
    	messages,
    	add_generation_prompt=True,
    	tokenize=True,
    	return_dict=True,
    	return_tensors="pt",
    ).to(model.device)
    return inputs, model, tokenizer


@app.cell
def _(GenerationConfig, MAX_NEW_TOKENS, MODEL_NAME, model):
    model_config = GenerationConfig.from_pretrained(MODEL_NAME).to_dict()
    default_config = GenerationConfig._get_default_generation_params()

    for k,v in model_config.items():
        if v == None and k in default_config:
            model_config[k] = default_config[k]
    model_config["max_new_tokens"] = MAX_NEW_TOKENS

    print(model_config)
    print(model.device)
    return (model_config,)


@app.cell
def _():
    # print(model_config["max_length"], model_config["max_new_tokens"])
    # print(model.dtype)
    return


@app.cell
def _(
    MAX_NEW_TOKENS,
    MODEL_NAME,
    SEED,
    datetime,
    inputs,
    mlflow,
    model,
    model_config,
    tokenizer,
):
    """ Params logged:
    temperature, top_p, top_k, max_length, max_new_tokens, do_sample, repetition_penalty, no_repeat_ngram_size (the ones that are always set, last two are neutral by default); min_p, top_h, typical_p, epsilon_cutoff, eta_cutoff (the ones that I think are worth experimenting with beyond the first 5, to be changed one at a time to start, esp the last 4); anything else in a model's specific GenerationConfig (so nothing is silently changed; anything beyond these first 3 sets of params should be either unset or a neutral value); seed=333997, precision, quantization, reasoning on/off, system prompt (in case anything gets appended automatically); model ID, revision/commit hash, device (mps/cpu/cuda); user prompt; num tokens generated, generation time, whether it stopped at the limit (generated tokens == max_length or max_new_tokens)

    Artifacts: the poem generated

    Prompt registry: system prompt (if this also captures the prompt with any inference-time additions, I can remove it from the params logged list)

    Portability: for now, quantization is Metal-based; later plan to make it platform-agnostic
    e.g. Hermes 14 B needs a MetalConfig(bits=4, group_size=64) passed as quantization_config= to from_pretrained, with device_map="mps". https://huggingface.co/docs/transformers/v5.20.2/en/quantization/metal has a short generic example.
    Since not every model needs to be quantized, that should probably go in src/model_config.toml, per model.
    """
    my_params = {"seed": SEED, "device": model.device, "precision": model.dtype} # needs: reasoning setting, model revision, and quantization level (quantization is Metal-only to start)... maybe all of these can go in src/model_config.toml, I think format varies. Will also need to track reasoning and output tokens separately.

    mlflow.set_experiment("initial-exploration-poem-gen")
    mlflow.set_tracking_uri('http://localhost:5001')
    dt = str(datetime.now().strftime("%Y%m%d%H%M%S"))
    with mlflow.start_run(run_name=MODEL_NAME.replace("/", "_") + dt) as run: # this can later become a function callable per model... or, how can we enable parallel runs in mlflow? (Gated by available memory... we can prob run 3 8B models at once, for ex, so needs to be dynamic) ADD RUN NAME, later also DATE
        mlflow.log_params(model_config | my_params)
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS) # default suggested max_new_tokens=256
        poem = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        mlflow.log_text(poem, artifact_file="outputs/poem.txt")
    return (poem,)


@app.cell
def _():
    # with ______ (set run name based on current date/time)
    #      create a sub-run for a model (named base on model name) with desired params, GGUF format for speed
    #      generate NUM_POEMS_EACH responses to the prompt (no parallelism yet... when I get there, I think only 2-3 small models can fit on my mac VRAM at once)
    return


@app.cell
def _():
    # outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)  # default suggested max_new_tokens=256
    return


@app.cell
def _(poem):
    print(poem)
    return


if __name__ == "__main__":
    app.run()
