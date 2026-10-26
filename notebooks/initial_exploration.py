import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mlflow
    import tomllib
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig, MetalConfig, set_seed
    from datetime import datetime

    # maybe add some openrouter models soon, see what the smartest models can do...
    return (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        GenerationConfig,
        MetalConfig,
        datetime,
        mlflow,
        set_seed,
        tomllib,
    )


@app.cell
def _(tomllib):
    with open("src/model_config.toml", "rb") as m:
        model_info = tomllib.load(m)
    print(model_info)
    return (model_info,)


@app.cell
def _(mlflow, model_info, set_seed):
    set_seed(model_info["default"]["seed"])
    SYSTEM_PROMPT = mlflow.genai.load_prompt("prompts:/system_prompt/1").template
    POEM_GEN_PROMPT = mlflow.genai.load_prompt("prompts:/poem_generation_prompt/1").template
    MAX_NEW_TOKENS = model_info["default"]["max_new_tokens"]
    # NUM_POEMS = 1 # number of poems generated per model, per config
    return MAX_NEW_TOKENS, POEM_GEN_PROMPT, SYSTEM_PROMPT


@app.cell
def _(model_info):
    # temporarily just testing the logic with Hermes
    model_info["llm"] = model_info["llm"][:1]
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    MAX_NEW_TOKENS,
    MetalConfig,
    POEM_GEN_PROMPT,
    SEED,
    SYSTEM_PROMPT,
    datetime,
    mlflow,
    model_info,
):
    for llm in model_info["llm"]:
        MODEL_NAME = llm["name"]
        commit_hash = llm["commit_hash"]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if llm["quantization"]["quantize"]: # not sure how long this will take, and if the process must be repeated every time the script runs... maybe I should write a helper script to quantize the flagged models and save that version's weights locally, logged in MLFlow. Tbh, I should probably save all my model weights at least once... and only resave (non-destructive) when checksums don't match
            bits = llm["quantization"]["bits"]
            group_size = llm["quantization"]["group_size"]
            quantization_config = MetalConfig(bits=bits, group_size=group_size)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="mps", # eventually want this to run on Linux out of the box too
                quantization_config=quantization_config,
                revision=commit_hash
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="mps", revision=commit_hash)
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

        model_config = GenerationConfig.from_pretrained(MODEL_NAME).to_dict()
        default_config = GenerationConfig._get_default_generation_params()

        for k,v in model_config.items():
            if v == None and k in default_config:
                model_config[k] = default_config[k]
        model_config["max_new_tokens"] = MAX_NEW_TOKENS

        print(model_config)
        print(model.device)
        print("precision:", model.dtype)

        my_params = {"seed": SEED, "device": model.device, "precision": model.dtype, "commit_hash": commit_hash} # needs: reasoning setting... Will also need to track reasoning and output tokens separately.

        mlflow.set_tracking_uri('http://localhost:5001')
        mlflow.set_experiment("initial-exploration-poem-gen")
        dt = str(datetime.now().strftime("%Y%m%d%H%M%S"))
        with mlflow.start_run(run_name=MODEL_NAME.replace("/", "_") + "_"+ dt) as run: # this can later become a function callable per model... or, how can we enable parallel runs in mlflow? (Gated by available memory... we can prob run 3 8B models at once, for ex, so needs to be dynamic) ADD RUN NAME, later also DATE
            mlflow.log_params(model_config | my_params)
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS) # default suggested max_new_tokens=256
            poem = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
            mlflow.log_text(poem, artifact_file="outputs/poem.txt")

        print(poem)
    return


@app.cell
def _():
    """ Params that still need to be logged:
    quantization, reasoning on/off, model ID, revision/commit hash, num tokens generated, generation time, whether it stopped at the limit (generated tokens == max_length or max_new_tokens)

    Portability: for now, quantization is Metal-based; later plan to make it platform-agnostic
    e.g. Hermes 14 B needs a MetalConfig(bits=4, group_size=64) passed as quantization_config= to from_pretrained, with device_map="mps". https://huggingface.co/docs/transformers/v5.20.2/en/quantization/metal has a short generic example.
    Since not every model needs to be quantized, that should probably go in src/model_config.toml, per model.
    """
    return


@app.cell
def _():
    # with ______ (set run name based on current date/time)
    #      create a sub-run for a model (named base on model name) with desired params, GGUF format for speed
    #      generate NUM_POEMS_EACH responses to the prompt (no parallelism yet... when I get there, I think only 2-3 small models can fit on my mac VRAM at once)
    return


if __name__ == "__main__":
    app.run()
