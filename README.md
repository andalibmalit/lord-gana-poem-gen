# README
For now, this repo does the following:
1. Store a system prompt used with a variety of LLMs to generate English-language devotional poems addressed to Lord Gana.
2. Store the list of models being explored and their configuration settings (temperature, reasoning on/off). Models can be run locally or remotely.
3. Perform generation sweeps using the system prompt across the models and config settings, in parallel if memory allows.
4. Store the parameters and results of each sweep in a local MLFlow server.
5. Export the results from MLFlow for future analysis and publication/sharing.

Libraries used: Marimo, Hugging Face transformers, OpenRouter (for remote models, later), MLFlow for storage, polars for dataframe analysis

**Current entry point:** notebooks/initial_exploration.py

I should probably also compare translations/OCR done by other LLMs besides Claude, and do some NLP analysis for faithfulness. E.g. low difference between the two.

Prompt for generation: "Please write an original, English-language devotional poem addressed to Lord Gana."

## On the system prompt
In addition to the various translation options, experiment with "I" vs "You" in the first part