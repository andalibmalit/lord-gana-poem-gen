import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mlflow

    return


@app.cell
def _():
    #mlflow.set_experiment("initial-exploration-poem-gen")
    print("Hello from my notebook!")
    return


if __name__ == "__main__":
    app.run()
