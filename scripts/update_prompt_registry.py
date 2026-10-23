"""Update MLFlow prompt registry given a prompt and its metadata. Create a new prompt if it does not exist."""

import mlflow
from argparse import ArgumentParser


# Register a new prompt
def register_prompt(template, commit_message, task, author, language="en"):
    prompt = mlflow.genai.register_prompt(
        name="system-prompt",
        template=template,
        # Optional: Provide a commit message to describe the changes
        commit_message=commit_message,
        # Optional: Specify tags for this prompt
        tags={
            "author": author,
            "task": task,
            "language": language,
        },
    )
    # The prompt object contains information about the registered prompt
    print(f"Created prompt '{prompt.name}' (version {prompt.version})")


parser = ArgumentParser()
# ... later