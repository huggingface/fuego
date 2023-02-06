# Gradio app to run fuego.github_run() on Hugging Face Spaces
# Hosted at https://hf.co/nateraw/fuego
import gradio as gr
import yaml

import fuego


def fuego_github_run_wrapper(
    token,
    github_repo_id,
    github_repo_branch,
    script,
    requirements_file,
    extra_requirements,
    output_dirs,
    script_args,
    space_hardware,
    private,
    delete_space_on_completion,
    downgrade_hardware_on_completion,
    extra_run_metadata,
):
    if not token.strip():
        return "token with write access is required. Get one from https://hf.co/settings/tokens", "", ""
    if script_args.strip():
        script_args = yaml.safe_load(script_args)
    if extra_run_metadata.strip():
        extra_run_metadata = yaml.safe_load(extra_run_metadata)

    if not requirements_file.strip():
        requirements_file = None

    if extra_requirements.strip():
        extra_requirements = [x.strip() for x in extra_requirements.split("\n")]
    else:
        extra_requirements = None

    if output_dirs.strip():
        output_dirs = [x.strip() for x in output_dirs.split(",")]

    github_repo_id = github_repo_id.strip()
    if not github_repo_id:
        return "GitHub repo ID is required", "", ""

    script = script.strip()
    if not script:
        return "script is required", "", ""

    github_repo_branch = github_repo_branch.strip()
    if not github_repo_branch:
        return "github repo branch is required", "", ""

    space_url, dataset_url = fuego.github_run(
        github_repo_id.strip(),
        script.strip(),
        requirements_file,
        github_repo_branch,
        space_hardware=space_hardware,
        private=private,
        delete_space_on_completion=delete_space_on_completion,
        downgrade_hardware_on_completion=downgrade_hardware_on_completion,
        space_output_dirs=output_dirs,
        extra_run_metadata=extra_run_metadata,
        extra_requirements=extra_requirements,
        token=token,
        **script_args,
    )
    return "Launched Successfully!", space_url, dataset_url


examples = [
    [
        "",
        "pytorch/examples",
        "main",
        "vae/main.py",
        "vae/requirements.txt",
        "",
        "./results",
        "epochs: 3",
        "cpu-basic",
        False,
        True,
        True,
        "",
    ],
    [
        "",
        "huggingface/transformers",
        "main",
        "examples/pytorch/text-classification/run_glue.py",
        "examples/pytorch/text-classification/requirements.txt",
        "tensorboard\ngit+https://github.com/huggingface/transformers@main#egg=transformers",
        "./outputs,./logs",
        "model_name_or_path: bert-base-cased\ntask_name: mrpc\ndo_train: True\ndo_eval: True\nmax_seq_length: 128\nper_device_train_batch_size: 32\nlearning_rate: 2e-5\nnum_train_epochs: 3\noutput_dir: ./outputs\nlogging_dir: ./logs\nlogging_steps: 20\nreport_to: tensorboard",
        "cpu-basic",
        False,
        True,
        True,
        "",
    ],
]
description = """
This app lets you run scripts from GitHub on Spaces, using any hardware you'd like. Just point to a repo, the script you'd like to run, the dependencies to install, and any args to pass to your script, and watch it go. 😎

It uses 🔥[fuego](https://github.com/huggingface/fuego)🔥 under the hood to launch your script in one line of Python code. Give the repo a ⭐️ if you think its 🔥.

**Note: You'll need a Hugging Face token with write access, which you can get from [here](https://hf.co/settings/tokens)**

## Pricing

Runs using this tool are **free** as long as you use `cpu-basic` hardware. 🔥

**See pricing for accelerated hardware (anything other than `cpu-basic`) [here](https://hf.co/pricing#spaces)**

## What this space does:
  1. Spins up 2 new HF repos for you: a "runner" space repo and an "output" dataset repo.
  2. Uploads your code to the space, as well as some wrapper code that invokes your script.
  3. Runs your code on the space via the wrapper. Logs should show up in the space.
  4. When the script is done, it takes anything saved to the `output_dirs` and uploads the files within to the output dataset repo
  5. Deletes the space (or downgrades, or just leaves on). Depends on your choice of `delete_space_on_completion` and `downgrade_hardware_on_completion`.

## Notes

- If your space ends up having a "no application file" issue, you may need to "factory reset" the space. You can do this from the settings page of the space.
"""

interface = gr.Interface(
    fuego_github_run_wrapper,
    inputs=[
        gr.Textbox(lines=1, placeholder="Hugging Face token with write access", type="password"),
        gr.Textbox(lines=1, placeholder="Source code GitHub repo ID (ex. huggingface/fuego)"),
        gr.Textbox(lines=1, placeholder="Branch of GitHub repo (ex. main)", value="main"),
        gr.Textbox(lines=1, placeholder="Path to python script in the GitHub repo"),
        gr.Textbox(lines=1, placeholder="Path to pip requirements file in the repo"),
        gr.Textbox(
            lines=5,
            placeholder="Any extra pip requirements to your script, just as you would write them in requirements.txt",
        ),
        gr.Textbox(
            lines=1,
            placeholder="Name of output directory to save assets to from within your script. Use commas if you have multiple.",
            value="./outputs, ./logs",
        ),
        gr.Textbox(lines=10, placeholder="Script args to your python file. Input here as YAML."),
        gr.Dropdown(
            ["cpu-basic", "cpu-upgrade", "t4-small", "t4-medium", "a10g-small", "a10g-large", "a100-large"],
            label="Spaces Hardware",
            value="cpu-basic",
        ),
        gr.Checkbox(False, label="Should space/dataset be made as private repos?"),
        gr.Checkbox(True, label="Delete the space on completion?"),
        gr.Checkbox(
            True, label="Downgrade hardware of the space on completion? Only applicable if not deleting on completion."
        ),
        gr.Textbox(
            lines=5,
            placeholder="Any extra metadata (input as YAML) you would like to store within the run's metadata (found in dataset card).",
        ),
    ],
    outputs=[
        gr.Textbox(label="Message"),
        gr.Textbox(label="Runner Space URL"),
        gr.Textbox(label="Output Dataset URL"),
    ],
    title="🔥Fuego🔥 GitHub Script Runner",
    description=description,
    examples=examples,
    cache_examples=False,
).launch()
