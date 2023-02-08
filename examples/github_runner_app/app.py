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
    script_args,
    output_dirs,
    private,
    delete_space_on_completion,
    downgrade_hardware_on_completion,
    space_hardware,
):
    if not token.strip():
        return gr.update(
            value="""## token with write access is required. Get one from <a href="https://hf.co/settings/tokens" target="_blank">here</a>""",
            visible=True,
        )

    if script_args.strip():
        script_args = yaml.safe_load(script_args)

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
        return gr.update(value="## GitHub repo ID is required", visible=True)

    script = script.strip()
    if not script:
        return gr.update(value="## script is required", visible=True)

    github_repo_branch = github_repo_branch.strip()
    if not github_repo_branch:
        return gr.update("## github repo branch is required", visible=True)

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
        extra_requirements=extra_requirements,
        token=token,
        **script_args,
    )
    output_message = f"""
    ## Job launched successfully! 🚀
       - <a href="{space_url}" target="_blank">Link to Space</a>
       - <a href="{dataset_url}" target="_blank">Link to Dataset</a>
    """
    return gr.update(value=output_message, visible=True)


description = """
This app lets you run scripts from GitHub on Spaces, using any hardware you'd like. Just point to a repo, the script you'd like to run, the dependencies to install, and any args to pass to your script, and watch it go. 😎

It uses 🔥[fuego](https://github.com/huggingface/fuego)🔥 under the hood to launch your script in one line of Python code. Give the repo a ⭐️ if you think its 🔥.

**Note: You'll need a Hugging Face token with write access, which you can get from [here](https://hf.co/settings/tokens)**
"""

additional_info = """
## Pricing

Runs using this tool are **free** as long as you use `cpu-basic` hardware. 🔥

**See pricing for accelerated hardware (anything other than `cpu-basic`) [here](https://hf.co/pricing#spaces)**

## What this space does:
  1. Spins up 2 new HF repos for you: a "runner" space repo and an "output" dataset repo.
  2. Uploads your code to the space, as well as some wrapper code that invokes your script.
  3. Runs your code on the space via the wrapper. Logs should show up in the space.
  4. When the script is done, it takes anything saved to the `output_dirs` and uploads the files within to the output dataset repo
  5. Deletes the space (or downgrades, or just leaves on). Depends on your choice of `delete_space_on_completion` and `downgrade_hardware_on_completion`.

## FAQ

- If your space ends up having a "no application file" issue, you may need to "factory reset" the space. You can do this from the settings page of the space.
"""

output_message = gr.Markdown("", visible=False)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown("# 🔥Fuego🔥 GitHub Script Runner")
    gr.Markdown(description)
    with gr.Accordion("👀 More Details (Hardware Pricing, How it Works, and FAQ)", open=False):
        gr.Markdown(additional_info)

    with gr.Row():
        token = gr.Textbox(lines=1, label="Hugging Face token with write access", type="password")

    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown("What script would you like to run? Also, what are its dependencies?")
                github_repo_id = gr.Textbox(lines=1, label="GitHub repo ID (ex. huggingface/fuego)")
                github_repo_branch = gr.Textbox(
                    lines=1, label="Branch of GitHub repo (ex. main)", value="main", interactive=True
                )
                script = gr.Textbox(lines=1, label="Path to python script in the GitHub repo")
                requirements_file = gr.Textbox(lines=1, label="Path to pip requirements file in the repo")
                extra_requirements = gr.Textbox(
                    lines=5,
                    label="Any extra pip requirements to your script, just as you would write them in requirements.txt",
                )
        with gr.Column():
            with gr.Box():
                gr.Markdown("How should we run your script?")
                script_args = gr.Textbox(lines=10, label="Script args to your python file. Input here as YAML.")
                spaces_output_dirs = gr.Textbox(
                    lines=1,
                    label="Name of output directory to save assets to from within your script. Use commas if you have multiple.",
                    value="./outputs, ./logs",
                )
                private = gr.Checkbox(False, label="Should space/dataset be made as private repos?")
                delete_space_on_completion = gr.Checkbox(True, label="Delete the space on completion?")
                downgrade_hardware_on_completion = gr.Checkbox(
                    True,
                    label="Downgrade hardware of the space on completion? Only applicable if not deleting on completion.",
                )
    with gr.Row():
        with gr.Column():
            spaces_hardware = gr.Dropdown(
                ["cpu-basic", "cpu-upgrade", "t4-small", "t4-medium", "a10g-small", "a10g-large", "a100-large"],
                label="Spaces Hardware",
                value="cpu-basic",
                interactive=True,
            )
            spaces_hardware_msg = gr.Markdown(
                """
                🔴 **The hardware you chose is not free, and you will be charged for it** 🔴

                If you want to run your script for free, please choose `cpu-basic` as your hardware.
            """,
                visible=False,
            )
            spaces_hardware.change(
                lambda x: gr.update(visible=True) if x != "cpu-basic" else gr.update(visible=False),
                inputs=[spaces_hardware],
                outputs=[spaces_hardware_msg],
            )

    with gr.Row():
        with gr.Accordion("👀 Examples", open=False):
            gr.Examples(
                [
                    [
                        "pytorch/examples",
                        "main",
                        "vae/main.py",
                        "vae/requirements.txt",
                        "",
                        "epochs: 3",
                        "./results",
                        False,
                        True,
                        True,
                        "cpu-basic",
                    ],
                    [
                        "huggingface/transformers",
                        "main",
                        "examples/pytorch/text-classification/run_glue.py",
                        "examples/pytorch/text-classification/requirements.txt",
                        "tensorboard\ngit+https://github.com/huggingface/transformers@main#egg=transformers",
                        "model_name_or_path: bert-base-cased\ntask_name: mrpc\ndo_train: True\ndo_eval: True\nmax_seq_length: 128\nper_device_train_batch_size: 32\nlearning_rate: 2e-5\nnum_train_epochs: 3\noutput_dir: ./outputs\nlogging_dir: ./logs\nlogging_steps: 20\nreport_to: tensorboard",
                        "./outputs,./logs",
                        False,
                        True,
                        True,
                        "cpu-basic",
                    ],
                ],
                inputs=[
                    github_repo_id,
                    github_repo_branch,
                    script,
                    requirements_file,
                    extra_requirements,
                    script_args,
                    spaces_output_dirs,
                    private,
                    delete_space_on_completion,
                    downgrade_hardware_on_completion,
                    spaces_hardware,
                ],
                outputs=[
                    github_repo_id,
                    github_repo_branch,
                    script,
                    requirements_file,
                    extra_requirements,
                    script_args,
                    spaces_output_dirs,
                    private,
                    delete_space_on_completion,
                    downgrade_hardware_on_completion,
                    spaces_hardware,
                ],
                cache_examples=False,
            )

    with gr.Row():
        submit = gr.Button("Submit")
        reset_btn = gr.Button("Reset fields")

    with gr.Row():
        output_message.render()

    submit.click(
        fuego_github_run_wrapper,
        inputs=[
            token,
            github_repo_id,
            github_repo_branch,
            script,
            requirements_file,
            extra_requirements,
            script_args,
            spaces_output_dirs,
            private,
            delete_space_on_completion,
            downgrade_hardware_on_completion,
            spaces_hardware,
        ],
        outputs=[output_message],
    )

    def reset_fields():
        return {
            output_message: gr.update(value="", visible=False),
            github_repo_id: gr.update(value=""),
            github_repo_branch: gr.update(value="main"),
            script: gr.update(value=""),
            requirements_file: gr.update(value=""),
            extra_requirements: gr.update(value=""),
            script_args: gr.update(value=""),
            spaces_output_dirs: gr.update(value="./outputs, ./logs"),
            private: gr.update(value=False),
            delete_space_on_completion: gr.update(value=True),
            downgrade_hardware_on_completion: gr.update(value=True),
            spaces_hardware: gr.update(value="cpu-basic"),
        }

    reset_btn.click(
        reset_fields,
        outputs=[
            output_message,
            github_repo_id,
            github_repo_branch,
            script,
            requirements_file,
            extra_requirements,
            script_args,
            spaces_output_dirs,
            private,
            delete_space_on_completion,
            downgrade_hardware_on_completion,
            spaces_hardware,
        ],
    )

if __name__ == "__main__":
    demo.launch(debug=True)
