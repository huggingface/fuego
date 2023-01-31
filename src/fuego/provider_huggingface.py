import json
import os
import tempfile
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfFolder, add_space_secret, create_repo, upload_file, upload_folder, whoami
from huggingface_hub.repocard import RepoCard

from .provider_utils import Provider


HUGGINGFACE_COMPUTE_TARGETS_MAP = {
    "cpu": "cpu-basic",
    "cpu-upgrade": "cpu-upgrade",
    "t4-small": "t4-small",
    "t4-medium": "t4-medium",
    "a10g-small": "a10g-small",
    "a10g-large": "a10g-large",
    "a100-large": "a100-large",
}

_task_run_script_template = """import os
from pathlib import Path
from threading import Thread
import gradio as gr
from huggingface_hub import upload_folder, HfFolder, delete_repo
import subprocess
import sys

from {script} import main

HfFolder().save_token(os.getenv("HF_TOKEN"))
output_dataset_id = "{output_dataset_id}"

this_space_repo_id = "{space_id}"
delete_space_on_completion = {delete_space_on_completion}

output_dir = "{output_dir}"
Path(output_dir).mkdir(exist_ok=True, parents=True)

script_args = {script_args}
script_args_lst = {script_args_lst}
unpack_script_args_to_main = {unpack_script_args_to_main}

def main_wrapper():
    print('-' * 80)
    print("Starting...")
    print('-' * 80)

    # Do the work
    # main()
    # main(**script_args)
    if not unpack_script_args_to_main:
        sys.argv = ["{script}"] + script_args_lst
        main()
    else:
        main(**script_args)

    print('-' * 80)
    print("Done Running!")
    print('-' * 80)

    # Save the work
    print("Uploading outputs to dataset repo")
    upload_folder(
        repo_id=output_dataset_id,
        folder_path=output_dir,
        path_in_repo='./outputs',
        repo_type='dataset',
    )

    # Delete self.
    if delete_space_on_completion:
        delete_repo(this_space_repo_id, repo_type="space")

with gr.Blocks() as demo:
    gr.Markdown(Path('about.md').read_text())

thread = Thread(target=main_wrapper, daemon=True)
thread.start()
demo.launch()
"""

_about_md_content = """
# Task Runner

This space is running some job!

- Check out the associated [output repo]({output_repo_url})
"""


class HuggingFaceProvider(Provider):
    # TODO - rethink init logic for providers so signature here is always the same.
    # Maybe we automate configuration process across clouds and save/load from some cache file?
    def __init__(self, token=None):
        token = token or os.getenv("HF_TOKEN", HfFolder().get_token())
        if token is None:
            raise RuntimeError(
                "Unable to find HuggingFace token. Please set HF_TOKEN env variable or log in with `huggingface-cli login`"
            )

        if HfFolder().get_token() != token:
            HfFolder().save_token(token)

    def create_run(
        self,
        script: str,
        instance_name: str = None,
        instance_type: str = "cpu",
        instance_count: int = 1,
        requirements_file: Optional[str] = None,
        # Hugging Face Specific.
        dataset_id: Optional[str] = None,
        private: bool = False,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        save_code_snapshot_in_dataset_repo: bool = False,
        delete_space_on_completion: bool = True,
        unpack_script_args_to_main: bool = False,
        space_save_dir: str = "outputs",
        # Script Args
        **kwargs,
    ):
        space_id = instance_name
        space_hardware = HUGGINGFACE_COMPUTE_TARGETS_MAP[instance_type]
        if space_hardware is None:
            raise ValueError(
                f"Invalid instance type: {instance_type}. Should be one of {list(HUGGINGFACE_COMPUTE_TARGETS_MAP.keys())}"
            )

        """Submits a run on a compute target. Returns run info"""
        if instance_count > 1:
            raise NotImplementedError("Multi-node runs not supported yet")

        script_args = kwargs or {}
        script_args_lst = list(chain(*((f"--{n}", f"{v}") for n, v in script_args.items())))
        print("Script args: ", script_args)
        print("Script args str list:", script_args_lst)

        username = whoami()["name"]
        task_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        space_id = space_id or f"{username}/task-runner-{task_id}"
        dataset_id = dataset_id or f"{username}/task-outputs-{task_id}"

        # Create 2 new repos. One space for running code, one dataset for storing artifacts
        space_repo_url = create_repo(
            space_id,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=space_hardware,
            private=private,
        )
        dataset_repo_url = create_repo(dataset_id, exist_ok=True, repo_type="dataset", private=private)
        print(f"Created Repo at: {space_repo_url}")
        print(f"Created Dataset at: {dataset_repo_url}")

        # Add current HF token to the new space, so it has ability to push to output dataset
        add_space_secret(space_id, "HF_TOKEN", HfFolder().get_token())

        # We want to ignore at the very least README.md and .git folder of the cloned
        # GitHub repo, but you can include more filters if you want.
        if ignore_patterns is None:
            ignore_patterns = []
        elif isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]
        ignore_patterns += [".git*", "README.md"]

        source_dir = Path(script).parent
        # We push the source up to the Space
        upload_folder(
            repo_id=space_id,
            folder_path=str(source_dir),
            path_in_repo=".",
            repo_type="space",
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        # Optionally, you can also push the source to the output dataset
        if save_code_snapshot_in_dataset_repo:
            upload_folder(
                repo_id=dataset_id,
                folder_path=str(source_dir),
                path_in_repo="./code_snapshot",
                repo_type="dataset",
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

        # We put together some metadata here about the task and push that to the dataset
        # for safekeeping.
        task_metadata = dict(
            script=script,
            requirements_file=requirements_file,
            space_id=space_id,
            space_hardware=space_hardware,
        )
        upload_file(
            repo_id=dataset_id,
            path_or_fileobj=json.dumps(task_metadata, indent=2, sort_keys=False).encode(),
            path_in_repo="task_metadata.json",
            repo_type="dataset",
        )
        print("Uploaded task metadata to dataset repo for tracking!")

        # Next, we need to tell the Space to use the wrapper app instead of the default app.py
        card = RepoCard.load(space_id, repo_type="space")
        if card.data.app_file != "task_run_wrapper.py":
            card.data.app_file = "task_run_wrapper.py"
            card.push_to_hub(space_id, repo_type="space")

        # After that, we upload an "about.md" file to display something while the runner is up
        about_md_content = _about_md_content.format(output_repo_url=dataset_repo_url)
        upload_file(
            repo_id=space_id,
            path_or_fileobj=about_md_content.encode(),
            path_in_repo="about.md",
            repo_type="space",
        )

        # Finally, update the wrapper file with the given information and push it to the Hub.
        task_wrapper_content = _task_run_script_template.format(
            script=Path(script).stem,
            script_args=script_args,
            output_dataset_id=dataset_id,
            output_dir=space_save_dir,
            space_id=space_id,
            delete_space_on_completion=delete_space_on_completion,
            script_args_lst=script_args_lst,
            unpack_script_args_to_main=unpack_script_args_to_main,
        )
        upload_file(
            repo_id=space_id,
            path_or_fileobj=task_wrapper_content.encode(),
            path_in_repo="task_run_wrapper.py",
            repo_type="space",
        )
        return space_repo_url, dataset_repo_url

    def list_runs(self, **kwargs):
        raise NotImplementedError
