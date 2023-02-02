import json
import tempfile
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import fire
import git
from huggingface_hub import HfFolder, add_space_secret, create_repo, upload_file, upload_folder
from huggingface_hub.repocard import RepoCard
from huggingface_hub.utils import logging


logger = logging.get_logger(__name__)


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


def run(
    script: str,
    requirements_file: Optional[str] = None,
    space_id: str = None,
    space_hardware: str = "cpu",
    dataset_id: Optional[str] = None,
    private: bool = False,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    save_code_snapshot_in_dataset_repo: bool = False,
    delete_space_on_completion: bool = True,
    unpack_script_args_to_main: bool = False,
    space_output_dir: str = "outputs",
    token: Optional[str] = None,
    extra_run_metadata: Optional[dict] = None,
    **kwargs,
):
    """Create a Hugging Face Space and run a script in it. When finished, the outputs will be saved to a Hugging Face Dataset Repo.

    Args:
        script (`str`):
            Path to the script to run.
        requirements_file (`str`, optional):
            Path to requirements file for the job. Defaults to None.
        space_id (`str`, optional):
            ID of the Hugging Face Space. Defaults to None.
        space_hardware (`str`, optional):
            Hardware for the Hugging Face Space. Defaults to "cpu".
        dataset_id (`str`, optional):
            ID of the Hugging Face Dataset Repo. Defaults to None.
        private (bool, optional):
            If True, both the Hugging Face Space and Dataset Repo will be private. Defaults to False.
        allow_patterns (`List[str]`, optional):
            List of file patterns to include in the parent directory of `script`. Defaults to None.
        ignore_patterns (`List[str]`, optional):
            List of file patterns to exclude in the parent directory of `script`. Defaults to None.
        save_code_snapshot_in_dataset_repo (`bool`, optional):
            If True, a code snapshot will be saved in the Hugging Face Dataset Repo. Defaults to False.
        delete_space_on_completion (`bool`, optional):
            If True, the Hugging Face Space will be deleted after the job completes. Defaults to True.
        unpack_script_args_to_main (`bool`, optional):
            If True, kwargs will be unpacked to the main function. Otherwise, they will be used to override
            `sys.argv`, which assumes your `main` function handles argument parsing. Defaults to False.
        space_output_dir (`str`, optional):
            Dir in the space that will be uploaded to output dataset on run completion. Defaults to "outputs".
        token (`str`, optional):
            Hugging Face token. Uses your cached token (if available) by default. Defaults to None.
        extra_run_metadata (`dict`, optional):
            Extra metadata to add to the run metadata json file that gets added to the output dataset. Defaults to None.
        **kwargs:
            Keyword arguments are passed to the script as argparse args or unpacked to the main function.

    Raises:
        ValueError: When `space_hardware` is not a valid Hugging Face Space hardware type.

    Returns:
        Tuple[str, str]: Tuple of the Hugging Face Space URL and Hugging Face Dataset Repo URL.
    """
    space_hardware = HUGGINGFACE_COMPUTE_TARGETS_MAP[space_hardware]
    if space_hardware is None:
        raise ValueError(
            f"Invalid instance type: {space_hardware}. Should be one of {list(HUGGINGFACE_COMPUTE_TARGETS_MAP.keys())}"
        )

    script_args = kwargs or {}
    script_args_lst = list(chain(*((f"--{n}", f"{v}") for n, v in script_args.items())))
    logger.info("Script args: ", script_args)
    logger.info("Script args str list:", script_args_lst)

    task_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    space_id = space_id or f"task-runner-{task_id}"
    dataset_id = dataset_id or f"task-outputs-{task_id}"

    # Create 2 new repos. One space for running code, one dataset for storing artifacts
    space_repo_url = create_repo(
        space_id,
        exist_ok=True,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=space_hardware,
        private=private,
        token=token,
    )
    space_id = space_repo_url.repo_id

    dataset_repo_url = create_repo(dataset_id, exist_ok=True, repo_type="dataset", private=private, token=token)
    dataset_id = dataset_repo_url.repo_id

    logger.info(f"Created Repo at: {space_repo_url}")
    logger.info(f"Created Dataset at: {dataset_repo_url}")

    # Add current HF token to the new space, so it has ability to push to output dataset
    add_space_secret(space_id, "HF_TOKEN", HfFolder().get_token(), token=token)

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
        token=token,
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
            token=token,
        )

    # We put together some metadata here about the task and push that to the dataset
    # for safekeeping.
    run_metadata = dict(
        script=script,
        requirements_file=requirements_file,
        space_id=space_id,
        space_hardware=space_hardware,
        **extra_run_metadata or {},
    )

    upload_file(
        repo_id=dataset_id,
        path_or_fileobj=json.dumps(run_metadata, indent=2, sort_keys=False).encode(),
        path_in_repo="run_metadata.json",
        repo_type="dataset",
        token=token,
    )
    logger.info("Uploaded run metadata to dataset repo for tracking!")

    # Next, we need to tell the Space to use the wrapper app instead of the default app.py
    # Additionally, we add the "fuego" tag to the Space so it can be searched for later
    card = RepoCard.load(space_id, repo_type="space")
    changes_made_to_card = False
    if card.data.app_file != "task_run_wrapper.py":
        card.data.app_file = "task_run_wrapper.py"
        changes_made_to_card = True
    elif "fuego" not in card.data.tags:
        card.data.tags.append("fuego")
        changes_made_to_card = True
    if changes_made_to_card:
        card.push_to_hub(space_id, repo_type="space", token=token)

    # After that, we upload an "about.md" file to display something while the runner is up
    about_md_content = _about_md_content.format(output_repo_url=dataset_repo_url)
    upload_file(
        repo_id=space_id,
        path_or_fileobj=about_md_content.encode(),
        path_in_repo="about.md",
        repo_type="space",
        token=token,
    )

    # Finally, update the wrapper file with the given information and push it to the Hub.
    task_wrapper_content = _task_run_script_template.format(
        script=Path(script).stem,
        script_args=script_args,
        output_dataset_id=dataset_id,
        output_dir=space_output_dir,
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
        token=token,
    )
    return space_repo_url, dataset_repo_url


def github_run(
    github_repo_id: str,
    script: str,
    requirements_file: Optional[str] = None,
    github_repo_branch: str = "main",
    space_id: str = None,
    space_hardware: str = "cpu",
    dataset_id: Optional[str] = None,
    private: bool = False,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    save_code_snapshot_in_dataset_repo: bool = False,
    delete_space_on_completion: bool = True,
    unpack_script_args_to_main: bool = False,
    space_output_dir: str = "outputs",
    token: Optional[str] = None,
    extra_run_metadata: Optional[dict] = None,
    **kwargs,
):
    """Create a run from code within a GitHub repo. See `run` for more details."""
    # We clone the GitHub repo into a temporary directory
    with tempfile.TemporaryDirectory() as tmp:
        repo_url = f"https://github.com/{github_repo_id}"
        repo = git.Repo.clone_from(repo_url, tmp, branch=github_repo_branch)
        tempdir = Path(tmp)

        script_path = tempdir / script
        if not script_path.exists():
            raise ValueError(f"Could not find script {script} in repo {repo_url}")
        script = str(script_path)

        if requirements_path is not None:
            requirements_path = tempdir / requirements_file
            if not requirements_path.exists():
                raise ValueError(f"Could not find requirements file {requirements_file} in repo {repo_url}")
            requirements_file = str(requirements_path)

        return run(
            script=str(script_path),
            requirements_file=requirements_file,
            space_id=space_id,
            space_hardware=space_hardware,
            dataset_id=dataset_id,
            private=private,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            save_code_snapshot_in_dataset_repo=save_code_snapshot_in_dataset_repo,
            delete_space_on_completion=delete_space_on_completion,
            unpack_script_args_to_main=unpack_script_args_to_main,
            space_output_dir=space_output_dir,
            token=token,
            extra_run_metadata=dict(
                github_repo_id=github_repo_id,
                github_repo_branch=github_repo_branch,
                github_repo_sha=repo.head.object.hexsha,
                **extra_run_metadata or {},
            )
            ** kwargs,
        )


def cli_run():
    fire.Fire(
        {
            "run": run,
            "github_run": github_run,
        }
    )
