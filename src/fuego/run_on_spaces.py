import json
import tempfile
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import fire
import git
from huggingface_hub import HfFolder, SpaceHardware, add_space_secret, create_repo, upload_file, upload_folder
from huggingface_hub.utils import logging


logger = logging.get_logger(__name__)


SPACES_HARDWARE_TYPES = [x.value for x in SpaceHardware]


_status_checker_content = """import os
import subprocess
import time
from pathlib import Path
from threading import Thread
from typing import List, Union

import gradio as gr
from huggingface_hub import HfFolder, delete_repo, upload_folder, get_space_runtime, request_space_hardware, upload_file, hf_hub_download


def process_is_complete(process_pid):
    '''Checks if the process with the given PID is still running'''
    p = subprocess.Popen(["ps", "-p", process_pid], stdout=subprocess.PIPE)
    out = p.communicate()[0].decode("utf-8").strip().split("\\n")
    return len(out) == 1

def get_task_status(output_dataset_id):
    '''Downloads a file .meta/task_status.txt from the output dataset repo and returns its contents'''
    file_path = hf_hub_download(output_dataset_id, ".meta/task_status.txt", repo_type="dataset")
    task_status = Path(file_path).read_text().strip()
    return task_status

def set_task_status(output_dataset_id, status="done"):
    '''Uploads a file .meta/task_status.txt to the output dataset repo with the given status'''
    upload_file(
        repo_id=output_dataset_id,
        path_or_fileobj=status.encode(),
        path_in_repo=".meta/task_status.txt",
        repo_type="dataset",
    )

def check_for_status(
    process_pid, this_space_id, output_dataset_id, output_dirs, delete_on_completion, downgrade_hardware_on_completion
):
    task_status = get_task_status(output_dataset_id)
    print("Task status (found in dataset repo)", task_status)
    if task_status == "done":
        print("Task was already done, exiting...")
        return
    elif task_status == "preparing":
        print("Setting task status to running...")
        set_task_status(output_dataset_id, "running")

    print("Watching PID of script to see if it is done running")
    while True:
        if process_is_complete(process_pid):
            print("Process is complete! Uploading assets to output dataset repo")
            for output_dir in output_dirs:
                if Path(output_dir).exists():
                    print("Uploading folder", output_dir)
                    upload_folder(
                        repo_id=output_dataset_id,
                        folder_path=str(output_dir),
                        path_in_repo=str(Path('.outputs') / output_dir),
                        repo_type="dataset",
                    )
                else:
                    print("Folder", output_dir, "does not exist, skipping")

            print("Finished uploading outputs to dataset repo...Finishing up...")
            if delete_on_completion:
                print("Deleting space...")
                delete_repo(repo_id=this_space_id, repo_type="space")
            elif downgrade_hardware_on_completion:
                runtime = get_space_runtime(this_space_id)
                if runtime.hardware != "cpu-basic":
                    print("Requesting downgrade to CPU Basic...")
                    request_space_hardware(repo_id=this_space_id, hardware="cpu-basic")
                else:
                    print("Space is already on cpu-basic, not downgrading.")
                print("Downgrading hardware...")
            print("Done! Setting task status to done in dataset repo")
            set_task_status(output_dataset_id, "done")
            return
        print("Didn't find it...sleeping for 5 seconds.")
        time.sleep(5)


def main(
    this_space_repo_id: str,
    output_dataset_id: str,
    output_dirs: Union[str, List[str]] = "./outputs",
    delete_on_completion: bool = True,
    downgrade_hardware_on_completion: bool = True,
):
    token_env_var = os.getenv("HF_TOKEN")
    if token_env_var is None:
        raise ValueError(
            "Please set HF_TOKEN environment variable to your Hugging Face token. You can do this in the settings tab of your space."
        )

    if isinstance(output_dirs, str):
        output_dirs = [output_dirs]

    HfFolder().save_token(token_env_var)

    # Watch python script's process to see when it's done running
    process_pid = os.getenv("USER_SCRIPT_PID", None)

    with gr.Blocks() as demo:
        gr.Markdown(Path("about.md").read_text())

    thread = Thread(
        target=check_for_status,
        daemon=True,
        args=(
            process_pid,
            this_space_repo_id,
            output_dataset_id,
            output_dirs,
            delete_on_completion,
            downgrade_hardware_on_completion,
        ),
    )
    thread.start()
    demo.launch()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
"""

# TODO - align with the GPU Dockerfile a bit more
_dockerfile_cpu_content = """FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install --no-cache-dir fire gradio datasets huggingface_hub

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

RUN chmod +x start_server.sh

CMD ["./start_server.sh"]
"""

_dockerfile_gpu_content = """FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda clean -ya


ENV PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces

RUN pip install --no-cache-dir fire gradio datasets huggingface_hub

# Install user requirements
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

RUN chmod +x start_server.sh

CMD ["./start_server.sh"]
"""

_start_server_template = """#!/bin/bash

# Start the python script in the background asynchronously
nohup {command} &

# Save the PID of the python script so we can reference it in the status checker
export USER_SCRIPT_PID=$!

# Start a simple web server to watch the status of the python script
python status_checker.py {status_checker_args}
"""

_about_md_template = """
# Task Runner

This space is running some job!

- Check out the associated [output repo]({output_repo_url})
"""


def _convert_dict_to_args_str(args_dict: dict) -> str:
    """Convert a dictionary of arguments to a string of arguments that can be passed to a command line script"""
    args_str = ""
    for arg_name, arg_value in args_dict.items():
        if isinstance(arg_value, (str, list, dict, tuple)):
            args_str += f' --{arg_name} "{repr(arg_value)}"'
        else:
            args_str += f" --{arg_name} {arg_value}"
    return args_str.strip()


def run(
    script: str,
    requirements_file: Optional[str] = None,
    space_id: str = None,
    space_hardware: str = "cpu-basic",
    dataset_id: Optional[str] = None,
    private: bool = False,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    save_code_snapshot_in_dataset_repo: bool = False,
    delete_space_on_completion: bool = True,
    downgrade_hardware_on_completion: bool = True,
    space_output_dirs: Optional[List[str]] = None,
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
        downgrade_hardware_on_completion (`bool`, optional):
            If True, and `delete_space_on_completion` is False, the Hugging Face Space hardware will be
            downgraded to "cpu-basic" after the job completes. Defaults to True.
        space_output_dirs (`str`, optional):
            Dirs in the space that will be uploaded to output dataset on run completion. If unspecified,
            will default to ["outputs", "logs"].
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
    if space_hardware not in SPACES_HARDWARE_TYPES:
        raise ValueError(f"Invalid instance type: {space_hardware}. Should be one of {SPACES_HARDWARE_TYPES}")

    if space_output_dirs is None:
        space_output_dirs = ["outputs", "logs"]

    # The command to run in the space
    # Ex. python train.py --learning_rate 0.1
    command = f"python {script} {_convert_dict_to_args_str(kwargs)}"

    task_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    space_id = space_id or f"task-runner-{task_id}"
    dataset_id = dataset_id or f"task-outputs-{task_id}"

    # Create 2 new repos. One space for running code, one dataset for storing artifacts
    space_repo_url = create_repo(
        space_id,
        exist_ok=True,
        repo_type="space",
        space_sdk="docker",
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

    if requirements_file is None:
        requirements_file = source_dir / "requirements.txt"
        requirements_file.touch()
        requirements_file = str(requirements_file)

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
            path_in_repo=".snapshot",
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
        path_in_repo=".meta/run_metadata.json",
        repo_type="dataset",
        token=token,
    )
    logger.info("Uploaded run metadata to dataset repo for tracking!")
    upload_file(
        repo_id=dataset_id,
        path_or_fileobj="preparing".encode(),
        path_in_repo=".meta/task_status.txt",
        repo_type="dataset",
    )

    # about.md
    upload_file(
        repo_id=space_id,
        path_or_fileobj=_about_md_template.format(output_repo_url=dataset_repo_url).encode(),
        path_in_repo="about.md",
        repo_type="space",
        token=token,
    )

    # start_server.sh
    upload_file(
        repo_id=space_id,
        path_or_fileobj=_start_server_template.format(
            command=command,
            status_checker_args=_convert_dict_to_args_str(
                dict(
                    this_space_repo_id=space_id,
                    output_dataset_id=dataset_id,
                    output_dirs=space_output_dirs,
                    delete_on_completion=delete_space_on_completion,
                    downgrade_hardware_on_completion=downgrade_hardware_on_completion,
                )
            ),
        ).encode(),
        path_in_repo="start_server.sh",
        repo_type="space",
        token=token,
    )

    # status_checker.py
    upload_file(
        repo_id=space_id,
        path_or_fileobj=_status_checker_content.encode(),
        path_in_repo="status_checker.py",
        repo_type="space",
        token=token,
    )

    # Dockerfile
    dockerfile_content = (
        _dockerfile_cpu_content if space_hardware in ["cpu-basic", "cpu-upgrade"] else _dockerfile_gpu_content
    )
    upload_file(
        repo_id=space_id,
        path_or_fileobj=dockerfile_content.encode(),
        path_in_repo="Dockerfile",
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
    space_hardware: str = "cpu-basic",
    dataset_id: Optional[str] = None,
    private: bool = False,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    save_code_snapshot_in_dataset_repo: bool = False,
    delete_space_on_completion: bool = True,
    downgrade_hardware_on_completion: bool = True,
    space_output_dirs: Optional[List[str]] = None,
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

        if requirements_file is not None:
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
            downgrade_hardware_on_completion=downgrade_hardware_on_completion,
            space_output_dirs=space_output_dirs,
            token=token,
            extra_run_metadata=dict(
                github_repo_id=github_repo_id,
                github_repo_branch=github_repo_branch,
                github_repo_sha=repo.head.object.hexsha,
                **extra_run_metadata or {},
            ),
            **kwargs,
        )


def cli_run():
    fire.Fire(
        {
            "run": run,
            "github_run": github_run,
        }
    )
