from typing import Optional, List
from pathlib import Path

from .provider_utils import Provider
from .runtime import is_azureml_available

if is_azureml_available():
    from azure.ai.ml import MLClient, command, Input, Output
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import AmlCompute, Environment


# TODO - add a ton more of these, more options for cpu, gpu, etc.
AZUREML_COMPUTE_TARGETS_MAP = {
    "cpu": "Standard_DS3_v2",
    "K80": "Standard_NC6",
    "2xK80": "Standard_NC12",
    "V100": "Standard_NC6s_v3",
    "2xV100": "Standard_NC12s_v3",
}

class AzureMLProvider(Provider):
    # TODO - rethink init logic for providers so signature here is always the same.
    # Maybe we automate configuration process across clouds and save/load from some cache file?
    def __init__(self, config_file_path: str = 'config.json'):
        if not is_azureml_available():
            raise ImportError("AzureML is not available. Please install it with `pip install fuego[azureml]`")
        self.ml_client = MLClient.from_config(DefaultAzureCredential(), path=config_file_path)

    def create_run(
        self,
        script: str,
        instance_name: str,
        instance_type: str,
        requirements_file: Optional[str]=None,
        # AzureML Specific.
        idle_time_before_scale_down: int=120,
        tier: str="lowpriority",
        experiment_name: str='fuego-experiment',
        docker_image: str="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
        environment_name: str="fuego-env",
        display_name: str="fuego-run",
        # Script Args
        **kwargs
    ):
        """Submits a run on a compute target. Returns run info"""
        print("Inside AZML Provider")

        self.create_compute_target(
            name=instance_name,
            instance_type=instance_type,
            idle_time_before_scale_down=idle_time_before_scale_down,
            tier=tier
        )
        self.create_environment(
            name=environment_name,
            docker_image=docker_image,
            requirements_file=requirements_file
        )

        kwargs_as_args = " ".join([f"--{k} {v}" for k, v in kwargs.items()])
        cmd = f"python {script} {kwargs_as_args}"
        print(f"Invocation Command:\n\n{command}\n")

        job = command(
            code=str(Path(script).parent),
            command=cmd,
            environment=environment_name,
            compute=instance_name,
            experiment_name=experiment_name,
            display_name=display_name
        )
        print("Submitting job to AzureML...\n")
        print(job)
        self.ml_client.jobs.create_or_update(job)

    def list_runs(self, **kwargs):
        raise NotImplementedError

    def create_compute_target(
        self,
        name,
        instance_type,
        # AzureML Specific
        min_nodes=0,
        max_nodes=1,
        idle_time_before_scale_down=120,
        tier="lowpriority"
    ):
        azml_instance_type = AZUREML_COMPUTE_TARGETS_MAP.get(instance_type)
        if not azml_instance_type:
            raise ValueError(f"Invalid instance type {instance_type}, must be one of {list(AZUREML_COMPUTE_TARGETS_MAP)}")

        print(f"Creating compute target {name} of type {instance_type} (azml: {azml_instance_type})")
        target = AmlCompute(
            name=name,
            size=azml_instance_type,
            min_instances=min_nodes,
            max_instances=max_nodes,
            idle_time_before_scale_down=idle_time_before_scale_down,
            tier=tier
        )
        self.ml_client.begin_create_or_update(target)

    def list_compute_targets(self, **kwargs):
        raise NotImplementedError
    
    def delete_compute_target(self, name, **kwargs):
        raise NotImplementedError

    def create_environment(self, name, docker_image, requirements_file=None):
        name, version = name.split(":")
        env = self.ml_client.environments.get(name, version=version)
        if env:
            return

        environment = Environment(
            image=docker_image,
            conda_file=requirements_file,
            name=name,
        )
        self.ml_client.environments.create_or_update(environment)
