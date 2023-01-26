import itertools
import os
from pathlib import Path
from typing import Optional

from tabulate import tabulate

from .provider_utils import Provider
from .runtime import is_azureml_available


if is_azureml_available():
    from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
    from azureml.core.authentication import ServicePrincipalAuthentication
    from azureml.core.compute import AmlCompute, ComputeTarget
    from azureml.core.runconfig import DockerConfiguration, MpiConfiguration


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
    def __init__(
        self,
        workspace_config="config.json",
        subscription_id=None,
        resource_group=None,
        workspace_name=None,
        tenant_id=None,
        client_id=None,
        client_secret=None,
    ):
        if not is_azureml_available():
            raise ImportError("AzureML is not available. Please install it with `pip install fuego[azureml]`")

        subscription_id = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
        resource_group = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
        workspace_name = workspace_name or os.environ.get("AZURE_WORKSPACE_NAME")
        tenant_id = tenant_id or os.environ.get("AZURE_TENANT_ID")
        client_id = client_id or os.environ.get("AZURE_CLIENT_ID")
        client_secret = client_secret or os.environ.get("AZURE_CLIENT_SECRET")

        do_use_service_principal = all([tenant_id, client_id, client_secret])
        if do_use_service_principal:
            auth = ServicePrincipalAuthentication(
                tenant_id=tenant_id,
                service_principal_id=client_id,
                service_principal_password=client_secret,
            )
        else:
            auth = None

        do_use_config_file = not all([subscription_id, resource_group, workspace_name])
        if do_use_config_file:
            self.ws = Workspace.from_config(workspace_config, auth=auth)
        else:
            self.ws = Workspace(
                subscription_id=subscription_id,
                resource_group=resource_group,
                workspace_name=workspace_name,
                auth=auth,
            )

    def create_run(
        self,
        script: str,
        instance_name: str,
        instance_type: str,
        instance_count: int = 1,
        requirements_file: Optional[str] = None,
        # AzureML Specific.
        idle_time_before_scale_down: int = 120,
        vm_priority: str = "lowpriority",
        experiment_name: str = "fuego-experiment",
        docker_image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
        environment_name: str = "fuego-env",
        display_name: str = "fuego-run",
        # Script Args
        **kwargs,
    ):
        """Submits a run on a compute target. Returns run info"""
        print("Inside AZML Provider")

        target = self.create_compute_target(
            name=instance_name,
            instance_type=instance_type,
            idle_time_before_scale_down=idle_time_before_scale_down,
            vm_priority=vm_priority,
        )
        env = self.create_environment(
            name=environment_name, docker_image=docker_image, requirements_file=requirements_file
        )
        exp = Experiment(self.ws, experiment_name)

        args = list(itertools.chain(*((f"--{n}", f"{v}") for n, v in kwargs.items())))

        script_run_config = ScriptRunConfig(
            source_directory=Path(script).parent,
            script=Path(script).name,
            compute_target=target,
            environment=env,
            distributed_job_config=None
            if instance_count == 1
            else MpiConfiguration(process_count_per_node=1, node_count=instance_count),
            docker_runtime_config=DockerConfiguration(use_docker=docker_image is not None),
            arguments=args,
        )
        run = exp.submit(script_run_config)
        print("Submitting job to AzureML...\n")
        print(run)
        print()

    def list_runs(self, experiment_name=None, **kwargs):  # TODO- proper args

        # Grabs most recent exp if not provided. Definitely log this info
        experiment_name = experiment_name or list(self.ws.experiments)[-1]
        runs = list(self.ws.experiments[experiment_name].get_runs())

        return tabulate(
            [[r.experiment.name, r.id, r.status] for r in runs], headers=["Experiment", "Run ID", "Status"]
        )

    def create_compute_target(
        self,
        name,
        instance_type,
        # AzureML Specific
        min_nodes=0,
        max_nodes=1,
        idle_time_before_scale_down=120,
        vm_priority="lowpriority",
    ):
        azml_instance_type = AZUREML_COMPUTE_TARGETS_MAP.get(instance_type)
        if not azml_instance_type:
            raise ValueError(
                f"Invalid instance type {instance_type}, must be one of {list(AZUREML_COMPUTE_TARGETS_MAP)}"
            )

        print(f"Creating compute target {name} with instance type {azml_instance_type}...")
        if name in self.ws.compute_targets:
            return ComputeTarget(workspace=self.ws, name=name)
        else:
            config = AmlCompute.provisioning_configuration(
                vm_size=azml_instance_type,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                vm_priority=vm_priority,
                idle_seconds_before_scaledown=idle_time_before_scale_down,
            )
            target = ComputeTarget.create(self.ws, name, config)
            # target.wait_for_completion(show_output=True)
        return target

    def list_compute_targets(self, **kwargs):
        raise NotImplementedError

    def delete_compute_target(self, name, **kwargs):
        raise NotImplementedError

    def create_environment(self, name: str, docker_image=None, requirements_file=None):
        env = self.ws.environments.get(name)

        if env:
            return env

        if docker_image is None and requirements_file is None:
            raise ValueError("Must provide either docker_image or requirements_file when creating env")

        if requirements_file:
            requirements_file = Path(requirements_file)
            if not requirements_file.exists():
                raise RuntimeError(
                    f"Given requirements file '{requirements_file}' does not exist at the provided path"
                )
            elif requirements_file.suffix == ".txt":
                env = Environment.from_pip_requirements(name, requirements_file)
            elif requirements_file.suffix in [".yml", ".yaml"]:
                env = Environment.from_conda_specification(name, requirements_file)
            else:
                print("Couldn't resolve env from requirements file")
        else:
            env = Environment(name)

        if docker_image:
            env.docker.base_image = docker_image

        return env
