import os
from pathlib import Path
from typing import Optional

from .provider_utils import Provider
from .runtime import is_sagemaker_available


if is_sagemaker_available():
    import sagemaker
    from sagemaker.huggingface import HuggingFace


# TODO - add a ton more of these, more options for cpu, gpu, etc.
SAGEMAKER_COMPUTE_TARGETS_MAP = {
    "cpu": "ml.m5.large",
    "K80": "ml.p2.xlarge",
}


class SagemakerProvider(Provider):
    # TODO - rethink init logic for providers so signature here is always the same.
    # Maybe we automate configuration process across clouds and save/load from some cache file?
    def __init__(
        self,
        arn=None,
    ):
        if not is_sagemaker_available():
            raise ImportError("Sagemaker is not available. Please install it with `pip install fuego[sagemaker]`")

        arn = arn or os.environ.get("SAGEMAKER_ROLE_ARN")
        try:
            role = arn or sagemaker.get_execution_role()
        except ValueError:
            # TODO - maybe auth with boto3 here via role name?
            if arn is None:
                raise RuntimeError(
                    "Could not determine sagemaker execution role automatically.\n\nThis probably means you're running this script outside of a Sagemaker Studio Notebook.\n"
                    "To resolve this issue, please provide a valid ARN to the --role flag or set it as the SAGEMAKER_ROLE_ARN environment variable."
                )

        self.role = role

    def create_run(
        self,
        script: str,
        instance_name: str,  # NOTE: instance_name not used in sagemaker...should probably remove from base.
        instance_type: str,
        instance_count: int = 1,
        requirements_file: Optional[str] = None,
        # Sagemaker Specific.
        image_uri: str = "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
        py_version: str = "py38",
        # Script Args
        **kwargs,
    ):
        """Submits a run on a compute target. Returns run info"""

        sagemaker_instance_type = SAGEMAKER_COMPUTE_TARGETS_MAP.get(instance_type)
        if not sagemaker_instance_type:
            raise ValueError(
                f"Invalid instance type {instance_type}, must be one of {list(SAGEMAKER_COMPUTE_TARGETS_MAP)}"
            )

        huggingface_estimator = HuggingFace(
            entry_point=Path(script).name,
            source_dir=str(Path(script).parent),
            instance_type=sagemaker_instance_type,
            instance_count=instance_count,
            role=self.role,
            py_version=py_version,
            image_uri=image_uri,
            hyperparameters=kwargs,
            dependencies=[str(requirements_file)] if requirements_file else None,
        )

        huggingface_estimator.fit(wait=True)

    def list_runs(self, experiment_name=None, **kwargs):  # TODO- proper args
        raise NotImplementedError

    def create_compute_target(
        self,
        name,
        instance_type,
    ):
        """Not applicable for Sagemaker...should update base provider"""
        raise NotImplementedError

    def list_compute_targets(self, **kwargs):
        raise NotImplementedError

    def delete_compute_target(self, name, **kwargs):
        raise NotImplementedError
