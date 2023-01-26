import json
from pathlib import Path
from typing import Optional

import fire

from .provider_azml import AzureMLProvider


# TODO - something like a registry w/ all available providers
providers = {
    "azureml": AzureMLProvider,
}


def run(
    script: Path,
    provider: str = "azureml",
    instance_name: str = "fuego-compute",
    instance_type: str = "cpu",
    instance_count: int = 1,
    requirements_file: Optional[str] = None,
    **kwargs,
):

    if provider not in providers:
        raise ValueError(f"Provider {provider} not supported. Supported providers: {providers}")

    # Initialization should check for auth, etc.
    provider = providers[provider]()

    print(f"script: {script}")
    print(f"provider: {provider}")
    print(f"instance_name: {instance_name}")
    print(f"instance_type: {instance_type}")
    print(f"instance_count: {instance_count}")
    print(f"requirements_file: {requirements_file}")
    print(f"kwargs:\n{json.dumps(kwargs, indent=4)}")

    provider.create_run(
        script=script,
        instance_name=instance_name,
        instance_type=instance_type,
        instance_count=instance_count,
        requirements_file=requirements_file,
        **kwargs,
    )


def main():
    fire.Fire()


if __name__ == "__main__":
    main()
