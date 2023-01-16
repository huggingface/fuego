import fire

from .provider_azml import AzureMLProvider

# TODO - something like a registry w/ all available providers
providers = {
    "azureml": AzureMLProvider,
}

def run(
    script,
    provider="azureml",
    instance_name="feugo-compute",
    instance_type="cpu",
    requirements_file=None,
    **kwargs
):

    if provider not in providers:
        raise ValueError(f"Provider {provider} not supported. Supported providers: {providers}")
    
    # Initialization should check for auth, etc.
    provider = providers[provider]()

    provider.create_run(
        script=script,
        instance_name=instance_name,
        instance_type=instance_type,
        requirements_file=requirements_file,
        **kwargs
    )

def main():
    fire.Fire()

if __name__ == "__main__":
    main()
