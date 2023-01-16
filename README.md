# fuego

A 🔥 tool for running code in the cloud

## The idea

A unified interface to view/manage runs, compute instances, data, etc. across your authenticated cloud providers.

## WIP API

The Python API and CLI should have very similar experiences so folks can use whichever they prefer.

#### Python

```python
import fuego

fuego.run(
    # Fuego Run Args - these are the same across providers
    script='./examples/simple_example/run.py',
    requirements_file=None,
    provider='azureml',
    instance_name='feugo-cpu-compute',
    instance_type='cpu',  # K80, V100, 2xK80, 2xV100, etc...

    # Provider Specific Args (AzureML in this case)
    # hopefully we minimize these so exp is same across providers.
	# Probably possible to get more opinionated about how we handle this.
    environment_name='AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:47',

    # Script kwargs - these are passed to the script
    message="Howdy, world!"
)
```

#### CLI

```
fuego run \
  --provider azureml \
  --environment_name AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:47 \
  --instance_name fuego-cpu-compute \
  --instance_type cpu \
  ./examples/simple_example/run.py \
  --message "Howdy, world"
```
