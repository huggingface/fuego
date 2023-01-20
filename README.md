# fuego

A 🔥 tool for running code in the cloud

## The idea

A unified interface to view/manage runs, compute instances, data, etc. across your authenticated cloud providers.

## Installation

For now, you can install from source:

```bash
git clone https://github.com/huggingface/fuego.git
cd fuego
pip install -e ".[azureml]"
```

## WIP API

The Python API and CLI should have very similar experiences so folks can use whichever they prefer.

#### Python


```python
import fuego

fuego.run(
    # Fuego Run Args - these are the same across providers
    script='./examples/simple_example_with_requirements/run.py',
    provider='azureml',
    instance_type='cpu',
    instance_count=1,
    requirements_file='./examples/simple_example_with_requirements/requirements.txt',
    # Script kwargs - these are passed to the script as argparse args
    message='Howdy, world!',
)
```

#### CLI

```
fuego run \
    --provider azureml \
    --instance-type cpu \
    --instance-count 1 \
    --requirements-file ./examples/simple_example_with_requirements/requirements.txt \
    ./examples/simple_example_with_requirements/run.py \
    --message "'Howdy, world!'"
```
