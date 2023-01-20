# Examples

## Simple Example With Requirements

This example shows how to install dependencies via a requirements.txt file. The example itself just takes in one string script flag, `--message`, which will be saved to `./logs` wherever the script was run from.

From this `./examples` directory in a local clone, you can try this example out with either Python or the CLI...

#### Run from Python

```python
from fuego import run

fuego.run(
    script='./simple_example_with_requirements/run.py',
    provider='azureml',
    instance_type='cpu',
    instance_count=1,
    requirements_file='./simple_example_with_requirements/requirements.txt',
    # Script kwargs - these are passed to the script as argparse args
    message='Howdy, world!',
)
```

#### Run from CLI

```
fuego run \
    --provider azureml \
    --instance-type cpu \
    --instance-count 1 \
    --requirements-file ./simple_example_with_requirements/requirements.txt \
    ./simple_example_with_requirements/run.py \
    --message "'Howdy, world!'"
```