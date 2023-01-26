# Examples

## Simple Example With Requirements

This example shows how to install dependencies via a requirements.txt file. The example itself just takes in one string script flag, `--message`, which will be saved to `./logs` wherever the script was run from.

From this `./examples` directory in a local clone, you can try this example out with either Python or the CLI...

#### Run from Python

```python
import fuego

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

## Transformers Text Classification

This example is a copy of the `run_glue.py` script from the `transformers` library that we can use to check GPU-based cloud runs.

Here, we'll run on a K80 instance on AzureML.

#### Run from Python

```python
import fuego

fuego.run(
    script='./transformers_text_classification/run_glue.py',
    provider='azureml',
    instance_type='K80',
    instance_count=1,
    instance_name='fuego-gpu-compute',
    environment_name='transformers-text-classification-env',
    requirements_file='./transformers_text_classification/requirements.txt',
    # Script kwargs - these are passed to the script as argparse args
    model_name_or_path='bert-base-cased',
    task_name='mrpc',
    do_train=True,
    do_eval=True,
    max_seq_length=128,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    output_dir='./outputs',
    logging_dir='./logs',
)
```