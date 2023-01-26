# fuego

A 🔥 tool for running code in the cloud

### Note

❗ **This project is a WIP and just an _idea_ right now. Under active development.** ❗

🤗 Suggestions/ideas/feedback from community are welcome!

## The idea

A unified interface to view/manage runs, compute instances, data, etc. across your authenticated cloud providers.

## Roadmap

Currently, this project only works for AzureML training runs. The idea is to support other cloud providers (AWS, GCP, etc.) and other run types (inference, data preparation, etc.) in the future.

- [x] Add AzureML Provider + ability to do basic training runs
- [ ] Dataset upload
- [ ] Ability to attach datasets to runs
- [ ] Add Sagemaker Provider
- [ ] Run from GitHub repos instead of local files?
- [ ] Multi-GPU examples
- [ ] Multi-Node examples


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
