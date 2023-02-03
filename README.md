# fuego

A 🔥 tool for running code in the cloud

🔥[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/fuego/blob/main/examples/fuego_demo.ipynb)🔥

### Note

❗ **This project is a WIP and just an _idea_ right now. Under active development.** ❗

🤗 Suggestions/ideas/feedback from community are welcome! Please feel free to submit a [new idea](https://github.com/huggingface/fuego/discussions/new?category=ideas) in the discussions tab.

## The idea

A nice interface for running scripts on Hugging Face Spaces

## Installation

For now, you can install from source:

```bash
git clone https://github.com/huggingface/fuego.git
cd fuego
pip install -e "."
```

## WIP API

The Python API and CLI should have very similar experiences so folks can use whichever they prefer.

See the examples folder for more details.

#### Python


```python
import fuego

fuego.run(
    script='run.py',
    requirements_file='requirements.txt',
    # Kwargs
    message='hello world',
)
```

#### CLI

```bash
fuego run --script run.py --requirements_file requirements.txt --message "hello world"
```
