from argparse import ArgumentParser
from pathlib import Path


def main(message: str = "Hello, world!", output_dir="./outputs"):
    logdir = Path(output_dir)
    logdir.mkdir(exist_ok=True, parents=True)
    outfile_path = logdir / "message.txt"
    outfile_path.write_text(message)


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--message", type=str, default="Hello, world!")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    return parser.parse_args(args=args)


if __name__ == "__main__":
    main(**vars(parse_args()))
