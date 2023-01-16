from pathlib import Path

import fire


def main(message: str = "Hello, world!"):
    logdir = Path("./logs")
    logdir.mkdir(exist_ok=True, parents=True)
    outfile_path = logdir / "message.txt"
    outfile_path.write_text(message)


if __name__ == "__main__":
    fire.Fire(main)
