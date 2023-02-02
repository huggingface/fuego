from setuptools import find_packages, setup


def get_version() -> str:
    rel_path = "src/fuego/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


requirements = [
    "fire",
    "huggingface_hub>=0.12.0",
    "GitPython",
]

setup(
    name="fuego",
    description="Fuego",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huggingface/fuego",
    version=get_version(),
    author="Nathan Raw",
    author_email="nate@huggingface.com",
    license="Apache",
    install_requires=requirements,
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={"console_scripts": ["fuego=fuego.run_on_spaces:cli_run"]},
)
