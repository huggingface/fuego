from pathlib import Path

import fuego


space_url, dataset_url = fuego.run(
    script=str(Path(__file__).parent / "run.py"),
    delete_space_on_completion=True,  # When debugging, set this to False
    # Kwargs
    message="Howdy, world!!!",
)
print(f"space_url: {space_url}")
print(f"dataset_url: {dataset_url}")
