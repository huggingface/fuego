import fuego


space_url, dataset_url = fuego.run(
    script="run.py",
    requirements_file="requirements.txt",
    space_hardware="t4-small",
    delete_space_on_completion=True,  # When debugging, set this to False
    downgrade_hardware_on_completion=True,  # When debugging, set this to False
)
print(f"space_url: {space_url}")
print(f"dataset_url: {dataset_url}")
