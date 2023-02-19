import fuego


space_url, dataset_url = fuego.github_run(
    github_repo_id="pytorch/examples",
    script="vae/main.py",
    requirements_file="vae/requirements.txt",
    space_output_dirs=["./results"],
    # Kwargs, passed as argparse args to the script
    epochs=3,
)
print(f"Space: {space_url}")
print(f"Dataset: {dataset_url}")
