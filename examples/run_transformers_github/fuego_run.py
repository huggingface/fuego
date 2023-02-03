import fuego

space_url, dataset_url = fuego.github_run(
    github_repo_id="huggingface/transformers",
    script="examples/pytorch/text-classification/run_glue.py",
    requirements_file="examples/pytorch/text-classification/requirements.txt",
    space_hardware="t4-small",
    # Adding additional pip requirements to the requirements.txt file
    extra_requirements=["tensorboard", "git+https://github.com/huggingface/transformers@ea55bd8#egg=transformers"],
    # Kwargs, passed as argparse args to the script
    model_name_or_path="bert-base-cased",
    task_name="mrpc",
    do_train=True,
    do_eval=True,
    max_seq_length=128,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    output_dir="./outputs",
    logging_dir="./logs",
    logging_steps=20,
    report_to="tensorboard",
)
print(f"Space: {space_url}")
print(f"Dataset: {dataset_url}")
