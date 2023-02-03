import fuego


# Since we use the default unpack_script_args_to_main=False, the kwargs are used
# to override sys.argv, which assumes the `main` function in `run_glue.py` handles argparsing.
space_url, dataset_url = fuego.run(
    script="run_glue.py",
    requirements_file="requirements.txt",
    space_hardware="t4-medium",
    # Kwargs
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
