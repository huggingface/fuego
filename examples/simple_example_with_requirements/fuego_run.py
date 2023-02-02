import fuego


# since this example takes arguments to main, we specify to unpack them
fuego.run(
    script="run.py",
    requirements_file="requirements.txt",
    unpack_script_args_to_main=True,
    # Kwargs
    message="Howdy, world!",
)
