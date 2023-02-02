import fuego


# message here is a kwarg to fuego.run, which is passed to run.py
# since this example takes arguments to main, we specify to unpack them
fuego.run("run.py", unpack_script_args_to_main=True, message="Howdy, world!")
