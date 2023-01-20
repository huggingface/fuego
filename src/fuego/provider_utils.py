from typing import List, Optional


class Provider:
    """Unified interface for all cloud providers."""

    def __init__(self, **kwargs):
        """Authentication happens here. Internal obj saved as attribute, then used in other methods"""
        raise NotImplementedError

    def create_run(
        self, script: str, instance_name: str, instance_type: str, requirements_file: Optional[str] = None, **kwargs
    ):
        """Submits a run on a compute target. Returns run info"""
        raise NotImplementedError

    def list_runs(self, **kwargs):
        """Returns a list of run objects"""
        raise NotImplementedError

    def create_compute_target(self, name, instance_type, **kwargs):
        """Creates a compute target. Returns info about it"""
        raise NotImplementedError

    def list_compute_targets(self, **kwargs):
        """Returns a list of compute targets w some info about them"""
        raise NotImplementedError

    def delete_compute_target(self, name, **kwargs):
        """Deletes a compute target. Returns info about it?"""
        raise NotImplementedError
