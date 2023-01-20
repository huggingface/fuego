"""Check presence of installed packages at runtime."""
import sys

import packaging.version

_PY_VERSION: str = sys.version.split()[0].rstrip("+")

if packaging.version.Version(_PY_VERSION) < packaging.version.Version("3.8.0"):
    import importlib_metadata  # type: ignore
else:
    import importlib.metadata as importlib_metadata  # type: ignore


_package_versions = {}

_CANDIDATES = {
    "azure.ai.ml": {"azure-ai-ml"},
}

# Check once at runtime
for candidate_name, package_names in _CANDIDATES.items():
    _package_versions[candidate_name] = "N/A"
    for name in package_names:
        try:
            _package_versions[candidate_name] = importlib_metadata.version(name)
            break
        except importlib_metadata.PackageNotFoundError:
            pass


def _get_version(package_name: str) -> str:
    return _package_versions.get(package_name, "N/A")


def _is_available(package_name: str) -> bool:
    return _get_version(package_name) != "N/A"


# Python
def get_python_version() -> str:
    return _PY_VERSION


# AzureML SDKv2 (azure.ai.ml)
def is_azureml_available() -> bool:
    return _is_available("azure.ai.ml")


def get_azureml_version() -> str:
    return _get_version("azure.ai.ml")
