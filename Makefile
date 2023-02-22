.PHONY: quality style

# Check that source code meets quality standards
quality:
	black --check --diff --preview .
	ruff .

# Format source code automatically
style:
	black .
	ruff --fix .
