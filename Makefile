.PHONY: quality style

# Check that source code meets quality standards

quality:
	black --check . src
	isort --check-only . src
	flake8 . src

# Format source code automatically

style:
	black . src
	isort . src