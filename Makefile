.PHONY: help install install-dev precommit-install format lint test precommit build check clean

UV ?= uv
VENV_BIN ?= .venv/bin
PYTHON ?= $(VENV_BIN)/python
PRECOMMIT ?= $(VENV_BIN)/pre-commit
PYTEST ?= $(VENV_BIN)/pytest
RUFF ?= $(VENV_BIN)/ruff
PYTEST_ARGS ?=

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install:  ## Create the environment and install mm-kit
	$(UV) venv --python 3.11
	$(UV) pip install --python $(PYTHON) -e .

install-dev:  ## Install mm-kit with test, lint, and build tools
	$(UV) venv --python 3.11
	$(UV) pip install --python $(PYTHON) -e '.[dev]'

precommit-install:  ## Install the Git pre-commit hook
	$(PRECOMMIT) install

format:  ## Format and import-sort every Python file
	$(RUFF) format .
	$(RUFF) check --fix --select I .

lint:  ## Check formatting and strict Ruff rules
	$(RUFF) format --check .
	$(RUFF) check .

test:  ## Run the test suite; pass extra options with PYTEST_ARGS
	$(PYTEST) -q $(PYTEST_ARGS)

precommit:  ## Run every pre-commit hook against every tracked file
	$(PRECOMMIT) run --all-files

build:  ## Build source and wheel distributions
	$(PYTHON) -m build

check: lint test precommit  ## Run all release checks

clean:  ## Remove local build and Python cache artifacts
	rm -rf build dist .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
