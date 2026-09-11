.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv and install everything
	uv sync --all-groups

test:  ## Run the suite - no downloads, no network
	uv run pytest -q

lint:  ## Lint and check formatting
	uv run ruff check shared tests projects
	uv run ruff format --check shared tests projects

fmt:  ## Auto-format
	uv run ruff format shared tests projects
	uv run ruff check --fix shared tests projects

sources:  ## Re-verify every dataset URL still serves
	uv run python shared/data.py --check

all:  ## Run all ten projects end to end
	@for d in projects/*/; do echo "=== $$d"; uv run python $$d/run.py || exit 1; done

.PHONY: help install test lint fmt sources all
