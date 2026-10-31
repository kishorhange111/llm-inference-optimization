# Common developer commands. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format test check clean kv

help:           ## list the available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:        ## this package + GPU benchmarking extras
	pip install -e ".[gpu]"

install-dev:    ## lint/test tooling + this package (editable)
	pip install -e ".[dev]"

lint:           ## ruff lint + format check (what CI runs)
	ruff check .
	ruff format --check .

format:         ## auto-fix lint findings and reformat
	ruff check --fix .
	ruff format .

test:           ## run the test suite (CPU only)
	pytest

check: lint test ## everything CI runs

kv:             ## KV-cache capacity table (no GPU needed)
	python scripts/run_experiments.py kv

clean:          ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
