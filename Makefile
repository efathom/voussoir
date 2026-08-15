.PHONY: lint format typecheck test ci install-hooks docs-build docs-serve

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

typecheck:
	mypy src/voussoir

test:
	pytest

ci: lint typecheck test

install-hooks:
	pre-commit install
	bash scripts/install_hooks.sh

docs-build:
	.venv/bin/mkdocs build --strict

docs-serve:
	.venv/bin/mkdocs serve
