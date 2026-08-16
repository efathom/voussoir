.PHONY: lint format typecheck test ci install-hooks docs-build docs-serve \
        run-01 run-02 run-02b run-03 run-04-validator run-04-publisher run-04-caller \
        run-05-observability run-05-otlp run-05-cascade

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

# Examples — copy .env.example → .env and fill in real values first; the loader
# sources .env so the a2a peer example shares one secret across both terminals.
run-01: ; bash scripts/run_example.sh 01_hello_agent
run-02: ; bash scripts/run_example.sh 02_research_agent
run-02b: ; bash scripts/run_example.sh 02b_research_agent_yase
run-03: ; bash scripts/run_example.sh 03_multi_agent_research
run-04-validator: ; bash scripts/run_example.sh 04_validator_judge
run-04-publisher: ; bash scripts/run_example.sh 04_a2a_peer publisher.py
run-04-caller: ; bash scripts/run_example.sh 04_a2a_peer caller.py
run-05-observability: ; bash scripts/run_example.sh 05_observability console_exporter_demo.py
run-05-otlp: ; bash scripts/run_example.sh 05_observability otlp_phoenix_demo.py
run-05-cascade: ; bash scripts/run_example.sh 05_observability streaming_cascade_demo.py
