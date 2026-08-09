.PHONY: install lint format format-check typecheck test smoke reproduce security pre-commit clean docker-build all

VENV = .venv
UV = uv

install:
	$(UV) sync --extra dev

lint:
	$(UV) run ruff check src tests scripts

format:
	$(UV) run ruff format src tests scripts

format-check:
	$(UV) run ruff format --check src tests scripts

typecheck:
	$(UV) run mypy src/procureguard

test:
	$(UV) run pytest -q

smoke:
	$(UV) run python scripts/reproduce.py --smoke

reproduce:
	$(UV) run python scripts/reproduce.py
	$(UV) run python scripts/update_readme.py

security:
	$(UV) run bandit -r src/procureguard
	$(UV) run pip-audit --local --skip-editable

pre-commit:
	$(UV) run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ .coverage coverage.xml
	rm -rf reports reports_smoke
	rm -rf data/processed data/raw data/interim

docker-build:
	docker build -t procureguard-ai .

all: install lint format-check typecheck test smoke security
