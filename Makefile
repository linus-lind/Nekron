.PHONY: help venv install install-dev install-hooks lint format-check type-check test config-check check clean mlflow-ui

VENV := .venv

ifeq ($(OS),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
	PYTHON_SYS := py -3.13
	PYTHON := $(VENV_BIN)/python.exe
	TORCH_INDEX_ARGS := --extra-index-url https://download.pytorch.org/whl/cu126
else
	VENV_BIN := $(VENV)/bin
	PYTHON_SYS := python3.13
	PYTHON := $(VENV_BIN)/python
	TORCH_INDEX_ARGS :=
endif

PIP := $(PYTHON) -m pip

help: ## Show available project tasks
	@$(PYTHON_SYS) -c "import re; from pathlib import Path; rows=[m.groups() for line in Path('Makefile').read_text().splitlines() if (m:=re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line))]; print('\n'.join(f'  {name:<18} {description}' for name, description in rows))"

$(PYTHON):
	$(PYTHON_SYS) -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(PYTHON) ## Create the Python 3.13 virtual environment

install: venv ## Install runtime dependencies in editable mode
	$(PIP) install -e . $(TORCH_INDEX_ARGS)

install-dev: venv ## Install runtime and development dependencies
	$(PIP) install -e ".[dev]" $(TORCH_INDEX_ARGS)
	"$(MAKE)" install-hooks

install-hooks: ## Install repository-local pre-commit and pre-push hooks
	$(PYTHON) -m pre_commit install --config .git-hooks-config.yaml -t pre-commit -t pre-push

lint: ## Run Ruff without modifying files
	$(PYTHON) -m ruff check src tests scripts

format-check: ## Verify Ruff formatting
	$(PYTHON) -m ruff format --check src tests scripts

type-check: ## Run strict Mypy over the package
	$(PYTHON) -m mypy src scripts

test: ## Run the complete unit and integration test suite
	$(PYTHON) -m pytest tests

config-check: ## Compose every Hydra config group to catch a broken YAML
	$(PYTHON) scripts/check_configs.py

check: lint format-check type-check config-check test ## Run the complete local quality gate

mlflow-ui: ## Open the local MLflow tracking UI
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1

clean: ## Remove disposable caches
	$(PYTHON) scripts/clean_workspace.py

.DEFAULT_GOAL := help
