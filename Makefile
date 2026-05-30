.PHONY: help setup test lint-fix lint-verify clean all

help:
	@echo "Available commands:"
	@echo "  make setup        - Install dependencies and configure pre-commit"
	@echo "  make lint-fix     - Automatically fix style, formatting, and imports"
	@echo "  make lint-verify  - Run all checks in read-only mode (Ruff, Mypy)"
	@echo "  make test         - Run tests (only unit tests in this codebase)"
	@echo "  make all          - Fix, verify, and test"
	@echo "  make clean        - Clean cache and temporary files"

setup:
	@echo "Upgrading pip..."
	python -m pip install --upgrade pip
	@echo "Installing project and dev dependencies..."
	python -m pip install -e ".[dev]"
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Setup complete! Ready to code."

lint-fix:
	@echo "Fixing style, formatting, and imports..."
	ruff check . --fix
	ruff format .

lint-verify:
	@echo "Verifying code quality (read-only)..."
	ruff check .
	ruff format --check .
	mypy src/

test:
	@echo "Running tests..."
	pytest tests/

all: lint-fix lint-verify test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -exec rm -rf {} +
	@echo "Environment cleaned."
