# [Project Name]

[One sentence describing what this project does.]

This repository is built with a focus on simplicity, maintainability, and strict code quality — designed to be easy to navigate for anyone reviewing the codebase.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core Architecture Dependencies](#core-architecture-dependencies)
- [Project Structure & Design Principles](#project-structure--design-principles)
- [Code Quality & CI/CD](#code-quality--cicd)

---

## Quick Start

Make sure you are inside your virtual environment, then run:

```bash
make setup    # Create data directories, install dependencies and configure pre-commit
make all      # Lint, verify, and test in one command
```

### Commands

Common tasks are available via `make` to simplify the developer experience.

```bash
make lint-fix      # Auto-fix formatting, style, and import order
make lint-verify   # Read-only checks — what CI runs
make test-unit     # Run only unit tests
make test-e2e      # Run end-to-end and non-regression tests
make test          # Run the entire test suite (unit + E2E)
make all           # lint-fix → lint-verify → test
make clean         # Remove all cache directories
make docs-serve    # Preview the documentation locally
make docs-deploy   # Deploy the documentation to GitHub Pages
```

---

## Core Architecture Dependencies

This production template relies on two lightweight, decoupled core internal libraries to manage pipeline execution and configuration:

- [dataconf-manager](https://github.com/MaximeRighini/dataconf-manager): Automates multi-layer YAML configuration deep-merging with dynamic text anchor resolution, and abstracts local file system read/write operations for Polars DataFrames.
- [pipeline-orchestrator](https://github.com/MaximeRighini/pipeline-orchestrator): An execution engine that automatically resolves, sequences, and traces atomic functional steps (`Nodes`) based on their Python type signatures while managing automatic I/O data dumping.

> **Modular Framework Note**: These dependencies are completely optional. If you are building an ultra-lightweight project that does not require runtime profile overrides (e.g., multi-environment or multi-market contexts) or formal execution tracing and tracking, you can strip both entries out of your `pyproject.toml` file and delete the `config/`, `src/nodes/`, and `src/pipelines/` directories.

---

## Project Structure & Design Principles

This repository follows a strict separation of concerns, keeping each component focused and easy to navigate.

```text
├── config/              # Flexible YAML configuration layers (e.g., general/, env/)
├── Makefile             # Central command interface (local & CI)
├── pyproject.toml       # Single-source dependency and tool configuration
├── src/
│   ├── constants.py     # Schema definitions and soft-coded string mappings
│   ├── utils/           # Shared, atomic helper functions
│   ├── modules/         # Domain-driven processing logic
│   ├── nodes/           # Optional: Atomic execution units wrapping business logic
│   ├── pipelines/       # Optional: Ordered node sequences with auto I/O resolution
│   └── run.py           # Single orchestration entry point
└── tests/               # Pytest test suite mirroring the src/ structure
```

### 1. Configuration (`config/`)

Powered by `dataconf-manager`, the `config/` directory does not have a "set in stone" structure. \
It uses a multi-layer deep merge strategy defined by a `priority_order` (e.g., `["general", "market", "env"]` for instance).

- Later layers override earlier ones without erasing entire sub-dictionaries.
- String values support dynamic anchor resolution (e.g., `data/{market}/output.parquet`).

*For full configuration rules, refer to the `dataconf-manager` repository.*

### 2. Orchestration (`src/nodes/` & `src/pipelines/`)

Powered by `pipeline-orchestrator`, this pattern removes boilerplate and separates business logic from I/O operations:

- **Nodes**: A `Node` wraps a single python function. It handles no I/O. Its input dependencies are automatically inferred from the function's arguments, and its outputs are explicitly declared.
- **Pipelines**: A `Pipeline` is an ordered sequence of nodes. It injects the config, manages the shared memory context, and handles automatic data loading and dumping. If an output key matches a path defined in `config/data`.yaml, the pipeline automatically saves it to disk.

*For advanced usage (fail-fast behavior, naming conventions), refer to the `pipeline-orchestrator` repository.*

---

## Code Quality & CI/CD

This codebase enforces code quality at three stages of development:

1. **`make lint-verify`** runs Ruff and Mypy in read-only mode. Run it regularly during development to catch linting and type errors before they accumulate.

2. **Pre-commit hooks** act as a safety net on every commit, ensuring broken or badly formatted code never reaches the remote repository.

3. **GitHub Actions** triggers automatically on every push. It runs the same environment and the same checks as local development, so there are no surprises. Any pull request that fails linting or tests is blocked from being merged.
