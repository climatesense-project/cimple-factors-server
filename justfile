# Justfile for CIMPLE Factors Server
# Install just: https://github.com/casey/just

# ============================================================================
# Default Command
# ============================================================================

# Default recipe to display help
default:
    @just --list

# ============================================================================
# Setup and Installation Commands
# ============================================================================

# Install dependencies
install:
    uv sync

# Setup development environment
setup-dev: install
    uv sync --group dev
    @just pre-commit-install

# Install pre-commit hooks
pre-commit-install:
    uv run pre-commit install
    uv run pre-commit install --hook-type commit-msg

# ============================================================================
# Development and Quality Commands
# ============================================================================

# Run code formatting
format:
    uv run ruff format app
    uv run ruff check --fix app

# Run all quality checks
check:
    uv run ruff check app
    uv run ty check

# Run pre-commit on all files
pre-commit-all:
    uv run pre-commit run --all-files

# Run tests
test FILE="":
    #!/usr/bin/env bash
    if [ -n "{{FILE}}" ]; then
        uv run pytest "{{FILE}}" -v
    else
        uv run pytest tests/ -v
    fi

# ============================================================================
# Runtime Commands
# ============================================================================

# Run the server
run CONFIG:
    uv run python -m app.main
