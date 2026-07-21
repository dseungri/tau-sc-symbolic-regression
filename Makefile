.PHONY: help install-base install-dev verify lint format test smoke-test tree clean-generated

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  install-base     Install package with base dependencies"
	@echo "  install-dev      Install package with development dependencies"
	@echo "  verify           Run lint, tests, and synthetic smoke test"
	@echo "  lint             Run ruff checks"
	@echo "  format           Format code with ruff"
	@echo "  test             Run pytest"
	@echo "  smoke-test       Run synthetic smoke test"
	@echo "  tree             Show repository tree"
	@echo "  clean-generated  Remove generated caches and reports, never incoming data"

install-base:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

verify: lint test smoke-test

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

smoke-test:
	$(PYTHON) -m tau_sr smoke-test --output-report reports/smoke_tests/synthetic_smoke_report.json

tree:
	find . -maxdepth 3 -type f | sort

clean-generated:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov build dist
	find logs -type f ! -name README.md ! -name .gitkeep -delete
	find results -type f ! -name README.md ! -name .gitkeep -delete
	find reports -type f \( -name "*.json" -o -name "*.html" -o -name "*.png" \) -delete
