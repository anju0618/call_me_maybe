.PHONY: install run debug clean fclean lint lint-strict

SRC_DIR = src
VENV = .venv
PYTHON = $(VENV)/bin/python3
FLAKE8 = $(VENV)/bin/flake8
MYPY = $(VENV)/bin/mypy

install:
	uv sync

run:
	$(PYTHON) -m $(SRC_DIR)

debug:
	$(PYTHON) -m pdb -m $(SRC_DIR)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf src/__pycache__
	rm -rf .python-version
	rm -rf llm_sdk/llm_sdk/__pycache__
	rm -rf data/output

fclean: clean
	rm -rf $(VENV)

lint:
	$(FLAKE8) $(SRC_DIR)
	$(MYPY) --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC_DIR)

lint-strict:
	$(FLAKE8) $(SRC_DIR)
	$(MYPY) --explicit-package-bases --strict $(SRC_DIR)

test:
	@echo "Running test suite with pytest..."
	uv run python -m pytest tests/ -v