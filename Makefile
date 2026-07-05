.PHONY: install run debug clean lint lint-strict

SRC_DIR = src

install:
		uv sync

run:
		uv run python3 -m $(SRC_DIR)

debug:
		uv run python3 -m pdb -m $(SRC_DIR)

clean:
		find . -type d -name "__pycache__" -exec rm -rf {} +
		rm -rf .mypy_cache
		rm -rf .ruff_cache
		rm -rf src/__pycache__

lint:
		uv run flake8 $(SRC_DIR)
		uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC_DIR)

lint-strict:
		uv run flake8 $(SRC_DIR)
		uv run mypy --strict $(SRC_DIR)