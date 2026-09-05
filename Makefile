.PHONY: install lint format typecheck test test-all models demo

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

test:
	pytest -m "not integration"

test-all:
	pytest

models:
	affectlab models download

demo:
	affectlab image tests/data/astronaut.png --save docs/images/demo_astronaut.png
