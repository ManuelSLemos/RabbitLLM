.PHONY: install dev lint format test test-cov typecheck clean bash

install:
	uv sync

dev: install

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

test:
	uv run pytest tests/

test-cov:
	uv run pytest tests/ --cov=rabbitllm --cov-report=term-missing

typecheck:
	uv run mypy src/rabbitllm/ --ignore-missing-imports

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov/ .coverage .coverage.*

bash:
	docker run --gpus all --rm -it -v $(PWD):/app -w /app python:3.12 bash
