.PHONY: test test-unit test-integration test-cov lint fmt clean

test:
	.venv/bin/python -m pytest tests/ -v

test-unit:
	.venv/bin/python -m pytest tests/ -v -m unit

test-integration:
	.venv/bin/python -m pytest tests/ -v -m integration

test-cov:
	.venv/bin/python -m pytest tests/ -v --cov=agentsciml --cov-report=term-missing

lint:
	.venv/bin/python -m ruff check src/ tests/

fmt:
	.venv/bin/python -m ruff format src/ tests/

clean:
	rm -rf .pytest_cache __pycache__ .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
