.PHONY: setup test lint run clean

setup:
	python3.13 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/python -m pytest tests/

lint:
	.venv/bin/rruff check .

run:
	.venv/bin/python src/main.py

clean:
	rm -rf .venv __pycache__ .pytest_cache