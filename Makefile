.PHONY: install run test test-ui lint build-ui build

# Convenience wrapper. For a one-command install without touching your system
# Python, prefer:   pipx run humangpt
install:
	python -m pip install -e ".[dev]"

run:
	python -m humangpt

test:
	python -m pytest -q
	$(MAKE) test-ui

test-ui:
	cd ui && npm install --no-audit --no-fund && npm test

lint:
	python -m ruff check .

build-ui:
	cd ui && npm install --no-audit --no-fund && npm run build

build:
	python -m pip install --upgrade build twine
	python -m build
	python -m twine check dist/*
	python -m twine upload dist/*