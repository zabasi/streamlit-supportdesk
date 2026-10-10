.PHONY: setup test lint demo web clean
PY ?= python3
SAMPLE = SUPPORTDESK_CONFIG=configs/sample.toml

setup:   ## install with the dev and live extras
	$(PY) -m pip install -e ".[dev,live]"

test:    ## offline, free
	$(PY) -m pytest

lint:
	ruff check src tests

demo:    ## on the synthetic sample: index, evaluate both pipelines, compare, snapshot each
	$(SAMPLE) supportdesk index
	$(SAMPLE) supportdesk eval --pipeline baseline --split dev --run-id baseline-dev
	$(SAMPLE) supportdesk eval --pipeline advanced --split dev --run-id advanced-dev
	$(SAMPLE) supportdesk report baseline-dev advanced-dev
	$(SAMPLE) supportdesk snapshot --all

web:     ## web UI on the sample
	$(SAMPLE) supportdesk web

clean:
	rm -rf artifacts/runs artifacts/reports artifacts_sample index index_sample .pytest_cache .ruff_cache
