# PedX Crawler - Road-Crossing Video Discovery Makefile

# Interpreter used by every target. Override it when 'python3' is not on PATH —
# notably on Windows, where 'python3' is usually a Microsoft Store stub and the
# real launcher is 'py':   make test-unit PYTHON=py
PYTHON ?= python3

.PHONY: help install install-dev setup run run-verbose dry-run clean test test-unit test-custom run-filtered test-filtered test-filtered-recent bridge clean-temp ci-run run-yolo test-yolo run-internvl3 test-internvl3

# Default target
help:
	@echo "PedX Crawler - Road-Crossing Video Discovery - Available targets:"
	@echo "  install       - Install Python dependencies"
	@echo "  install-dev   - Install dependencies plus the test extras (pytest)"
	@echo "  setup         - Copy .env.example to .env (you need to add your API key)"
	@echo "  run           - Run the PedX crawler script"
	@echo "  run-verbose   - Run with verbose output"
	@echo "  dry-run       - Preview the plan and quota cost without spending a unit"
	@echo "  run-filtered  - Run with quality filter enabled"
	@echo "  bridge        - Convert discovery.csv into PedX-Insight's mapping CSVs"
	@echo "  test          - Test the script with a small sample (live crawl)"
	@echo "  test-unit     - Run the offline pytest suite (no API key / network)"
	@echo "  test-filtered - Test with quality filter (small sample)"
	@echo "  run-yolo      - Run with YOLO quality filter"
	@echo "  test-yolo     - Test with YOLO quality filter (small sample)"
	@echo "  run-internvl3 - Run with InternVL3 quality filter"
	@echo "  test-internvl3- Test with InternVL3 quality filter (small sample)"
	@echo "  clean         - Clean output files"
	@echo "  clean-temp    - Clean temporary files from quality filter"
	@echo ""
	@echo "On Windows, add PYTHON=py if 'python3' is not available."

# Install dependencies
install:
	$(PYTHON) -m pip install -r crawler/requirements.txt

# Install dependencies plus the test extras
install-dev:
	$(PYTHON) -m pip install -r crawler/requirements-dev.txt

# Setup environment
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please add your YouTube API key to .env"; \
	else \
		echo ".env file already exists"; \
	fi

# Run PedX crawler
run:
	$(PYTHON) crawler/pedx-crawler.py

# Run with verbose output
run-verbose:
	$(PYTHON) crawler/pedx-crawler.py --verbose

# Preview the plan and quota estimate without any API calls
dry-run:
	$(PYTHON) crawler/pedx-crawler.py --dry-run

# Clean output files
clean:
	rm -f data/outputs/*.csv

# Test with a small sample
test:
	$(PYTHON) crawler/pedx-crawler.py --per-city 5 --verbose

# Run the offline unit-test suite (stubs external deps; no API key needed)
test-unit:
	$(PYTHON) -m pytest crawler/tests -q

# Test with custom date and per-city limit
test-custom:
	$(PYTHON) crawler/pedx-crawler.py --since 2024-01-01 --per-city 10 --verbose

# Run with quality filter enabled
run-filtered:
	$(PYTHON) crawler/pedx-crawler.py --enable-quality-filter --verbose

# Test with quality filter (small sample)
test-filtered:
	$(PYTHON) crawler/pedx-crawler.py --enable-quality-filter --per-city 2 --verbose

# Test with quality filter and custom date
test-filtered-recent:
	$(PYTHON) crawler/pedx-crawler.py --enable-quality-filter --since 2025-01-01 --per-city 2 --verbose

# Convert the crawler's discovery CSV into PedX-Insight's input CSVs.
# Override the paths as needed, e.g.
#   make bridge INSIGHT=../PedX-Insight DISCOVERY=data/outputs/discovery.csv
DISCOVERY ?= data/outputs/discovery.csv
INSIGHT ?= ../PedX-Insight
bridge:
	$(PYTHON) scripts/discovery_to_mapping.py \
		--discovery $(DISCOVERY) \
		--out-mapping-one-each $(INSIGHT)/mapping_one_each.csv \
		--out-mapping $(INSIGHT)/mapping.csv \
		--append

# Clean temporary files from quality filter.
# When run from the repo root the default --temp-dir ('tmp') resolves to
# ./tmp, so clean both that and the legacy crawler/tmp location.
clean-temp:
	rm -rf tmp/* crawler/tmp/*

# CI-friendly run (assumes API key is set via environment)
ci-run:
	$(PYTHON) crawler/pedx-crawler.py --verbose

# Run with YOLO quality filter
run-yolo:
	$(PYTHON) crawler/pedx-crawler.py --filter yolo --verbose --cities-file data/cities.txt

# Test with YOLO quality filter (small sample)
test-yolo:
	$(PYTHON) crawler/pedx-crawler.py --filter yolo --verbose --per-city 3 --cities-file data/cities.txt

# Run with InternVL3 quality filter
run-internvl3:
	$(PYTHON) crawler/pedx-crawler.py --filter internvl3 --verbose --cities-file data/cities.txt --device cpu --threshold 0.9

# Test with InternVL3 quality filter (small sample)
test-internvl3:
	$(PYTHON) crawler/pedx-crawler.py --filter internvl3 --verbose --per-city 3 --cities-file data/cities.txt --device cpu --threshold 0.9
