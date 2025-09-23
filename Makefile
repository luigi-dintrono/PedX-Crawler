# PedX Crawler - Road-Crossing Video Discovery Makefile

.PHONY: help install setup run clean test run-filtered test-filtered clean-temp

# Default target
help:
	@echo "PedX Crawler - Road-Crossing Video Discovery - Available targets:"
	@echo "  install       - Install Python dependencies"
	@echo "  setup         - Copy .env.example to .env (you need to add your API key)"
	@echo "  run           - Run the PedX crawler script"
	@echo "  run-verbose   - Run with verbose output"
	@echo "  run-filtered  - Run with quality filter enabled"
	@echo "  test          - Test the script with a small sample"
	@echo "  test-filtered - Test with quality filter (small sample)"
	@echo "  clean         - Clean output files"
	@echo "  clean-temp    - Clean temporary files from quality filter"

# Install dependencies
install:
	python3 -m pip install -r crawler/requirements.txt

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
	python3 crawler/pedx-crawler.py

# Run with verbose output
run-verbose:
	python3 crawler/pedx-crawler.py --verbose

# Clean output files
clean:
	rm -f data/outputs/*.csv

# Test with a small sample
test:
	python3 crawler/pedx-crawler.py --per-city 5 --verbose

# Test with custom date and per-city limit
test-custom:
	python3 crawler/pedx-crawler.py --since 2024-01-01 --per-city 10 --verbose

# Run with quality filter enabled
run-filtered:
	python3 crawler/pedx-crawler.py --enable-quality-filter --verbose

# Test with quality filter (small sample)
test-filtered:
	python3 crawler/pedx-crawler.py --enable-quality-filter --per-city 2 --verbose

# Test with quality filter and custom date
test-filtered-recent:
	python3 crawler/pedx-crawler.py --enable-quality-filter --since 2025-01-01 --per-city 2 --verbose

# Clean temporary files from quality filter
clean-temp:
	rm -rf crawler/tmp/*

# CI-friendly run (assumes API key is set via environment)
ci-run:
	python3 crawler/pedx-crawler.py --verbose