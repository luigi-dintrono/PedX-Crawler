# PedX Crawler - road-user video discovery

# Interpreter used by every target. Override it when 'python3' is not on PATH -
# notably on Windows, where 'python3' is usually a Microsoft Store stub and the
# real launcher is 'py':   make test-unit PYTHON=py
PYTHON ?= python3

CITIES ?= data/cities.txt
OUTPUT ?= data/outputs/discovery.csv
DISCOVERY ?= data/outputs/discovery.csv
INSIGHT ?= ../PedX-Insight

.PHONY: help install install-yolo install-internvl3 install-eval install-dev setup \
        run run-verbose dry-run run-yolo run-internvl3 smoke test test-unit \
        bridge label-sample evaluate clean clean-temp ci-run

help:
	@echo "PedX Crawler - available targets:"
	@echo "  install            - core dependencies (crawl + bridge, no ML)"
	@echo "  install-yolo       - core + YOLO visual filter"
	@echo "  install-internvl3  - core + InternVL3 visual filter"
	@echo "  install-eval       - core + filter-evaluation script deps"
	@echo "  install-dev        - core + test deps (pytest)"
	@echo "  setup              - copy .env.example to .env (add your YouTube API key)"
	@echo "  dry-run            - print the search plan + quota estimate, no API calls"
	@echo "  run                - crawl with the metadata filter (profile: dashcam)"
	@echo "  run-yolo           - crawl + YOLO visual filter"
	@echo "  run-internvl3      - crawl + InternVL3 visual filter"
	@echo "  smoke              - tiny live crawl (2 per city, greedy sampling) to check your setup"
	@echo "  test-unit          - offline pytest suite (no API key / network)"
	@echo "  bridge             - convert discovery.csv into PedX-Insight's mapping CSVs"
	@echo "  label-sample       - draw a stratified sample of a run's decisions to hand-label"
	@echo "  evaluate           - precision / recall from labelled decisions"
	@echo "  clean              - delete run provenance (data/outputs/runs) and temp frames"
	@echo ""
	@echo "Override variables, e.g.  make run CITIES=data/cities_batch.txt PYTHON=py"

install:
	$(PYTHON) -m pip install -e .

install-yolo:
	$(PYTHON) -m pip install -e ".[yolo]"

install-internvl3:
	$(PYTHON) -m pip install -e ".[internvl3]"

install-eval:
	$(PYTHON) -m pip install -e ".[eval]"

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please add your YouTube API key to .env"; \
	else \
		echo ".env file already exists"; \
	fi

dry-run:
	$(PYTHON) crawler/pedx-crawler.py --dry-run --cities-file $(CITIES)

run:
	$(PYTHON) crawler/pedx-crawler.py --cities-file $(CITIES) --output $(OUTPUT) --append --verbose

run-verbose: run

run-yolo:
	$(PYTHON) crawler/pedx-crawler.py --filter yolo --cities-file $(CITIES) --output $(OUTPUT) --append --verbose

run-internvl3:
	$(PYTHON) crawler/pedx-crawler.py --filter internvl3 --cities-file $(CITIES) --output $(OUTPUT) --append --verbose

# Tiny live crawl into a throwaway file; costs ~100-200 quota units per city.
smoke:
	$(PYTHON) crawler/pedx-crawler.py --cities-file $(CITIES) --output data/outputs/smoke.csv \
		--per-city 2 --sampling greedy --max-pages-per-city 2 --verbose

test: smoke

test-unit:
	$(PYTHON) -m pytest

# Convert the crawler's discovery CSV into PedX-Insight's input CSVs.
bridge:
	$(PYTHON) scripts/discovery_to_mapping.py \
		--discovery $(DISCOVERY) \
		--out-mapping-one-each $(INSIGHT)/mapping_one_each.csv \
		--out-mapping $(INSIGHT)/mapping.csv \
		--append

# Filter validation: sample -> label by hand -> evaluate.  DECISIONS=data/outputs/runs/<id>/decisions.csv
DECISIONS ?=
LABELS ?= labels.csv
label-sample:
	$(PYTHON) scripts/evaluate_filter.py sample --decisions $(DECISIONS) --out $(LABELS)

evaluate:
	$(PYTHON) scripts/evaluate_filter.py evaluate --decisions $(DECISIONS) --labels $(LABELS)

# Deletes only regenerable run records and scratch frames. The committed reference CSVs
# in data/outputs/ are NOT touched (use git for those).
clean:
	rm -rf data/outputs/runs tmp crawler/tmp

clean-temp:
	rm -rf tmp/* crawler/tmp/*

ci-run:
	$(PYTHON) crawler/pedx-crawler.py --cities-file $(CITIES) --output $(OUTPUT) --append
