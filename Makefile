.PHONY: help install data demo run report downscale dashboard test lint clean

PYTHON ?= python3
export PYTHONPATH := src:.

help:
	@echo "install    install the package with dev extras"
	@echo "data       generate and describe the synthetic record"
	@echo "demo       fast pipeline + static HTML report (~30 s)"
	@echo "run        full pipeline with spatial CV and scenario sweep (~3 min)"
	@echo "report     rebuild outputs/report.html from existing outputs"
	@echo "downscale  evaluate learned super-resolution vs interpolation"
	@echo "dashboard  launch the Streamlit prototype (needs streamlit)"
	@echo "test       run the test suite"
	@echo "lint       ruff check"
	@echo "clean      remove generated outputs and caches"

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/make_synthetic_data.py

demo:
	$(PYTHON) scripts/run_pipeline.py --fast
	$(PYTHON) scripts/make_report.py
	@echo "open outputs/report.html"

run:
	$(PYTHON) scripts/run_pipeline.py --scenario
	$(PYTHON) scripts/make_report.py
	@echo "open outputs/report.html"

report:
	$(PYTHON) scripts/make_report.py

downscale:
	$(PYTHON) scripts/evaluate_downscaling.py --plot

dashboard:
	streamlit run dashboard/app.py

test:
	$(PYTHON) -m pytest

lint:
	ruff check src scripts dashboard tests

clean:
	rm -rf outputs/*.csv outputs/*.json outputs/*.html outputs/figures
	rm -f data/processed/synthetic_daily.csv
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
