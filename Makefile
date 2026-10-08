PY ?= python

.PHONY: install test numbers figures check

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

numbers:
	$(PY) scripts/headline_numbers.py

figures:
	$(PY) scripts/make_figures.py

check: test numbers figures
	git diff --exit-code -- results/
