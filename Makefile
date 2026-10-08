PY ?= python

.PHONY: install test numbers figures demo check

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

numbers:
	$(PY) scripts/headline_numbers.py

figures:
	$(PY) scripts/make_figures.py

demo:
	$(PY) scripts/build_demo.py

check: test numbers figures demo
	git diff --exit-code -- results/ demo/index.html
