---
title: GENCO Contingency Verifier
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: What GENCO predicted and what the solver decided
---

# GENCO Contingency Verifier

An explorer for the 870 N-k contingencies of the frozen test in *Trust, but verify: making GENCO safe for N-k contingency
screening* (GENCO Challenge, GridFM Porto 2026). For each case it shows GENCO's prediction, the unresolved-case warning,
the Newton-Raphson check and the final outcome, on a map of the Texas 2,000-bus grid.

`index.html` is a single static page with the data built in (it loads one web font). It is built from the repository's result files by `python scripts/build_demo.py`
(template: `app.html`). To deploy, copy `index.html` and this README to a static Hugging Face Space.
