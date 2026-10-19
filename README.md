# news-mlops

Dynamic news classification as a production ML system (KSE MLOps course project, built over five labs).

## The problem in five sentences

Newsrooms and aggregators need every incoming article routed into a category (politics, business, wellness and so on) so that readers, feeds and editors can find it. The topics and language of news change over time, so a classifier trained once degrades as new events and vocabulary appear. We build a system that ingests articles in monthly batches, trains and validates a classifier with a time-based split, serves predictions, and monitors for drift. When the monitored signal crosses a threshold, a human is alerted and decides whether to retrain and promote a new model. The goal is not a state-of-the-art model but a system that is reproducible, integrated, observed and understood end to end.

Data: HuffPost News Category Dataset (headlines, short descriptions, categories and publication dates; the dense period 2012 to 2017 is used). See `docs/design.md`.

## Quick start
make setup # install the pinned environment with uv

make lint # run ruff

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | `docs/design.md` follows the template, every section filled or marked as an assumption | Lab 1 |
