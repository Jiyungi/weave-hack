# WeaveSelf — Python ML package

This package hosts the Python (Track A / Track B) side of WeaveSelf: serving, training,
data pipeline, orchestration, eval, and the shared Track 0 data contracts.

## Layout

```
weaveself/
  contracts/      # Track 0 shared data models (adapter file format, schemas)
  serving/        # Track A serving engine + inference API
  training/       # Track A NKT-Mirror train_adapter
  data/           # Track B data pipeline + curation
  orchestration/  # Track B LangGraph nightly-batch graph
  eval/           # Track B Weave eval
tests/            # Python test suite
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```
