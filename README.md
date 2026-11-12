# LLM-Class

Course project: reducing inference cost of a BERT prompt-injection classifier
(dynamic padding, early exit, token pruning, GQA conversion, TF-IDF cascade).
See `PLAN.md` for the full plan.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python src/prepare_data.py
```

Data: see `data/README.md` (MIT, S-Labs).
