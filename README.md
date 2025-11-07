# Dutch Bay 150 MW Wind Farm - Hardened Financial Model

This repository contains a lender-grade Python implementation of the Dutch Bay 150 MW wind project model.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pre-commit install
pytest -q
```

## Lender Case

The file `dutchbay_finmodel/configs/dutchbay_lendercase_2025Q4.yaml` encodes the lender case.
The golden regression test `tests/test_regression_golden.py` asserts IRR/DSCR/LLCR/PLCR
consistency for that case.

## Usage example

```python
from dutchbay_finmodel import load_config, build_financial_model, calculate_equity_irr

cfg = load_config("dutchbay_finmodel/configs/dutchbay_lendercase_2025Q4.yaml")
df, proj_cf, eq_cf, cap = build_financial_model(cfg)
print(calculate_equity_irr(cfg))
```
