#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from .core import ModelConfig, calculate_equity_irr


def solve_tariff_for_target_irr(
    base_cfg: ModelConfig,
    target_irr: float,
    bracket: Tuple[float, float] = (10.0, 40.0),
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    lo, hi = bracket
    f_lo = calculate_equity_irr(replace(base_cfg, tariff_lkr_per_kwh=lo)) - target_irr
    f_hi = calculate_equity_irr(replace(base_cfg, tariff_lkr_per_kwh=hi)) - target_irr
    if f_lo * f_hi > 0:
        return float("nan")
    mid = (lo + hi) / 2.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = calculate_equity_irr(replace(base_cfg, tariff_lkr_per_kwh=mid)) - target_irr
        if abs(f_mid) < tol:
            break
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid
