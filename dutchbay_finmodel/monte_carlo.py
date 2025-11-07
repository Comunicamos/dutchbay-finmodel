#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from .core import ModelConfig, build_financial_model, calculate_irr_robust


def generate_mc_parameters(n: int, base_cfg: ModelConfig, seed: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cf = np.clip(
        rng.normal(base_cfg.capacity_factor_p50, 0.03, size=n),
        0.25,
        0.55,
    )
    capex_mult = np.exp(rng.normal(0.0, 0.15, size=n))
    opex_mult = np.exp(rng.normal(0.0, 0.20, size=n))
    t_esc = np.clip(
        rng.normal(base_cfg.tariff_indexation_pct, 0.01, size=n),
        -0.01,
        0.03,
    )

    return pd.DataFrame(
        {
            "capacity_factor_p50": cf,
            "capex_usd_mn": base_cfg.capex_usd_mn * capex_mult,
            "fixed_opex_lkr_mn_year1": base_cfg.fixed_opex_lkr_mn_year1 * opex_mult,
            "tariff_indexation_pct": t_esc,
        }
    )


def run_monte_carlo(
    n: int,
    base_cfg: ModelConfig,
    param_df: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    if param_df is None:
        param_df = generate_mc_parameters(n, base_cfg, seed=seed)

    results = []
    for i, row in param_df.iterrows():
        cfg = replace(
            base_cfg,
            capacity_factor_p50=float(row["capacity_factor_p50"]),
            capex_usd_mn=float(row["capex_usd_mn"]),
            fixed_opex_lkr_mn_year1=float(row["fixed_opex_lkr_mn_year1"]),
            tariff_indexation_pct=float(row["tariff_indexation_pct"]),
        )
        df, _, eq_cf, _ = build_financial_model(cfg)
        irr = calculate_irr_robust(eq_cf)["irr"]
        min_dscr = float(df["DSCR"].replace([float("inf")], np.nan).min())
        results.append(
            {
                "scenario": int(i),
                "equity_irr": irr,
                "min_dscr": min_dscr,
                "dscr_breach": bool(min_dscr < base_cfg.min_dscr_covenant),
            }
        )

    return pd.DataFrame(results)
