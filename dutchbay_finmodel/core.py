#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    # Technical
    nameplate_mw: float = 150.0
    economic_life_years: int = 20
    ppa_term_years: int = 20
    capacity_factor_p50: float = 0.40
    degradation_rate: float = 0.006

    # Tariff & FX
    tariff_lkr_per_kwh: float = 20.30
    tariff_indexation_pct: float = 0.00
    fx_lkr_per_usd: float = 330.0
    fx_escalation_pct: float = 0.03

    # Losses & curtailment & fees
    loss_factor: float = 0.97
    curtailment_pct: float = 0.02
    wheeling_charge_lkr_per_kwh: float = 0.0
    other_fees_pct_of_revenue: float = 0.0

    # CAPEX & OPEX
    capex_usd_mn: float = 155.0
    capex_spend_profile: Tuple[float, float] = (1.0, 0.0)
    fixed_opex_lkr_mn_year1: float = 900.0
    fixed_opex_escalation_pct: float = 0.02
    variable_opex_pct_of_revenue: float = 0.0

    # Capital structure
    max_debt_ratio: float = 0.80
    usd_debt_ratio_of_debt: float = 0.45
    usd_dfi_pct: float = 0.10
    usd_mkt_pct: float = 0.90
    usd_dfi_rate: float = 0.065
    usd_mkt_rate: float = 0.070
    lkr_debt_rate: float = 0.075

    # Debt terms
    debt_tenor_years: int = 15
    grace_years: int = 1

    # Tax
    tax_rate: float = 0.0

    # Covenants
    min_dscr_covenant: float = 1.15

    def capex_lkr_total(self) -> float:
        return self.capex_usd_mn * 1e6 * self.fx_lkr_per_usd

    def total_debt_lkr(self) -> float:
        return self.capex_lkr_total() * self.max_debt_ratio

    def total_equity_lkr(self) -> float:
        return self.capex_lkr_total() * (1.0 - self.max_debt_ratio)


def _generation_profile_mwh(cfg: ModelConfig) -> np.ndarray:
    hours = 8760.0
    gen = []
    for year in range(1, cfg.economic_life_years + 1):
        cf = cfg.capacity_factor_p50 * ((1.0 - cfg.degradation_rate) ** (year - 1))
        gen.append(cfg.nameplate_mw * hours * cf)
    return np.array(gen)


def _net_generation_profile_mwh(cfg: ModelConfig) -> np.ndarray:
    return _generation_profile_mwh(cfg) * cfg.loss_factor * (1.0 - cfg.curtailment_pct)


def _tariff_profile_lkr(cfg: ModelConfig) -> np.ndarray:
    vals = []
    for year in range(1, cfg.economic_life_years + 1):
        if year <= cfg.ppa_term_years:
            vals.append(cfg.tariff_lkr_per_kwh * (1.0 + cfg.tariff_indexation_pct) ** (year - 1))
        else:
            vals.append(0.0)
    return np.array(vals)


def _revenue_lkr(cfg: ModelConfig) -> np.ndarray:
    gen = _net_generation_profile_mwh(cfg)
    tariff = _tariff_profile_lkr(cfg)
    gross = gen * 1000.0 * tariff
    wheeling = gen * 1000.0 * cfg.wheeling_charge_lkr_per_kwh
    fees = gross * cfg.other_fees_pct_of_revenue
    return gross - wheeling - fees


def _opex_lkr(cfg: ModelConfig, revenue: np.ndarray) -> np.ndarray:
    fixed = []
    for year in range(1, cfg.economic_life_years + 1):
        fixed.append(
            cfg.fixed_opex_lkr_mn_year1 * 1e6 * (1.0 + cfg.fixed_opex_escalation_pct) ** (year - 1)
        )
    fixed = np.array(fixed)
    variable = revenue * cfg.variable_opex_pct_of_revenue
    return fixed + variable


def _build_debt_schedules(cfg: ModelConfig) -> Dict[str, np.ndarray]:
    years = cfg.economic_life_years
    n = cfg.debt_tenor_years
    g = cfg.grace_years

    total_debt = cfg.total_debt_lkr()
    usd_debt = total_debt * cfg.usd_debt_ratio_of_debt
    lkr_debt = total_debt - usd_debt
    usd_dfi_debt = usd_debt * cfg.usd_dfi_pct
    usd_mkt_debt = usd_debt * cfg.usd_mkt_pct

    def leg(opening: float, rate: float):
        p = np.zeros(years)
        it = np.zeros(years)
        if opening <= 0.0:
            return p, it
        if n <= g:
            for t in range(min(n, years)):
                it[t] = opening * rate
            return p, it
        annual = opening / float(n - g)
        bal = opening
        for t in range(years):
            if t < n:
                if t < g:
                    it[t] = bal * rate
                else:
                    pay = min(annual, bal)
                    p[t] = pay
                    it[t] = bal * rate
                    bal -= pay
        return p, it

    udp, udi = leg(usd_dfi_debt, cfg.usd_dfi_rate)
    ump, umi = leg(usd_mkt_debt, cfg.usd_mkt_rate)
    lp, li = leg(lkr_debt, cfg.lkr_debt_rate)

    tp = udp + ump + lp
    ti = udi + umi + li
    ds = tp + ti

    return {
        "usd_dfi_principal": udp,
        "usd_dfi_interest": udi,
        "usd_mkt_principal": ump,
        "usd_mkt_interest": umi,
        "lkr_principal": lp,
        "lkr_interest": li,
        "total_principal": tp,
        "total_interest": ti,
        "total_debt_service": ds,
    }


def _npv(rate: float, cf: np.ndarray, start: int = 0) -> float:
    periods = np.arange(start, start + len(cf))
    return float(np.sum(cf / (1.0 + rate) ** periods))


def _irr_bruteforce(cf: np.ndarray, lo: float = -0.5, hi: float = 1.5, steps: int = 20001) -> float:
    rates = np.linspace(lo, hi, steps)
    npvs = np.array([_npv(r, cf) for r in rates])
    signs = np.sign(npvs)
    idx = np.where(np.diff(signs) != 0)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    r0, r1 = rates[i], rates[i + 1]
    v0, v1 = npvs[i], npvs[i + 1]
    if v1 == v0:
        return float(r0)
    return float(r0 + (0.0 - v0) * (1.0 * (r1 - r0) / (v1 - v0)))


def calculate_irr_robust(cf: np.ndarray) -> Dict[str, float]:
    try:
        import numpy_financial as nf  # type: ignore

        irr = float(nf.irr(cf))
        if not np.isfinite(irr):
            raise ValueError
        return {"irr": irr, "method": "numpy_financial.irr"}
    except Exception:
        return {"irr": _irr_bruteforce(cf), "method": "bruteforce_linear"}


def _wacd(cfg: ModelConfig) -> float:
    td = cfg.total_debt_lkr()
    if td <= 0.0:
        return 0.0
    usd = td * cfg.usd_debt_ratio_of_debt
    lkr = td - usd
    u_dfi = usd * cfg.usd_dfi_pct
    u_mkt = usd * cfg.usd_mkt_pct
    return (u_dfi * cfg.usd_dfi_rate + u_mkt * cfg.usd_mkt_rate + lkr * cfg.lkr_debt_rate) / td


def build_financial_model(cfg: ModelConfig):
    years = cfg.economic_life_years

    revenue = _revenue_lkr(cfg)
    opex = _opex_lkr(cfg, revenue)
    debt = _build_debt_schedules(cfg)

    ebit = revenue - opex
    tax = np.where(ebit > 0.0, ebit * cfg.tax_rate, 0.0)
    cfads = ebit - tax - debt["total_interest"]
    cfads = np.maximum(cfads, -1e20)

    ds = debt["total_debt_service"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dscr = np.where(ds > 0.0, cfads / ds, np.inf)

    proj_cf = np.zeros(years + 1)
    proj_cf[0] = -cfg.capex_lkr_total()
    proj_cf[1:] = cfads

    eq_cf = np.zeros(years + 1)
    eq_cf[0] = -cfg.total_equity_lkr()
    eq_cf[1:] = cfads - ds

    cap = {
        "total_capex_lkr": cfg.capex_lkr_total(),
        "total_debt_lkr": cfg.total_debt_lkr(),
        "total_equity_lkr": cfg.total_equity_lkr(),
    }

    df = pd.DataFrame(
        {
            "Year": np.arange(1, years + 1),
            "Revenue_LKR": revenue,
            "Opex_LKR": opex,
            "EBIT_LKR": ebit,
            "Tax_LKR": tax,
            "CFADS_LKR": cfads,
            "DebtService_LKR": ds,
            "DSCR": dscr,
            "EquityCF_LKR": eq_cf[1:],
        }
    )

    return df, proj_cf, eq_cf, cap


def calculate_project_irr(cfg: ModelConfig) -> float:
    _, proj_cf, _, _ = build_financial_model(cfg)
    return float(calculate_irr_robust(proj_cf)["irr"])


def calculate_equity_irr(cfg: ModelConfig) -> float:
    _, _, eq_cf, _ = build_financial_model(cfg)
    return float(calculate_irr_robust(eq_cf)["irr"])


def calculate_dscr_series(cfg: ModelConfig) -> np.ndarray:
    df, _, _, _ = build_financial_model(cfg)
    return df["DSCR"].values


def calculate_llcr_plcr(cfg: ModelConfig, discount_rate: float | None = None) -> Dict[str, float]:
    df, _, _, _ = build_financial_model(cfg)
    debt = _build_debt_schedules(cfg)

    cfads = df["CFADS_LKR"].values
    ds = debt["total_debt_service"]
    td = cfg.total_debt_lkr()

    if td <= 0.0:
        return {"llcr": float("inf"), "plcr": float("inf")}

    if discount_rate is None:
        discount_rate = _wacd(cfg)

    years = np.arange(1, len(cfads) + 1)

    mask = ds > 0.0
    if mask.any():
        npv_debt = float(np.sum(cfads[mask] / (1.0 + discount_rate) ** years[mask]))
        llcr = npv_debt / td
    else:
        llcr = float("inf")

    npv_full = float(np.sum(cfads / (1.0 + discount_rate) ** years))
    plcr = npv_full / td

    return {"llcr": llcr, "plcr": plcr}


def evaluate_covenants(
    cfg: ModelConfig,
    min_dscr: float | None = None,
    lockup_dscr: float = 1.20,
    sweep_trigger_dscr: float = 1.30,
    sweep_pct: float = 0.50,
    discount_rate: float | None = None,
) -> Dict[str, Any]:
    df, _, _, _ = build_financial_model(cfg)
    debt = _build_debt_schedules(cfg)

    cfads = df["CFADS_LKR"].values
    ds = debt["total_debt_service"]
    dscr = df["DSCR"].values

    if min_dscr is None:
        min_dscr = cfg.min_dscr_covenant

    dscr_finite = np.where(np.isfinite(dscr), dscr, np.nan)

    breach_idx = np.where(dscr_finite < min_dscr)[0]
    lockup_idx = np.where((dscr_finite >= min_dscr) & (dscr_finite < lockup_dscr))[0]

    cov = calculate_llcr_plcr(cfg, discount_rate=discount_rate)

    sweeps = np.zeros_like(cfads)
    mask = (ds > 0.0) & (dscr >= sweep_trigger_dscr)
    sweeps[mask] = sweep_pct * np.maximum(cfads[mask] - ds[mask], 0.0)

    return {
        "min_dscr": float(np.nanmin(dscr_finite)),
        "dscr_breach_years": (breach_idx + 1).tolist(),
        "lockup_years": (lockup_idx + 1).tolist(),
        "llcr": float(cov["llcr"]),
        "plcr": float(cov["plcr"]),
        "total_swept_cash_lkr": float(sweeps.sum()),
    }
