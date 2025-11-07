from dutchbay_finmodel.core import ModelConfig
from dutchbay_finmodel.monte_carlo import generate_mc_parameters, run_monte_carlo


def test_generate_mc_parameters_basic():
    cfg = ModelConfig()
    params = generate_mc_parameters(100, cfg, seed=42)
    assert len(params) == 100
    for col in [
        "capacity_factor_p50",
        "capex_usd_mn",
        "fixed_opex_lkr_mn_year1",
        "tariff_indexation_pct",
    ]:
        assert col in params.columns


def test_run_monte_carlo_basic():
    cfg = ModelConfig()
    results = run_monte_carlo(50, cfg, seed=123)
    assert len(results) == 50
    assert {"equity_irr", "min_dscr", "dscr_breach"} <= set(results.columns)
