from dutchbay_finmodel.core import (
    ModelConfig,
    build_financial_model,
    calculate_dscr_series,
    calculate_llcr_plcr,
    evaluate_covenants,
)


def test_core_shapes_and_signs():
    cfg = ModelConfig()
    df, proj_cf, eq_cf, cap = build_financial_model(cfg)

    assert proj_cf.shape[0] == cfg.economic_life_years + 1
    assert eq_cf.shape[0] == cfg.economic_life_years + 1
    assert df.shape[0] == cfg.economic_life_years

    assert proj_cf[0] < 0
    assert eq_cf[0] < 0
    assert cap["total_capex_lkr"] == cap["total_debt_lkr"] + cap["total_equity_lkr"]

    dscr = calculate_dscr_series(cfg)
    assert dscr.shape[0] == cfg.economic_life_years


def test_llcr_plcr_and_covenants():
    cfg = ModelConfig()
    cov = evaluate_covenants(cfg)
    ratios = calculate_llcr_plcr(cfg)

    assert ratios["llcr"] > 0
    assert ratios["plcr"] > 0
    assert isinstance(cov["dscr_breach_years"], list)
    assert isinstance(cov["lockup_years"], list)
