from dutchbay_finmodel.core import ModelConfig, evaluate_covenants


def test_covenant_metrics_non_negative_sweep():
    cfg = ModelConfig()
    cov = evaluate_covenants(cfg)
    assert cov["total_swept_cash_lkr"] >= 0.0
