# backtests/rm_tuning/horizon_runner.py
# ============================================================
# Прогон WF для одного горизонта ребаланса.
# ИСПОЛЬЗУЕТ run_rs_wf из rs_is_wf_v2 — никакого дублирования.
# ============================================================
from backtests.portfolio_tests.rs_is_wf_v2 import run_rs_wf


def run_horizon(split_date: str,
                horizon: int,
                verbose: bool = True) -> dict:
    """
    Прогон WF с заданным rebalance_step.
    Возвращает метрики.
    """
    result = run_rs_wf(split_date, rebalance_step=horizon, verbose=verbose)
    metrics = result["metrics"]
    return {
        "horizon": horizon,
        "CAGR": metrics.get("CAGR"),
        "MDD": metrics.get("MDD"),
        "VOL": metrics.get("VOL"),
        "Sharpe": metrics.get("Sharpe"),
        "Sortino": metrics.get("Sortino"),
        "Calmar": metrics.get("Calmar"),
        "n_days": len(result["daily_ret"]),
    }