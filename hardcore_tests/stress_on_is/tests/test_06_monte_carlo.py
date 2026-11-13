# test_06_monte_carlo.py
import time
import numpy as np
import pandas as pd
from numba import njit
from ..common.data_loader import load_frozen_equity
from ..common.metrics import compute_metrics
from ..common.reporting import save_report
from ..config import MC_SIMULATIONS, MC_SEED, MC_BLOCK_SIZE

@njit
def _block_bootstrap(returns: np.ndarray, block_size: int, n_sims: int, seed: int):
    np.random.seed(seed)
    n = len(returns)
    n_blocks = n // block_size
    out = np.empty((n_sims, n))
    for i in range(n_sims):
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        for j in range(n_blocks):
            s = starts[j]
            out[i, j*block_size:(j+1)*block_size] = returns[s:s+block_size]
    return out

def run(verbose=True) -> dict:
    t0 = time.time()
    frozen = load_frozen_equity()
    base = pd.Series(frozen["sized_ret"].values, index=pd.DatetimeIndex(frozen["date"]))
    arr = base.dropna().values
    sims = _block_bootstrap(arr, MC_BLOCK_SIZE, MC_SIMULATIONS, MC_SEED)
    mdds = []
    for i in range(sims.shape[0]):
        eq = np.cumprod(1 + sims[i])
        cummax = np.maximum.accumulate(eq)
        dd = eq / cummax - 1
        mdds.append(dd.min())
    mdds = np.array(mdds)
    lines = ["# Test 06: Monte Carlo",
             "\n## Методика",
             f"Block bootstrap: {MC_SIMULATIONS} симуляций, block={MC_BLOCK_SIZE}.",
             "\n## MDD распределение",
             f"- median: {np.median(mdds)*100:+.2f}%",
             f"- p90: {np.percentile(mdds, 90)*100:+.2f}%",
             f"- p95: {np.percentile(mdds, 95)*100:+.2f}%",
             f"- p99: {np.percentile(mdds, 99)*100:+.2f}%",
             f"- worst: {mdds.min()*100:+.2f}%",
             f"\nВремя: {time.time()-t0:.1f}s"]
    save_report("test_06_monte_carlo", lines)

    from ..common.reporting import plot_hist
    plot_hist(mdds * 100, "Test 06: Monte Carlo MDD distribution",
              "test_06_mdd_hist", "MDD, %")

    return {"mdds": mdds}