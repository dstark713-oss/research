# риск_модели/гарч/garch_model.py
# ============================================================
# GARCH(1,1) с t-распределением.
#
# fit на train (дневные доходности базового портфеля),
# forecast на горизонт HORIZON (20 дней) → annualized vol.
# ============================================================
import warnings

import numpy as np
import pandas as pd

from arch import arch_model

from .config import (
    GARCH_P, GARCH_Q, GARCH_DIST, GARCH_MEAN, GARCH_VOL, GARCH_POWER,
    TRADING_DAYS,
)

warnings.filterwarnings("ignore")


def fit_garch(returns: pd.Series):
    """
    Обучает GARCH(1,1) с t-распределением.
    returns: дневные доходности (в долях, не %).
    Возвращает fitted arch result.
    """
    if len(returns) < 100:
        raise ValueError(f"Мало данных для GARCH: {len(returns)} < 100")
    y = returns.dropna() * 100.0
    am = arch_model(
        y,
        mean=GARCH_MEAN, vol=GARCH_VOL,
        p=GARCH_P, q=GARCH_Q,
        dist=GARCH_DIST, power=GARCH_POWER,
    )
    res = am.fit(disp="off", show_warning=False)
    return res


def forecast_vol_annualized(res, horizon: int = 20) -> float:
    """
    Прогноз волатильности на h дней, annualized.
    """
    f = res.forecast(horizon=horizon, reindex=False)
    var_pct2 = f.variance.values[-1]  # (horizon,) в (%)²
    mean_var_pct2 = float(np.mean(var_pct2))
    daily_var = mean_var_pct2 / (100.0 ** 2)
    daily_vol = np.sqrt(daily_var)
    return float(daily_vol * np.sqrt(TRADING_DAYS))


def conditional_vol_daily(res) -> pd.Series:
    """Условная волатильность (дневная, в долях) in-sample."""
    v = res.conditional_volatility / 100.0
    return pd.Series(v, name="cond_vol_daily")


def garch_persistence(res) -> float:
    """alpha + beta (persistence)."""
    p = res.params
    alpha = float(p.get("alpha[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))
    return alpha + beta