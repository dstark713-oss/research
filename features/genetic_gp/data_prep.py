# features/genetic_gp/data_prep.py
# ============================================================
# Подготовка панели для gplearn.
#
# gplearn работает с 2D (n_samples, n_features). Мы "сплющиваем"
# панель (n_dates * n_tickers, n_features), но сохраняем:
#   - date_ids: для каждой строки — номер даты (для cross-sectional IC)
#   - feature_names: имена колонок
#
# Все фичи — causality-preserving, считаются только на IS.
# Преобразованные OHLCV: доходности, range, volume ratios и т.д.
# ============================================================
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 20
ROOT = Path(__file__).parent.parent.parent
STOCKS_DB = ROOT / "dataset" / "sql" / "stocks.sqlite"
INDEXES_DB = ROOT / "indexes" / "sql" / "indexes.sqlite"
SPLIT_JSON = ROOT / "backtests" / "split_info.json"


def load_split_date():
    return json.loads(SPLIT_JSON.read_text())["split_date"]


def load_raw(split_date):
    conn = sqlite3.connect(STOCKS_DB)
    df = pd.read_sql(
        "SELECT ticker, date, open, high, low, close, volume FROM prices WHERE date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = df.pivot(index="date", columns="ticker", values=col).sort_index()
    return out


def load_imoex(split_date):
    conn = sqlite3.connect(INDEXES_DB)
    df = pd.read_sql(
        "SELECT date, close FROM indexes WHERE ticker='IMOEX' AND date < ?",
        conn, params=(split_date,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def build_features(raw: dict) -> dict:
    """
    Преобразованные OHLCV + базовые фичи (кроме mom_20_rel).
    Все — causality-preserving.
    """
    close = raw["close"]
    high = raw["high"]
    low = raw["low"]
    open_ = raw["open"]
    volume = raw["volume"]
    ret1 = close.pct_change(1)

    feats = {}

    # --- доходности разных горизонтов ---
    feats["ret_1"] = ret1
    feats["ret_5"] = close.pct_change(5)
    feats["ret_20"] = close.pct_change(20)
    feats["ret_60"] = close.pct_change(60)

    # --- волатильность ---
    feats["vol_20"] = ret1.rolling(20).std()
    feats["vol_60"] = ret1.rolling(60).std()

    # --- положение в диапазоне ---
    roll_max = close.rolling(252, min_periods=100).max()
    roll_min = close.rolling(252, min_periods=100).min()
    feats["price_position"] = (close - roll_min) / (roll_max - roll_min)
    feats["dist_from_high"] = close / roll_max - 1
    feats["dist_from_low"] = close / roll_min - 1

    # --- форма распределения ---
    feats["skew_60"] = ret1.rolling(60).skew()
    feats["kurt_60"] = ret1.rolling(60).kurt()

    # --- объёмные ---
    vol_ma20 = volume.rolling(20).mean()
    feats["rel_volume"] = volume / vol_ma20
    feats["volume_zscore"] = (volume - volume.rolling(60).mean()) / volume.rolling(60).std()
    feats["dollar_volume"] = close * volume
    feats["amihud"] = (ret1.abs() / volume.replace(0, np.nan)).rolling(20).mean()

    # --- OHLCV-трансформации ---
    feats["range_hl"] = (high - low) / close
    feats["close_to_high"] = close / high
    feats["close_to_low"] = close / low
    feats["open_to_close"] = open_ / close
    feats["high_to_low"] = high / low

    # --- rolling sharpe ---
    feats["rolling_sharpe_60"] = ret1.rolling(60).mean() / ret1.rolling(60).std()

    # --- momentum (только 60d, без 20d) ---
    imoex = load_imoex(load_split_date())
    imoex_ret60 = imoex.pct_change(60).reindex(close.index)
    feats["mom_60_rel"] = close.pct_change(60).sub(imoex_ret60, axis=0)

    return feats


def build_target(raw: dict, imoex: pd.Series) -> pd.DataFrame:
    """RS_{i,t} = ret_i(t→t+h) - ret_IMOEX(t→t+h). Последние HORIZON дней = NaN."""
    close = raw["close"]
    fwd = close.shift(-HORIZON) / close - 1
    imoex_fwd = imoex.shift(-HORIZON) / imoex - 1
    imoex_fwd = imoex_fwd.reindex(close.index)
    return fwd.sub(imoex_fwd, axis=0)


def flatten_panel(features: dict, target: pd.DataFrame):
    """
    Сплющивает панель в 2D для gplearn.
    Возвращает:
      X: np.ndarray (n_samples, n_features)
      y: np.ndarray (n_samples,)
      date_ids: np.ndarray (n_samples,) — номер даты для cross-sectional IC
      feature_names: list
      dates: DatetimeIndex
      tickers: Index
    """
    # выравниваем по датам и тикерам
    common_dates = target.index
    common_tickers = target.columns

    rows = []
    y_list = []
    date_ids = []

    # только даты, где таргет не NaN хотя бы у 20 акций
    valid_dates = []
    for d in common_dates:
        t = target.loc[d]
        if t.notna().sum() >= 20:
            valid_dates.append(d)

    date_to_id = {d: i for i, d in enumerate(valid_dates)}

    for d in valid_dates:
        t = target.loc[d]
        # фичи на эту дату
        f_dict = {}
        for name, df in features.items():
            if d in df.index:
                f_dict[name] = df.loc[d, common_tickers]
            else:
                f_dict[name] = pd.Series(np.nan, index=common_tickers)

        f_df = pd.DataFrame(f_dict, index=common_tickers)
        # оставляем только тикеры с не-NaN таргетом
        mask = t.notna()
        if mask.sum() < 20:
            continue
        f_df = f_df.loc[mask]
        t_sel = t[mask]

        # заполняем NaN фич медианой по дате (для gplearn)
        f_df = f_df.fillna(f_df.median())

        rows.append(f_df.values)
        y_list.append(t_sel.values)
        date_ids.append(np.full(len(t_sel), date_to_id[d]))

    if not rows:
        raise RuntimeError("flatten_panel: нет валидных данных")

    X = np.vstack(rows)
    y = np.concatenate(y_list)
    date_ids = np.concatenate(date_ids)

    # убираем Inf
    X = np.where(np.isfinite(X), X, np.nan)
    y = np.where(np.isfinite(y), y, np.nan)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    date_ids = date_ids[mask]

    return X, y, date_ids, list(features.keys()), valid_dates, common_tickers