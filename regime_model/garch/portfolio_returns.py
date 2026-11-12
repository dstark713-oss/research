# риск_модели/гарч/portfolio_returns.py
# ============================================================
# Восстановление ДНЕВНЫХ доходностей базового RS-портфеля
# (100% экспозиция, без масштабирования).
#
# Логика:
#   - на каждой rebal_date есть holdings (TOP-10, равные веса);
#   - держим до следующей rebal_date;
#   - ежедневная доходность портфеля = среднее дневных доходностей
#     акций в портфеле (равные веса, buy-and-hold между ребалансами);
#   - на rebal_date портфель переключается на новые holdings
#     (доходность rebal_date считается по СТАРОМУ портфелю до close,
#      затем переключение; либо по новому — зависит от execution model).
#
# ВАЖНО: execution model в существующем RS-бэктесте — MOC (close rebal_date).
# Значит:
#   - доходность между rebal_date_k и rebal_date_{k+1} — по новому портфелю;
#   - на rebal_date_k доходность = 0 (переключение по close);
#   - начиная с rebal_date_k+1 — новый портфель.
# ============================================================
import numpy as np
import pandas as pd


def build_daily_returns(close: pd.DataFrame,
                        holdings: dict) -> pd.Series:
    """
    Восстанавливает дневные доходности базового портфеля.
    
    close: wide DataFrame (index=date, cols=ticker).
    holdings: dict rebal_date -> list[ticker].
    
    Возвращает: pd.Series дневных доходностей (index=date), только дни,
    где портфель активен (от первой rebal_date до последней).
    """
    if not holdings:
        raise ValueError("holdings пуст")

    rebal_dates = sorted(holdings.keys())
    daily_ret = close.pct_change()  # дневные доходности каждой акции

    records = []
    for i, rd in enumerate(rebal_dates):
        tickers = holdings[rd]
        # валидные тикеры (есть цена)
        valid = [t for t in tickers if t in daily_ret.columns]
        if len(valid) < len(tickers) * 0.5:
            # если больше половины тикеров пропали — пропускаем период
            continue

        # период удержания: от rd (исключая) до следующей rebal_date (включая?)
        # MOC: на rd переключение по close → доходность rd = 0 (портфель
        # формируется в конце rd, доходность начинается с rd+1).
        # Доходность rd+1..rd_next — по этому портфелю.
        if i + 1 < len(rebal_dates):
            next_rd = rebal_dates[i + 1]
        else:
            next_rd = close.index[-1]

        # дни между rd и next_rd (исключая rd, включая next_rd? — MOC:
        # на next_rd тоже переключение по close, поэтому доходность
        # next_rd принадлежит СЛЕДУЮЩЕМУ портфелю. Значит, период
        # [rd+1, next_rd) — дни, где держим текущий портфель.
        period = daily_ret.loc[(daily_ret.index > rd) & (daily_ret.index < next_rd)]
        if len(period) == 0:
            continue

        # средняя дневная доходность по равным весам
        # (buy-and-hold внутри периода, но для простоты — равные веса
        # каждый день, что эквивалентно ежедневной ребалансировке весов
        # внутри TOP-10; для оценки vol это приемлемо)
        port_daily = period[valid].mean(axis=1)  # равные веса
        for d, r in port_daily.items():
            records.append({"date": d, "ret": r})

    if not records:
        raise RuntimeError("Не удалось построить дневные доходности портфеля")

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    # на случай дубликатов (если периоды пересекаются)
    df = df.drop_duplicates("date", keep="first")
    series = df.set_index("date")["ret"]
    series.name = "port_daily_ret"
    return series


def build_port_rs_at_rebalance(daily_returns: pd.Series,
                               rebal_dates: list) -> pd.Series:
    """
    Агрегирует дневные доходности в 20-дневные RS на датах ребаланса.
    Используется для сверки с rs_is_wf_equity.csv.
    """
    out = {}
    for i, rd in enumerate(rebal_dates):
        if i + 1 < len(rebal_dates):
            next_rd = rebal_dates[i + 1]
        else:
            next_rd = daily_returns.index[-1]
        period = daily_returns.loc[(daily_returns.index > rd) & (daily_returns.index <= next_rd)]
        if len(period) == 0:
            continue
        # cumulative 20d return
        out[rd] = (1 + period).prod() - 1
    return pd.Series(out).sort_index()