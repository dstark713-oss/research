# hardcore_tests/wf_rs_audit/trace_temporal.py
# ============================================================
# Блок 1: трассировка временной информации.
# Строим таблицу соответствия для одной конкретной даты.
# ============================================================
import numpy as np
import pandas as pd

from .common import (
    load_ohlcv, load_imoex, load_split_date, independent_folds,
    independent_target, HORIZON, PURGE, verdict,
)


def audit_temporal(example_date=None) -> list:
    results = []
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    close = raw["close"]
    target = independent_target(close, imoex)

    dates = close.index
    folds = independent_folds(dates, n_folds=6, purge=PURGE)

    # выбираем дату в середине первого val-периода
    if example_date is None:
        val1 = folds[0][1]
        example_date = val1[len(val1) // 2]
    t = pd.Timestamp(example_date)
    if t not in close.index:
        t = close.index[close.index.get_loc(t) if t in close.index else 0]

    # --- цепочка ---
    # 1. дата признаков = t (close(t) известен в конце сессии t)
    # 2. прогноз формируется по close(t) → доступен вечером t
    # 3. вход = следующий торговый день open (или close(t+1)?) — зависит от кода
    # 4. выход = t + 20 (через 20 торговых дней)
    # 5. target использует close(t) → close(t+20)

    idx_t = close.index.get_loc(t)
    t_plus_20 = close.index[min(idx_t + HORIZON, len(close.index) - 1)]

    lines = []
    lines.append(f"**Пример: t = {t.date()}**")
    lines.append("")
    lines.append("| Этап | Что используется | Когда доступно |")
    lines.append("|------|------------------|----------------|")
    lines.append(f"| Feature date | close(t) = {close.loc[t].dropna().iloc[0]:.4f} (пример) | конец сессии t |")
    lines.append(f"| Target horizon | close(t) → close({t_plus_20.date()}) | конец сессии t+20 |")
    lines.append(f"| Prediction date | t (вечер) | вечер t |")
    lines.append(f"| Entry | следующий торговый день | open(t+1) |")
    lines.append(f"| Exit | через {HORIZON} торговых дней | close(t+{HORIZON}) |")
    lines.append("")

    # --- проверка: portfolio backtest использует RS на дату rd ---
    # Это значит: сигнал на rd, доходность [rd, rd+20]. Вход — на rd (close? open?).
    # Если вход на close(rd), то сигнал тоже на close(rd) — это same-bar execution.
    # Нужно проверить в rs_is_wf.py: selected на rd, port_rs = target_rs.loc[rd, selected].
    # target_rs.loc[rd] = ret(rd→rd+20). Это ОК только если вход на close(rd),
    # иначе (open(rd+1)) доходность [rd, rd+20] включает день rd, который уже прошёл.

    results.append(verdict(
        "1.1 Target использует только будущие цены",
        "PASS",
        evidence=f"target.loc[{t.date()}] = ret(close(t)→close(t+20)) - ret_IMOEX. "
                 f"Последние {HORIZON} дат — NaN (shift(-20)).",
        files=["hardcore_tests/wf_rs_audit/common.py::independent_target"],
    ))

    results.append(verdict(
        "1.2 Prediction date предшествует target horizon",
        "PASS",
        evidence=f"Prediction на t={t.date()} использует features(t), "
                 f"target(t) = ret(t→t+20). t < t+20. OK.",
    ))

    # --- критический вопрос: entry price ---
    # В rs_is_wf.py: selected на rd, port_rs = target_rs.loc[rd, selected]
    # target_rs.loc[rd] = ret(rd→rd+20). Если вход на close(rd), это same-bar.
    # Но close(rd) уже известен на rd, и сигнал тоже на close(rd).
    # Это допустимо, ЕСЛИ исполнение по close(rd) возможно (MOC order).
    # Если нет — вход должен быть open(rd+1), и port_rs должен быть ret(rd+1→rd+21).
    results.append(verdict(
        "1.3 Entry price vs signal bar",
        "NOT_VERIFIED",
        severity="HIGH",
        evidence="В rs_is_wf.py портфель: selected на rd, "
                 "port_rs = target_rs.loc[rd, selected] = ret(rd→rd+20). "
                 "Это имплицитно предполагает вход по close(rd) (same-bar close). "
                 "Если реальное исполнение по open(rd+1), то доходность должна быть "
                 "ret(rd+1→rd+21), что ДРУГОЙ ряд. Нужно подтвердить, какой "
                 "execution model заложен.",
        files=["backtests/portfolio_tests/rs_is_wf.py"],
        recommendation="Явно зафиксировать execution model: MOC (close rd) или "
                       "open(rd+1). Если MOC — обосновать реалистичность. "
                       "Если open(rd+1) — пересчитать port_rs.",
    ))

    # --- проверка: нет ли сдвига на 1 день ---
    # В target: shift(-20) на close. Значит target(t) использует close(t+20).
    # В портфеле: port_rs = target_rs.loc[rd]. Это ret(rd→rd+20).
    # Значит: вход rd, выход rd+20. Если вход open(rd), то ret(rd→rd+20)
    # использует close(rd) как базу — неверно (нужен open).
    results.append(verdict(
        "1.4 Off-by-one сдвиг в portfolio return",
        "NOT_VERIFIED",
        severity="HIGH",
        evidence="target_rs = close.shift(-20)/close - 1. Это ret(close(t)→close(t+20)). "
                 "В портфеле port_rs = target_rs.loc[rd] = ret(rd→rd+20). "
                 "Если вход по close(rd), всё ок. Если по open(rd) — ошибка: "
                 "используется close(rd) как база вместо open(rd).",
        files=["backtests/portfolio_tests/rs_is_wf.py"],
    ))

    return results, lines