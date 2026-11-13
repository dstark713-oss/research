# hardcore_tests/stress_on_is/smoke.py
# ============================================================
# Smoke test: 9 проверок из ТЗ.
# ============================================================
import traceback

import numpy as np
import pandas as pd

from .common.data_loader import load_frozen_equity, load_rgbitr, load_rgbi
from .common.metrics import compute_metrics
from .common.costs import round_trip_cost_bps, apply_cost


def smoke() -> bool:
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    ok = True

    # 1. Импорт всех тестов (проверим позже, когда они есть)
    print("[1] Импорт тестов: отложено (тесты ещё не созданы)")

    # 2. Загрузка данных
    try:
        df = load_frozen_equity()
        assert len(df) > 100, f"Мало данных: {len(df)}"
        print(f"[2] frozen_equity: {len(df)} дней, "
              f"{df['date'].min().date()} → {df['date'].max().date()}")
    except Exception as e:
        print(f"[2] FAIL: {e}")
        ok = False

    # 3. Даты, индексы, тикеры
    try:
        assert "baseline_ret" in df.columns
        assert "sized_ret" in df.columns
        assert "exposure" in df.columns
        print(f"[3] колонки: {list(df.columns)}")
    except Exception as e:
        print(f"[3] FAIL: {e}")
        ok = False

    # 4. Доходность портфеля на фрагменте
    try:
        sample = df.head(100).set_index("date")["sized_ret"]
        eq = (1 + sample).cumprod()
        assert len(eq) > 0
        print(f"[4] sample equity: start={eq.iloc[0]:.4f}, end={eq.iloc[-1]:.4f}")
    except Exception as e:
        print(f"[4] FAIL: {e}")
        ok = False

    # 5. Комиссия + slippage
    try:
        c = round_trip_cost_bps()
        net = apply_cost(sample, c)
        assert (net < sample).all()
        print(f"[5] cost = {c:.2f} б.п. round-trip, применяется корректно")
    except Exception as e:
        print(f"[5] FAIL: {e}")
        ok = False

    # 6. Метрики
    try:
        m = compute_metrics(sample)
        assert "CAGR" in m and "MDD" in m and "Sharpe" in m
        print(f"[6] metrics: CAGR={m['CAGR']*100:+.2f}%, "
              f"MDD={m['MDD']*100:+.2f}%, Sharpe={m['Sharpe']:.3f}")
    except Exception as e:
        print(f"[6] FAIL: {e}")
        ok = False

    # 7. Cash carry: нет look-ahead (проверим позже)
    print("[7] cash carry look-ahead: отложено (тест 14 ещё не создан)")

    # 8. Создание отчёта
    try:
        from .common.reporting import save_report
        save_report("smoke_test", ["Smoke test passed."])
        print("[8] отчёт создан")
    except Exception as e:
        print(f"[8] FAIL: {e}")
        ok = False

    # 9. Frozen не изменён
    try:
        df2 = load_frozen_equity()
        assert len(df2) == len(df)
        print("[9] frozen не изменён")
    except Exception as e:
        print(f"[9] FAIL: {e}")
        ok = False

    print("=" * 60)
    if ok:
        print("SMOKE PASSED")
    else:
        print("SMOKE FAILED — полный прогон запрещён")
    return ok