# features/genetic_gp/guards_data.py
# ============================================================
# Тесты на lookahead bias и data leakage. Падают при нарушении.
#
# Тест 1 (lookahead_features): каждая фича на дату T не должна
#   меняться, если мы «обрежем» данные после T.
# Тест 2 (target_boundary): последние HORIZON дней IS имеют NaN в таргете.
# Тест 3 (target_no_future): таргет на дату T не должен меняться,
#   если обрезать данные после T (кроме дат, где таргет NaN — их не трогаем).
# Тест 4 (flatten_alignment): при сплющивании панели date_ids
#   соответствует датам, и порядок не перемешивается.
# Тест 5 (fitness_no_target_leak): fitness не учитывает строки с NaN таргетом.
# ============================================================
import numpy as np
import pandas as pd

from .data_prep import (
    HORIZON, load_raw, load_imoex, build_features, build_target, flatten_panel,
)


class LookaheadError(Exception):
    pass


class DataLeakError(Exception):
    pass


def _truncate_raw(raw: dict, cutoff_date) -> dict:
    """Обрезает все OHLCV-панели до cutoff_date (не включая)."""
    out = {}
    for k, df in raw.items():
        out[k] = df.loc[df.index < cutoff_date]
    return out


def _truncate_series(s: pd.Series, cutoff_date) -> pd.Series:
    return s.loc[s.index < cutoff_date]


def test_features_lookahead(split_date: str, n_check_dates: int = 20, seed: int = 0):
    """
    Для нескольких дат T: строим фичи на полных данных и на данных,
    обрезанных до T. Значения фич на датах < T должны совпадать.

    ВАЖНО: обрезаем СТРОГО до T, чтобы фичи на T не зависели от данных >= T.
    Фича на дату T — это информация, доступная на момент T (close(T)),
    поэтому cutoff = T + 1 день для OHLCV.
    """
    raw = load_raw(split_date)
    imoex = load_imoex(split_date)
    features_full = build_features(raw)

    dates = raw["close"].index
    if len(dates) < 300:
        raise RuntimeError("слишком мало дат для lookahead-теста")

    rng = np.random.RandomState(seed)
    # берём даты в середине ряда, чтобы было достаточно истории
    candidates = dates[300:-HORIZON]
    check_dates = rng.choice(candidates, size=min(n_check_dates, len(candidates)),
                              replace=False)
    check_dates = sorted(check_dates)

    print(f"[guard] lookahead: проверяем {len(check_dates)} дат")

    for T in check_dates:
        # обрезаем до T + 1 день (включая сам T — чтобы фича на T была посчитана)
        cutoff = T + pd.Timedelta(days=1)
        raw_tr = _truncate_raw(raw, cutoff)
        features_tr = build_features(raw_tr)

        for name, df_full in features_full.items():
            if name not in features_tr:
                continue
            df_tr = features_tr[name]
            # сравниваем на датах <= T (включительно)
            common_idx = df_full.index.intersection(df_tr.index)
            common_idx = common_idx[common_idx <= T]
            if len(common_idx) == 0:
                continue
            a = df_full.loc[common_idx].values
            b = df_tr.loc[common_idx].values
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() == 0:
                continue
            diff = np.abs(a[m] - b[m]).max()
            if diff > 1e-8:
                raise LookaheadError(
                    f"lookahead в фиче '{name}' на дате {T.date()}: "
                    f"max diff = {diff:.2e}"
                )
    print("[guard] lookahead: OK")


def test_target_lookahead(split_date: str, n_check_dates: int = 10, seed: int = 0):
    """
    Таргет на дату T (RS за [T, T+H]) не должен меняться при обрезке
    данных после T+H. То есть если cutoff > T+H, таргет на T должен быть
    одинаковым.
    """
    raw = load_raw(split_date)
    imoex = load_imoex(split_date)
    target_full = build_target(raw, imoex)

    dates = raw["close"].index
    if len(dates) < 300:
        raise RuntimeError("слишком мало дат для target-теста")

    rng = np.random.RandomState(seed)
    # даты, где T+H всё ещё в данных
    candidates = dates[300:-(HORIZON + 5)]
    check_dates = rng.choice(candidates, size=min(n_check_dates, len(candidates)),
                              replace=False)
    check_dates = sorted(check_dates)

    print(f"[guard] target lookahead: проверяем {len(check_dates)} дат")

    for T in check_dates:
        # обрезаем СТРОГО после T + HORIZON (чтобы таргет на T был определён)
        cutoff = dates[dates.get_loc(T) + HORIZON + 1]
        raw_tr = _truncate_raw(raw, cutoff)
        imoex_tr = _truncate_series(imoex, cutoff)
        target_tr = build_target(raw_tr, imoex_tr)

        if T not in target_tr.index or T not in target_full.index:
            continue
        a = target_full.loc[T].values
        b = target_tr.loc[T].values
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() == 0:
            continue
        diff = np.abs(a[m] - b[m]).max()
        if diff > 1e-8:
            raise LookaheadError(
                f"lookahead в target на дате {T.date()}: "
                f"max diff = {diff:.2e}"
            )
    print("[guard] target lookahead: OK")


def test_target_boundary(split_date: str):
    """
    Последние HORIZON дней IS должны иметь NaN в таргете
    (иначе таргет смотрит в VAL).
    """
    raw = load_raw(split_date)
    imoex = load_imoex(split_date)
    target = build_target(raw, imoex)

    if target.empty:
        raise DataLeakError("target is empty")

    last_dates = target.index[-HORIZON:]
    if not target.loc[last_dates].isna().all().all():
        bad = int(target.loc[last_dates].notna().sum().sum())
        raise DataLeakError(
            f"target boundary: {bad} non-NaN в последних {HORIZON} днях IS "
            f"(утечка в VAL)"
        )
    print("[guard] target boundary: OK")


def test_flatten_alignment(split_date: str):
    """
    flatten_panel: date_ids должны соответствовать валидным датам,
    и не должно быть перемешивания строк.
    """
    raw = load_raw(split_date)
    imoex = load_imoex(split_date)
    features = build_features(raw)
    target = build_target(raw, imoex)

    X, y, date_ids, names, valid_dates, tickers = flatten_panel(features, target)

    if len(valid_dates) == 0:
        raise DataLeakError("flatten: valid_dates пуст")

    if date_ids.max() != len(valid_dates) - 1:
        raise DataLeakError(
            f"flatten: date_ids.max()={date_ids.max()} != "
            f"len(valid_dates)-1={len(valid_dates)-1}"
        )
    if date_ids.min() != 0:
        raise DataLeakError(f"flatten: date_ids.min()={date_ids.min()} != 0")

    # порядок: date_ids должны быть неубывающими (по блокам дат)
    if not np.all(np.diff(date_ids) >= 0):
        raise DataLeakError("flatten: date_ids не неубывающие (перемешаны)")

    # на каждую дату — не менее 20 строк (как в flatten)
    counts = np.bincount(date_ids)
    if (counts < 20).any():
        bad = int((counts < 20).sum())
        raise DataLeakError(f"flatten: {bad} дат с <20 наблюдений")

    # y и X должны быть конечными (после фильтра)
    if not np.isfinite(y).all():
        raise DataLeakError("flatten: y содержит NaN/Inf")
    # X может содержать NaN (фичи), но не Inf
    if np.isinf(X).any():
        raise DataLeakError("flatten: X содержит Inf")

    print(f"[guard] flatten alignment: OK  ({len(valid_dates)} дат, "
          f"{X.shape[0]} строк, {X.shape[1]} фич)")


def test_fitness_no_target_leak(split_date: str):
    """
    Fitness не должен учитывать строки с NaN таргетом.
    Проверяем: если передать в fitness y с NaN, он не должен их учитывать.
    """
    from .fitness_cs import _cross_sectional_ic

    # синтетика: 100 дат × 50 наблюдений
    rng = np.random.RandomState(42)
    n_dates = 100
    n_per_date = 50
    date_ids = np.repeat(np.arange(n_dates), n_per_date)
    y = rng.randn(n_dates * n_per_date)
    x = y + rng.randn(n_dates * n_per_date) * 0.5  # коррелирует

    # без NaN — IC высокий
    fit1 = _cross_sectional_ic(y, x, date_ids)

    # добавляем NaN в половину строк — IC должен измениться,
    # но не должен быть -inf (мало данных только если < 30 дат)
    y_nan = y.copy()
    y_nan[::2] = np.nan
    fit2 = _cross_sectional_ic(y_nan, x, date_ids)

    if not np.isfinite(fit1):
        raise DataLeakError(f"fitness: fit1 не конечный ({fit1})")
    if not np.isfinite(fit2):
        raise DataLeakError(f"fitness: fit2 не конечный ({fit2})")

    # если fitness «видит» NaN как значения, fit2 будет сильно отличаться
    # допускаем небольшое отличие из-за уменьшения выборки
    if abs(fit1 - fit2) > 0.5:
        raise DataLeakError(
            f"fitness: подозрительно большое отличие при NaN "
            f"(fit1={fit1:.3f}, fit2={fit2:.3f}) — возможно, NaN учитываются"
        )

    print(f"[guard] fitness no target leak: OK  "
          f"(fit1={fit1:.3f}, fit2={fit2:.3f})")


def run_all(split_date: str):
    """Все тесты. При провале — raise."""
    print("\n[guard] === DATA GUARDS ===")
    test_features_lookahead(split_date)
    test_target_lookahead(split_date)
    test_target_boundary(split_date)
    test_flatten_alignment(split_date)
    test_fitness_no_target_leak(split_date)
    print("[guard] все тесты пройдены\n")


if __name__ == "__main__":
    from .data_prep import load_split_date
    run_all(load_split_date())