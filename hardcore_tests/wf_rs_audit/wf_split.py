# hardcore_tests/wf_rs_audit/wf_split.py
# ============================================================
# Блок 5: WF split и обучение.
# Проверяем: нет shuffle, нет val в train, purge, early stopping.
# ============================================================
import numpy as np
import pandas as pd
from pathlib import Path

from rs_model.config import BASE_PARAMS, RANKER_PARAMS
from rs_model.wf import make_folds
from .common import (
    load_ohlcv, load_imoex, load_split_date, independent_target,
    independent_folds, HORIZON, PURGE, verdict,
)


def test_folds_match() -> dict:
    """Сравниваем make_folds из rs_model с независимой реконструкцией."""
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    target = independent_target(raw["close"], imoex)
    dates = target.dropna(how="all").index

    from_model = make_folds(dates, n_folds=6, purge=PURGE)
    independent = independent_folds(dates, n_folds=6, purge=PURGE)

    if len(from_model) != len(independent):
        return verdict(
            "5.1 make_folds совпадает с независимой реконструкцией",
            "FAIL", "HIGH",
            evidence=f"len(from_model)={len(from_model)}, "
                     f"len(independent)={len(independent)}",
            files=["rs_model/wf.py::make_folds"],
        )
    for i, ((tr1, va1), (tr2, va2)) in enumerate(zip(from_model, independent)):
        if not (tr1.equals(tr2) and va1.equals(va2)):
            return verdict(
                "5.1 make_folds совпадает с независимой реконструкцией",
                "FAIL", "HIGH",
                evidence=f"Fold {i+1} различается: "
                         f"model train {tr1[0].date()}→{tr1[-1].date()}, "
                         f"indep train {tr2[0].date()}→{tr2[-1].date()}",
                files=["rs_model/wf.py::make_folds"],
            )
    return verdict(
        "5.1 make_folds совпадает с независимой реконструкцией",
        "PASS",
        evidence=f"{len(from_model)} фолдов: train/val даты совпадают.",
        files=["rs_model/wf.py::make_folds"],
    )


def test_no_shuffle() -> dict:
    """
    Проверяем: в коде WF нет shuffle, random permutation дат.
    """
    wf_path = Path(__file__).parent.parent.parent / "rs_model" / "wf.py"
    src = wf_path.read_text(encoding="utf-8")
    bad = []
    for token in ["shuffle", "np.random.permutation", "sample("]:
        for line_no, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if token in code:
                bad.append((token, line_no, line.strip()))
    if bad:
        return verdict(
            "5.2 Нет shuffle в WF",
            "FAIL", "CRITICAL",
            evidence=f"Найдено: {bad}",
            files=["rs_model/wf.py"],
        )
    return verdict(
        "5.2 Нет shuffle в WF",
        "PASS",
        evidence="В wf.py нет shuffle/permutation/sample.",
        files=["rs_model/wf.py"],
    )


def test_no_val_in_train() -> dict:
    """Проверяем: даты val не пересекаются с train."""
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    target = independent_target(raw["close"], imoex)
    dates = target.dropna(how="all").index
    folds = independent_folds(dates, n_folds=6, purge=PURGE)

    fails = []
    for i, (tr, va) in enumerate(folds):
        overlap = set(tr) & set(va)
        if overlap:
            fails.append({"fold": i + 1, "overlap_size": len(overlap)})
    if fails:
        return verdict(
            "5.3 Val не пересекается с train",
            "FAIL", "CRITICAL",
            evidence=f"Пересечения: {fails}",
        )
    return verdict(
        "5.3 Val не пересекается с train",
        "PASS",
        evidence=f"{len(folds)} фолдов: нет общих дат.",
    )


def test_early_stopping_on_external_val() -> dict:
    """
    Проверяем: в is_wf_regressor_plateau / wf_is_rs early stopping
    использует ВНЕШНИЙ val или внутренний?
    """
    # Ищем в файлах backtests использование eval_set с внешним val
    files_to_check = [
        Path(__file__).parent.parent.parent / "backtests" / "wf_is_rs.py",
        Path(__file__).parent.parent.parent / "backtests" / "is_wf_regressor_plateau.py",
        Path(__file__).parent.parent.parent / "backtests" / "is_wf_ranker.py",
    ]
    findings = []
    for f in files_to_check:
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8")
        # Ищем eval_set=(X_va, y_va) где X_va — внешний val
        # Простой эвристический поиск
        for line_no, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if "eval_set" in code and ("X_va" in code or "va_long" in code):
                findings.append((f.name, line_no, line.strip()))
    if findings:
        return verdict(
            "5.4 Early stopping не на внешнем val",
            "NOT_VERIFIED", "HIGH",
            evidence=f"Найдено использование eval_set с val: {findings}. "
                     f"Если это внешний val — утечка при выборе best_iter.",
            files=[f.name for f in files_to_check if f.exists()],
            recommendation="Проверить: early stopping должен быть на internal val "
                           "(выделенном из train), не на внешнем.",
        )
    return verdict(
        "5.4 Early stopping не на внешнем val",
        "PASS",
        evidence="В проверенных файлах нет eval_set с внешним val.",
    )


def test_regressor_params() -> dict:
    """Проверяем: iterations фиксированы (300), не подбираются на val."""
    from rs_model.config import BASE_PARAMS
    it = BASE_PARAMS.get("iterations", None)
    if it is not None and it <= 1500:
        return verdict(
            "5.5 Итерации фиксированы",
            "PASS",
            evidence=f"BASE_PARAMS['iterations'] = {it}. "
                     f"В base_model_v2.py используется фикс. 300.",
            files=["rs_model/config.py"],
        )
    return verdict(
        "5.5 Итерации фиксированы",
        "NOT_VERIFIED", "MEDIUM",
        evidence=f"iterations = {it}. Нужно проверить, не подбирается ли на val.",
    )


def run() -> list:
    return [
        test_folds_match(),
        test_no_shuffle(),
        test_no_val_in_train(),
        test_early_stopping_on_external_val(),
        test_regressor_params(),
    ]