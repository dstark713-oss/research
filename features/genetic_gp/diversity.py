# features/genetic_gp/diversity.py
# ============================================================
# Отбор непохожих кандидатов.
#
# Берём топ-N по fitness, затем жадно добавляем те, у которых
# |corr| с уже выбранными < threshold.
# ============================================================
import numpy as np


def select_diverse(candidates: list, max_n: int = 20,
                   corr_threshold: float = 0.7) -> list:
    """
    candidates: список dict {"program": ..., "fitness": ..., "values": np.array}
    Возвращает список выбранных (не более max_n), где корреляция
    между любыми двумя < corr_threshold.
    """
    if not candidates:
        return []

    # сортируем по fitness убыванию
    sorted_c = sorted(candidates, key=lambda c: c["fitness"], reverse=True)

    selected = []
    for cand in sorted_c:
        if len(selected) >= max_n:
            break
        v = cand["values"]
        # проверяем корреляцию с уже выбранными
        ok = True
        for s in selected:
            sv = s["values"]
            m = np.isfinite(v) & np.isfinite(sv)
            if m.sum() < 100:
                continue
            c = np.corrcoef(v[m], sv[m])[0, 1]
            if np.isfinite(c) and abs(c) > corr_threshold:
                ok = False
                break
        if ok:
            selected.append(cand)

    return selected