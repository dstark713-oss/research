# features/llm_pool/run.py
# ============================================================
# Оркестратор: запускает батчи по 50 фич, пока не наберём 10-20.
# Останавливается, как только набрали >= 10 ортогональных.
# ============================================================
from datetime import datetime
from pathlib import Path

from .common import load_split_date, load_ohlcv, load_imoex
from .batch_001_050 import build_batch_001_050
from .batch_051_100 import build_batch_051_100
from .evaluate import run_evaluate, OUT_MD


def main():
    split_date = load_split_date()
    raw = load_ohlcv(split_date)
    imoex = load_imoex(split_date)
    print(f"[run] split_date = {split_date}, data loaded")

    rep_lines = [f"# LLM Feature Pool Evaluation\n\nДата: {datetime.now():%Y-%m-%d %H:%M}\n"]
    all_selected = []

    # --- Batch 001-050 ---
    print("\n[run] === Batch 001-050 ===")
    features_1 = build_batch_001_050(raw)
    print(f"[run] построено {len(features_1)} фич")
    passed_1, selected_1 = run_evaluate("001_050", features_1, rep_lines)
    all_selected.extend(selected_1)
    print(f"[run] passed IC/IR: {len(passed_1)}, selected: {len(selected_1)}")

    # --- Batch 051-100 (если не набрали 10) ---
    if len(all_selected) < 10:
        print("\n[run] === Batch 051-100 ===")
        features_2 = build_batch_051_100(raw, imoex)
        print(f"[run] построено {len(features_2)} фич")
        passed_2, selected_2 = run_evaluate("051_100", features_2, rep_lines)
        # дедупликация по имени (на случай пересечений)
        new_sel = [s for s in selected_2 if s not in all_selected]
        all_selected.extend(new_sel)
        print(f"[run] passed IC/IR: {len(passed_2)}, selected: {len(selected_2)} (new: {len(new_sel)})")

    # --- Стоп ---
    if len(all_selected) >= 10:
        print(f"\n[run] STOP: набрали {len(all_selected)} ортогональных фич (>= 10)")
    else:
        print(f"\n[run] набрали {len(all_selected)}, нужно ещё (батч 101-150 — следующий шаг)")

    rep_lines.append(f"\n## ИТОГ\n\nОртогональных фич: **{len(all_selected)}**")
    rep_lines.append("\nСписок:")
    for s in all_selected:
        rep_lines.append(f"  - {s}")

    OUT_MD.write_text("\n".join(rep_lines), encoding="utf-8")
    print(f"\n[run] отчёт: {OUT_MD}")


if __name__ == "__main__":
    main()