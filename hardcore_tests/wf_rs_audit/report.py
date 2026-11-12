# hardcore_tests/wf_rs_audit/report.py
# ============================================================
# Блок 11: итоговая классификация + Markdown-отчёт.
# ============================================================
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
OUT_REPORT = OUT_DIR / "wf_rs_audit_report.md"


def overall_verdict(all_results: list) -> str:
    fails = [r for r in all_results if r["status"] == "FAIL"]
    critical = [r for r in fails if r["severity"] == "CRITICAL"]
    high = [r for r in fails if r["severity"] == "HIGH"]
    not_verified = [r for r in all_results if r["status"] == "NOT_VERIFIED"]

    if critical:
        return "WF INVALID"
    if high or len(not_verified) >= 3:
        return "WF CONDITIONALLY VALID"
    return "WF VERIFIED"


def render_report(all_results: list, temporal_lines: list,
                  split_date: str) -> str:
    lines = [
        "# Forensic Audit of Walk-Forward Backtest\n",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}\n",
        f"split_date = {split_date} (VAL не трогаем)\n",
        "\n## Executive Summary\n",
    ]

    n_pass = sum(1 for r in all_results if r["status"] == "PASS")
    n_fail = sum(1 for r in all_results if r["status"] == "FAIL")
    n_nv = sum(1 for r in all_results if r["status"] == "NOT_VERIFIED")
    overall = overall_verdict(all_results)

    lines.append(f"- Всего проверок: {len(all_results)}")
    lines.append(f"- PASS: {n_pass}")
    lines.append(f"- FAIL: {n_fail}")
    lines.append(f"- NOT_VERIFIED: {n_nv}")
    lines.append(f"\n**Вердикт: {overall}**\n")

    # --- таблица проверок ---
    lines.append("\n## Таблица всех проверок\n")
    lines.append("| # | Проверка | Статус | Severity | Доказательство |")
    lines.append("|---|----------|--------|----------|----------------|")
    for i, r in enumerate(all_results, 1):
        ev = r["evidence"][:200].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {i} | {r['name']} | {r['status']} | {r['severity']} | {ev} |")

    # --- CRITICAL/HIGH ---
    lines.append("\n## CRITICAL и HIGH проблемы\n")
    crit_high = [r for r in all_results
                 if r["status"] != "PASS" and r["severity"] in ("CRITICAL", "HIGH")]
    if not crit_high:
        lines.append("Нет CRITICAL/HIGH проблем.\n")
    else:
        for r in crit_high:
            lines.append(f"\n### {r['name']} — {r['status']} ({r['severity']})")
            lines.append(f"- **Evidence:** {r['evidence']}")
            if r["files"]:
                lines.append(f"- **Files:** {', '.join(r['files'])}")
            if r["recommendation"]:
                lines.append(f"- **Recommendation:** {r['recommendation']}")

    # --- трассировка ---
    lines.append("\n## Трассировка временной информации\n")
    lines.extend(temporal_lines)

    # --- итог ---
    lines.append("\n## Итоговый вердикт\n")
    lines.append(f"**{overall}**\n")
    if overall == "WF INVALID":
        lines.append("Есть CRITICAL проблемы — результатам нельзя доверять. "
                     "Требуется исправление до любых выводов.\n")
    elif overall == "WF CONDITIONALLY VALID":
        lines.append("Критических проблем нет, но есть HIGH/NOT_VERIFIED. "
                     "Результаты можно использовать с оговорками, "
                     "указанными в отчёте.\n")
    else:
        lines.append("Критических leakage/look-ahead не обнаружено. "
                     "Временная и портфельная механика подтверждена.\n")

    lines.append("\n## Ограничения аудита\n")
    lines.append("- Не проверялась робастность стратегии (это отдельный этап).")
    lines.append("- Не проверялись рыночные режимы (это отдельный этап).")
    lines.append("- Survivorship bias: если universe сформирован по текущему "
                 "состоянию, исторические делистингованные отсутствуют.")
    lines.append("- Execution model (MOC vs open(t+1)) — требует явной фиксации.")
    lines.append("- Отсутствие обнаруженной ошибки не доказывает её отсутствия.\n")

    return "\n".join(lines)


def save_report(text: str):
    OUT_REPORT.write_text(text, encoding="utf-8")
    print(f"\n[audit] отчёт: {OUT_REPORT}")