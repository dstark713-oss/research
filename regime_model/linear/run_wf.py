# regime_model/linear/run_wf.py
# ============================================================
# Точка запуска WF линейной модели.
# Fallback на IMOEX до появления достаточной истории портфеля.
# ============================================================
from regime_model.garch.data_loader import (
    load_split_date, load_stock_close, load_ml_holdings, load_ml_equity,
    load_imoex,
)
from regime_model.garch.portfolio_returns import build_daily_returns
from .walk_forward import run_linear_wf


def main():
    print("=" * 70)
    print("LINEAR RISK SIZING — WALK-FORWARD")
    print("=" * 70)

    split_date = load_split_date()
    print(f"split_date = {split_date}\n")

    close = load_stock_close(split_date)
    holdings = load_ml_holdings()
    ml_eq = load_ml_equity()

    rebal_dates_all = sorted(ml_eq["date"].tolist())
    rebal_dates = rebal_dates_all[1:]
    print(f"rebal_dates: {len(rebal_dates)} (из {len(rebal_dates_all)})")

    port_ret = build_daily_returns(close, holdings)
    print(f"port_ret (raw): {len(port_ret)} дней, "
          f"{port_ret.index.min().date()} → {port_ret.index.max().date()}")

    # добавить rd с 0 (MOC)
    n_added = 0
    for rd in rebal_dates:
        if rd not in port_ret.index:
            port_ret.loc[rd] = 0.0
            n_added += 1
    port_ret = port_ret.sort_index()
    print(f"port_ret (с rd): {len(port_ret)} дней (+{n_added})\n")

    # IMOEX returns (fallback для ранних дат)
    imoex_close = load_imoex(split_date)
    imoex_ret = imoex_close.pct_change().dropna()
    print(f"imoex_ret: {len(imoex_ret)} дней, "
          f"{imoex_ret.index.min().date()} → {imoex_ret.index.max().date()}\n")

    result = run_linear_wf(
        port_ret, rebal_dates,
        imoex_ret=imoex_ret,
        verbose=True,
    )

    exp = result["exposure"]
    if len(exp) > 0:
        print(f"\nExposure: n={len(exp)}, mean={exp.mean():.3f}, "
              f"min={exp.min():.3f}, max={exp.max():.3f}")
    else:
        print(f"\nExposure: пусто!")
    print(f"Errors: {len(result['errors'])}")
    for e in result["errors"][:10]:
        print(f"  {e['date'].date()}: {e['reason']}")


if __name__ == "__main__":
    main()