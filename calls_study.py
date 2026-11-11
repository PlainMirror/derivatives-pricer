"""Buying SPY calls instead of selling puts, on the same model, dates and costs.

    python calls_study.py

Rules, fixed before the first run: buy a call whenever flat, sized to the same SPY exposure as the
account (delta x 100 x spot per contract = equity, the rest in cash at 3%), close it at +100% of the
premium, at -50%, or when a third of its life is left. Calls at delta 0.30, 0.50 and 0.70, for 14,
21, 30 and 45 days. The put rows are the original put-selling rules on top of a full SPY position.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import pricer
from rules_study import END, halves_sharpe, md

ROOT = os.path.dirname(os.path.abspath(__file__))
DTES = [14, 21, 30, 45]
JOBS = ([("long call", d, t, dict(side="long_call", entry="always", time_exit=1 / 3, long_target=1.0, long_stop=0.5))
         for d in (0.30, 0.50, 0.70) for t in DTES]
        + [("short put on SPY", d, t, dict(side="put")) for d in (0.15, 0.20, 0.30) for t in DTES])


def run(job):
    name, delta, dte, rules = job
    df = pricer.load_data(end=END)
    eq, tr, _, _ = pricer.backtest(df, dte=dte, target_delta=delta, **rules)
    m = pricer.metrics(eq, df)
    h1, h2, full = halves_sharpe(eq)
    rc = tr["reason"].value_counts(normalize=True) if len(tr) else pd.Series(dtype=float)
    return {"strategy": name, "delta": delta, "dte": dte, "total_return_%": float(m["total_return_%"]),
            "cagr_%": float(m["cagr_%"]), "sharpe": full, "sharpe_2010_17": h1, "sharpe_2018_26": h2,
            "sortino": float(m["sortino"]), "max_dd_%": float(m["max_drawdown_%"]), "alpha_%": float(m["ann_alpha_%"]),
            "beta": float(m["beta_vs_spy"]), "trades": len(tr),
            "win_rate": float((tr["net_$"] > 0).mean()) if len(tr) else np.nan,
            "pct_target": float(rc.get("profit_target", 0)), "pct_stop": float(rc.get("stop", 0)),
            "pct_time": float(rc.get("time_exit", 0))}


def spy_row(df):
    px = pd.Series((df["spy_close"] * (1 + df["spy_div"] / df["spy_close"]).cumprod()).to_numpy(), index=pd.to_datetime(df["Date"]))
    eq = px / px.iloc[0] * 500_000
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    h1, h2, full = halves_sharpe(eq)
    return {"strategy": "SPY buy and hold", "total_return_%": 100 * (eq.iloc[-1] / eq.iloc[0] - 1),
            "cagr_%": 100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1), "sharpe": full, "sharpe_2010_17": h1,
            "sharpe_2018_26": h2, "max_dd_%": 100 * float((eq / eq.cummax() - 1).min())}


def main():
    df = pricer.load_data(end=END)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as pool:
        res = pd.DataFrame(list(pool.map(run, JOBS)))
    spy = spy_row(df)
    res.to_csv(os.path.join(ROOT, "results", "calls_study.csv"), index=False)
    lines = [f"# Buying calls against selling puts\n\n{df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}.\n"]
    for name in ("long call", "short put on SPY"):
        g = res[res["strategy"] == name]
        for col in ("sharpe", "cagr_%", "max_dd_%"):
            lines += [f"### {name}: {col}\n", md(g.pivot(index="delta", columns="dte", values=col).round(2)), ""]
        print(f"== {name}")
        print(g[["delta", "dte", "total_return_%", "cagr_%", "sharpe", "sharpe_2010_17", "sharpe_2018_26",
                 "max_dd_%", "alpha_%", "beta", "trades", "win_rate", "pct_target", "pct_stop", "pct_time"]].round(2).to_string(index=False))
    print("== SPY", {k: round(v, 2) for k, v in spy.items() if k != "strategy"})
    lines.append("SPY buy and hold: " + ", ".join(f"{k} {v:.2f}" for k, v in spy.items() if k != "strategy"))
    with open(os.path.join(ROOT, "results", "calls_study.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
