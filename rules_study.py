"""Which entry and exit rules to keep, decided by a rule fixed before any of them ran.

    python rules_study.py

Every variant changes ONE thing from the base rules and runs on all 12 settings (put delta 0.15,
0.20, 0.30 x 14, 21, 30, 45 days), over 2010 to today, with the Leisen-Reimer tree and the fitted
smile. Sharpe is then read separately for the two halves, 2010-2017 and 2018 onwards.

    THE RULE: a change is kept only if the median Sharpe across the 12 settings is higher than the
    base rules' median in BOTH halves.

The kept changes are then combined into one final rule set, which is reported for both halves as
well, since changes that each help alone can still clash. Finally the base and final rules are run
with cash-secured puts and compared with the CBOE S&P 500 PutWrite index (PUT).

Variants (written down before the first run):
    base  sell when VIX <= 19, close at 70% of the premium kept or a loss of 2x the premium, else expiry
    E1    no VIX cap: sell whenever flat
    E2    no VIX cap, but sell only when model at-the-money vol is above 20-day realised vol
    E3    no VIX cap, but skip days when VIX9D is above VIX (backwardation)
    X1    take profit at 50% of the premium instead of 70%
    X2    close when a third of the option's life is left
    X3    stop on the option's delta reaching 0.50 instead of a 2x premium loss
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import pricer

ROOT = os.path.dirname(os.path.abspath(__file__))
DELTAS, DTES = [0.15, 0.20, 0.30], [14, 21, 30, 45]
SPLIT = "2018-01-01"
END = "2026-11-11"          # exclusive: the last full session is Nov 10


def md(frame: pd.DataFrame) -> str:
    """A dataframe as a markdown table, without the tabulate dependency."""
    f = frame.reset_index()
    head = "| " + " | ".join(str(c) for c in f.columns) + " |\n|" + "---|" * len(f.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in f.itertuples(index=False)]
    return "\n".join([head] + body)
BASE = dict(entry="vix_cap", profit_target=0.70, time_exit=None, stop="loss", collateral="overlay")
VARIANTS = {
    "base": {},
    "E1 no VIX cap": {"entry": "always"},
    "E2 implied above realised": {"entry": "vrp"},
    "E3 skip backwardation": {"entry": "term"},
    "X1 50% profit target": {"profit_target": 0.50},
    "X2 time exit at 1/3 left": {"time_exit": 1 / 3},
    "X3 delta stop at 0.50": {"stop": "delta"},
}
ENTRY_OF = {"E1 no VIX cap": "always", "E2 implied above realised": "vrp", "E3 skip backwardation": "term"}


def halves_sharpe(eq: pd.Series) -> tuple[float, float, float]:
    return (pricer.sharpe_of(eq[eq.index < SPLIT]), pricer.sharpe_of(eq[eq.index >= SPLIT]), pricer.sharpe_of(eq))


def run_cell(job):
    name, rules, delta, dte = job
    df = pricer.load_data(end=END)
    eq, trades, _, _ = pricer.backtest(df, dte=dte, target_delta=delta, **rules)
    h1, h2, full = halves_sharpe(eq)
    m = pricer.metrics(eq, df)
    rc = trades["reason"].value_counts(normalize=True) if len(trades) else pd.Series(dtype=float)
    return {"variant": name, "delta": delta, "dte": dte, "sharpe_2010_17": h1, "sharpe_2018_26": h2,
            "sharpe_full": full, "max_dd": m["max_drawdown_%"], "alpha": m["ann_alpha_%"], "beta": m["beta_vs_spy"],
            "total_return": m["total_return_%"], "trades": len(trades),
            "win_rate": float((trades["net_$"] > 0).mean()) if len(trades) else np.nan,
            "pct_stop": float(rc.get("stop", 0)), "pct_profit": float(rc.get("profit_target", 0)),
            "pct_time": float(rc.get("time_exit", 0)), "pct_expiry": float(rc.get("expiry", 0))}


def run_variants(variants: dict, pool) -> pd.DataFrame:
    jobs = [(name, dict(BASE, **ch), d, t) for name, ch in variants.items() for d in DELTAS for t in DTES]
    return pd.DataFrame(list(pool.map(run_cell, jobs)))


def summary(res: pd.DataFrame) -> pd.DataFrame:
    return res.groupby("variant", sort=False).agg(
        median_2010_17=("sharpe_2010_17", "median"), median_2018_26=("sharpe_2018_26", "median"),
        median_full=("sharpe_full", "median"), worst_dd=("max_dd", "min"), median_trades=("trades", "median"),
        median_win=("win_rate", "median"))


def index_sharpes(start, end):
    """SPY total return and the CBOE PutWrite index over the same halves."""
    import yfinance as yf
    out = {}
    for tk in ("SPY", "^PUT"):
        try:
            px = yf.download(tk, start=start, end=end, auto_adjust=True, progress=False)["Close"]
            px = px.iloc[:, 0] if isinstance(px, pd.DataFrame) else px
            px = px.dropna()
            if len(px) > 500:
                out[tk] = halves_sharpe(px) + (float(px.iloc[-1] / px.iloc[0] - 1) * 100, float((px / px.cummax() - 1).min() * 100))
        except Exception as e:  # noqa: BLE001
            out[tk] = f"unavailable ({type(e).__name__})"
    return out


def main():
    df = pricer.load_data(end=END)                      # downloads once and caches for the workers
    lines = [f"# Rules study\n\n{df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}, split at {SPLIT}. "
             f"Model: {pricer.MODEL}.\n"]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as pool:
        res = run_variants(VARIANTS, pool)
        s = summary(res)
        b = s.loc["base"]
        s["kept"] = (s["median_2010_17"] > b["median_2010_17"]) & (s["median_2018_26"] > b["median_2018_26"])
        s.loc["base", "kept"] = False
        kept = [v for v in s.index if s.loc[v, "kept"]]
        print(s.round(3).to_string(), flush=True)
        lines += ["## One change at a time\n", md(s.round(3)), "",
                  f"Kept (median Sharpe up in both halves): {', '.join(kept) if kept else 'none'}\n"]

        final = {}
        entries = [ENTRY_OF[v] for v in kept if v in ENTRY_OF]
        if entries:
            filt = [e for e in entries if e != "always"]
            final["entry"] = "+".join(filt) if filt else "always"
        for v in kept:
            if v not in ENTRY_OF:
                final.update(VARIANTS[v])
        combos = {"base": {}, "final": final,
                  "base, cash-secured": {"collateral": "cash"}, "final, cash-secured": dict(final, collateral="cash")}
        res2 = run_variants(combos, pool)
    s2 = summary(res2)
    print(f"\nfinal rules: {dict(BASE, **final)}")
    print(s2.round(3).to_string())
    lines += [f"## Combined\n\nFinal rules: `{dict(BASE, **final)}`\n", md(s2.round(3)), ""]

    for name in ("final", "final, cash-secured"):
        grid = res2[res2["variant"] == name].pivot(index="delta", columns="dte", values="sharpe_full")
        lines += [f"### Sharpe by setting, {name}, full period\n", md(grid.round(2)), ""]
        g2 = res2[res2["variant"] == name].pivot(index="delta", columns="dte", values="max_dd")
        lines += [f"### Max drawdown %, {name}\n", md(g2.round(1)), ""]

    idx = index_sharpes(str(df["Date"].iloc[0].date()), END)
    lines.append("## Benchmarks over the same dates\n")
    lines.append("| | Sharpe 2010-17 | Sharpe 2018-26 | Sharpe full | total return % | max drawdown % |\n|---|---|---|---|---|---|")
    for k, v in idx.items():
        lines.append(f"| {k} | " + (" | ".join(f"{x:.2f}" for x in v) if isinstance(v, tuple) else v) + " |")
    print("\n".join(lines[-3:]))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    pd.concat([res, res2]).to_csv(os.path.join(ROOT, "results", "rules_study.csv"), index=False)
    with open(os.path.join(ROOT, "results", "rules_study.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote results/rules_study.md and results/rules_study.csv")


if __name__ == "__main__":
    main()
