"""How far is the back test's pricing model from real option prices?

    python validate_surface.py cache/SPY_2026-09-24.json 2026-09-24

The back test never sees an option quote. It builds every price from two index
levels (VIX9D and VIX) and an assumed linear skew. This takes one real SPY
option chain and, for the puts the back test would sell (about 0.15 to 0.30
delta, 14 to 45 days), compares

    model price   the pricer's own vol model through its own tree, exactly as the back test does
    real price    the quoted mid

and breaks the gap into its two sources: the at-the-money level and the skew.
The chain file is an Alpaca snapshot ({occ_symbol: {"latestQuote": {"bp", "ap"}}})
and is not included here.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from pricer import atm_vol, bin_tree, iv, skew_slope

R, Q, N_BACKTEST, N_FINE = 0.03, 0.012, 50, 200      # the back test's own rate, dividend yield and tree size
OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_puts(path: str, asof: dt.date) -> pd.DataFrame:
    rows = []
    for occ, snap in json.load(open(path)).items():
        m = OCC.match(occ)
        q = (snap or {}).get("latestQuote") or {}
        if not m or m.group(3) != "P" or not q.get("bp") or not q.get("ap") or q["ap"] <= q["bp"]:
            continue
        exp = dt.datetime.strptime(m.group(2), "%y%m%d").date()
        rows.append({"expiry": exp, "dte": (exp - asof).days, "K": int(m.group(4)) / 1000.0,
                     "bid": float(q["bp"]), "ask": float(q["ap"])})
    d = pd.DataFrame(rows)
    d["mid"] = (d["bid"] + d["ask"]) / 2
    return d[(d["dte"] >= 7) & (d["dte"] <= 52) & ((d["ask"] - d["bid"]) <= 0.25 * d["mid"])]


def real_iv(price, S, K, T):
    f = lambda s: bin_tree(S, K, T, R, s, Q, N_FINE) - price     # noqa: E731
    try:
        return brentq(f, 0.01, 2.0, xtol=1e-6)
    except ValueError:
        return float("nan")


def abs_delta(S, K, T, sig):
    d1 = (math.log(S / K) + (R - Q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return math.exp(-Q * T) * ndtr(-d1)


def main(path: str, day: str, spot: float | None = None) -> pd.DataFrame:
    asof = dt.date.fromisoformat(day)
    import yfinance as yf
    px = yf.download(["SPY", "^VIX", "^VIX9D"], start=day, end=str(asof + dt.timedelta(days=4)),
                     auto_adjust=False, progress=False)["Close"].iloc[0]
    S = float(spot or px["SPY"])
    vix, vix9d = float(px["^VIX"]), float(px["^VIX9D"])
    d = load_puts(path, asof)
    d = d[d["K"] < S].copy()
    d["T"] = d["dte"] / 365.0
    d["atm_model"] = [atm_vol(vix9d, vix, T) for T in d["T"]]
    d["iv_model"] = [iv(S, K, T, a, vix) for K, T, a in zip(d["K"], d["T"], d["atm_model"])]
    d["px_model"] = [bin_tree(S, K, T, R, s, Q, N_BACKTEST) for K, T, s in zip(d["K"], d["T"], d["iv_model"])]
    d["iv_real"] = [real_iv(p, S, K, T) for p, K, T in zip(d["mid"], d["K"], d["T"])]
    d = d.dropna(subset=["iv_real"])
    d["delta"] = [abs_delta(S, K, T, s) for K, T, s in zip(d["K"], d["T"], d["iv_real"])]
    d["px_err_pct"] = 100 * (d["px_model"] / d["mid"] - 1)
    d["iv_err_pts"] = 100 * (d["iv_model"] - d["iv_real"])

    print(f"SPY {day}: spot {S:.2f}, VIX9D {vix9d:.2f}, VIX {vix:.2f}; {len(d)} put quotes, 7 to 52 days")
    # ---- level: real at-the-money vol per expiry (nearest strike below spot) against the model's
    atm = d.loc[d.groupby("expiry")["K"].idxmax(), ["expiry", "dte", "iv_real", "atm_model"]].sort_values("dte")
    atm["err_pts"] = 100 * (atm["atm_model"] - atm["iv_real"])
    print(f"\n1. at-the-money level: model minus real, vol points, by expiry "
          f"(median {atm['err_pts'].median():+.2f}, range {atm['err_pts'].min():+.2f} to {atm['err_pts'].max():+.2f})")
    print(atm.assign(iv_real=100 * atm["iv_real"], atm_model=100 * atm["atm_model"]).round(2).to_string(index=False))
    # ---- skew: slope of vol against log-moneyness, puts between 0.05 and 0.50 delta
    slopes = []
    for exp, g in d[(d["delta"] > 0.05) & (d["delta"] < 0.50)].groupby("expiry"):
        if len(g) >= 8:
            slopes.append(-np.polyfit(np.log(g["K"] / S), g["iv_real"], 1)[0])
    print(f"   real / model at-the-money ratio: median {(atm['iv_real'] / atm['atm_model']).median():.2f}  (--atm-scale)")
    print(f"\n2. skew: real slope of vol against log-strike, median {np.median(slopes):.2f} "
          f"(range {min(slopes):.2f} to {max(slopes):.2f}); the model assumes {skew_slope(vix):.2f}")
    print(f"   the same slope rescaled to VIX 20: {np.median(slopes) * 20 / vix:.2f}  (--skew-base)")
    # ---- what it does to the prices the back test collects
    sel = d[(d["delta"] >= 0.10) & (d["delta"] <= 0.35)].copy()
    sel["delta_bucket"] = pd.cut(sel["delta"], [0.10, 0.175, 0.25, 0.35], labels=["0.15", "0.20", "0.30"])
    sel["dte_bucket"] = pd.cut(sel["dte"], [6, 17, 25, 37, 52], labels=["14d", "21d", "30d", "45d"])
    t = sel.groupby(["dte_bucket", "delta_bucket"], observed=True).agg(
        n=("K", "size"), real_mid=("mid", "median"), price_err_pct=("px_err_pct", "median"),
        iv_err_pts=("iv_err_pts", "median"))
    print("\n3. the puts the back test sells: model price against the real mid (negative = model too cheap)")
    print(t.round(2).to_string())
    print(f"\n   all {len(sel)} of them: median price error {sel['px_err_pct'].median():+.1f}%, "
          f"median vol error {sel['iv_err_pts'].median():+.2f} points")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    t.round(3).to_csv(os.path.join(ROOT, "results", f"surface_validation_{day}.csv"))
    return sel


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
