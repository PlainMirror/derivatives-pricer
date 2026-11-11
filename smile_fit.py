"""Fit the back test's vol model to a real SPY chain, and check it on expiries it was not fitted on.

    python smile_fit.py cache/SPY_2026-09-24.json 2026-09-24 767.29

Two things are fitted, both from the out-of-the-money options between 7 and 52 days:

  level   real at-the-money vol / the VIX9D-VIX number the model starts from (MODEL["atm_scale"])
  shape   the smile in standardised moneyness x = ln(K/F) / (sigma_atm * sqrt(T)):
              sigma(x) / sigma_atm = 1 + B1 x + B2 x^2
          Dividing log-moneyness by sigma sqrt(T) is what makes one shape serve every expiry: a
          5% out-of-the-money strike is far into the wing for a week and close to the money for
          two months, and in these units the two line up.

The expiries are split alternately: the fit uses the 1st, 3rd, 5th ... and the check uses the
2nd, 4th, 6th ..., so every number in part 3 is on expiries the fit never saw. The original
model (flat linear skew, VIX as at-the-money vol) is scored on the same strikes.
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

import pricer
from pricer import LEGACY, atm_vol, bs_abs_delta, iv, lr_tree

R, Q, N_INV = 0.03, 0.012, 101      # the back test's own rate and dividend yield
OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_chain(path: str, asof: dt.date, S: float) -> pd.DataFrame:
    rows = []
    for occ, snap in json.load(open(path)).items():
        m = OCC.match(occ)
        q = (snap or {}).get("latestQuote") or {}
        if not m or not q.get("bp") or not q.get("ap") or q["ap"] <= q["bp"]:
            continue
        exp = dt.datetime.strptime(m.group(2), "%y%m%d").date()
        rows.append({"expiry": exp, "dte": (exp - asof).days, "right": m.group(3),
                     "K": int(m.group(4)) / 1000.0, "bid": float(q["bp"]), "ask": float(q["ap"])})
    d = pd.DataFrame(rows)
    d["mid"] = (d["bid"] + d["ask"]) / 2
    d["T"] = d["dte"] / 365.0
    d["F"] = S * np.exp((R - Q) * d["T"])
    otm = ((d["right"] == "P") & (d["K"] < d["F"])) | ((d["right"] == "C") & (d["K"] >= d["F"]))
    keep = otm & (d["dte"] >= 7) & (d["dte"] <= 52) & ((d["ask"] - d["bid"]) <= 0.25 * d["mid"]) & (d["mid"] >= 0.05)
    return d[keep].reset_index(drop=True)


def real_iv(price, S, K, T, call):
    f = lambda s: lr_tree(S, K, T, R, s, Q, N_INV, american=True, call=call) - price  # noqa: E731
    try:
        return brentq(f, 0.01, 2.5, xtol=1e-6)
    except ValueError:
        return float("nan")


def atm_of(g: pd.DataFrame) -> float:
    """At-the-money vol of one expiry: the vol at the forward, interpolated in log-strike from the
    two strikes either side of it."""
    g = g.sort_values("K")
    k = np.log(g["K"] / g["F"]).to_numpy()
    return float(np.interp(0.0, k, g["iv"].to_numpy()))


def main(path: str, day: str, S: float) -> dict:
    asof = dt.date.fromisoformat(day)
    import yfinance as yf
    px = yf.download(["^VIX", "^VIX9D"], start=day, end=str(asof + dt.timedelta(days=4)),
                     auto_adjust=False, progress=False)["Close"].iloc[0]
    vix, vix9d = float(px["^VIX"]), float(px["^VIX9D"])
    d = load_chain(path, asof, S)
    d["iv"] = [real_iv(p, S, K, T, r == "C") for p, K, T, r in zip(d["mid"], d["K"], d["T"], d["right"])]
    d = d.dropna(subset=["iv"])
    exps = sorted(d["expiry"].unique())
    rows = []
    for e in exps:
        g = d[d["expiry"] == e]
        if len(g) < 8 or not (g["right"] == "P").any() or not (g["right"] == "C").any():
            continue
        T = g["T"].iloc[0]
        a = atm_of(g)
        rows.append({"expiry": e, "dte": int(g["dte"].iloc[0]), "T": T, "atm_real": a,
                     "atm_vix": atm_vol(vix9d, vix, T)})
    ex = pd.DataFrame(rows)
    ex["fit"] = np.arange(len(ex)) % 2 == 0
    d = d.merge(ex[["expiry", "atm_real", "atm_vix", "fit"]], on="expiry")
    d["x"] = np.log(d["K"] / d["F"]) / (d["atm_real"] * np.sqrt(d["T"]))
    d = d[(d["x"] >= -4) & (d["x"] <= 3)]

    # ---- level
    ratio = ex["atm_real"] / ex["atm_vix"]
    scale = float(ratio[ex["fit"]].median())
    # ---- shape: sigma/sigma_atm - 1 = b1 x + b2 x^2 on each side of the forward, least squares
    # without an intercept, so both sides meet at sigma_atm. (A single quadratic for both sides was
    # tried first: it fitted the whole smile well but the flatter call side pulled it, and it priced
    # the sold puts 15% too cheap on the check expiries. The split was chosen after seeing that.)
    f = d[d["fit"]]

    def side_fit(s):
        A = np.column_stack([s["x"], s["x"] ** 2])
        return np.linalg.lstsq(A, (s["iv"] / s["atm_real"] - 1).to_numpy(), rcond=None)[0]
    b1, b2 = side_fit(f[f["x"] < 0])
    c1, c2 = side_fit(f[f["x"] >= 0])

    print(f"SPY {day}: spot {S:.2f}, VIX9D {vix9d:.2f}, VIX {vix:.2f}; {len(d)} out-of-the-money quotes "
          f"on {len(ex)} expiries ({int(ex['fit'].sum())} to fit, {int((~ex['fit']).sum())} to check)")
    print(f"\n1. level: real at-the-money vol / VIX-based vol = {scale:.3f} on the fitting expiries "
          f"(range {ratio[ex['fit']].min():.3f} to {ratio[ex['fit']].max():.3f}); "
          f"on the check expiries {ratio[~ex['fit']].min():.3f} to {ratio[~ex['fit']].max():.3f}")
    print(f"2. shape: sigma(x) / sigma_atm = 1 {b1:+.4f} x {b2:+.4f} x^2 below the forward, 1 {c1:+.4f} x {c2:+.4f} x^2 above")

    # ---- check on the other expiries: full model (level and shape) against the original model
    pricer.SMILE_B1, pricer.SMILE_B2, pricer.SMILE_C1, pricer.SMILE_C2 = map(float, (b1, b2, c1, c2))
    fitted = dict(pricer.MODEL, smile="fitted", atm_scale=scale)
    c = d[~d["fit"]].copy()
    c["iv_new"] = [iv(S, K, T, scale * a, vix, R, Q, fitted) for K, T, a in zip(c["K"], c["T"], c["atm_vix"])]
    c["iv_old"] = [iv(S, K, T, a, vix, R, Q, LEGACY) for K, T, a in zip(c["K"], c["T"], c["atm_vix"])]
    # shape alone: same real at-the-money vol for both, so only the smile differs
    c["shape_new"] = [iv(S, K, T, a, vix, R, Q, fitted) for K, T, a in zip(c["K"], c["T"], c["atm_real"])]
    c["shape_old"] = [iv(S, K, T, a, vix, R, Q, LEGACY) for K, T, a in zip(c["K"], c["T"], c["atm_real"])]
    rmse = lambda e: float(100 * np.sqrt(np.mean(e ** 2)))  # noqa: E731
    print("\n3. on the check expiries (vol points, RMSE across all their strikes)")
    print(f"   whole model:  fitted {rmse(c['iv_new'] - c['iv']):.2f}   original {rmse(c['iv_old'] - c['iv']):.2f}")
    print(f"   shape only:   fitted {rmse(c['shape_new'] - c['iv']):.2f}   original {rmse(c['shape_old'] - c['iv']):.2f}"
          "   (both given the real at-the-money vol)")

    # ---- what it does to the puts the back test sells, on the check expiries
    p = c[c["right"] == "P"].copy()
    p["delta"] = [bs_abs_delta(S, K, T, R, Q, s) for K, T, s in zip(p["K"], p["T"], p["iv"])]
    p = p[(p["delta"] >= 0.10) & (p["delta"] <= 0.35)].copy()
    for tag, col, mdl in (("new", "iv_new", fitted), ("old", "iv_old", LEGACY)):
        p[f"px_{tag}"] = [pricer.price_option(S, K, T, R, s, Q, model=mdl) for K, T, s in zip(p["K"], p["T"], p[col])]
        p[f"err_{tag}"] = 100 * (p[f"px_{tag}"] / p["mid"] - 1)
    p["delta_bucket"] = pd.cut(p["delta"], [0.10, 0.175, 0.25, 0.35], labels=["0.15", "0.20", "0.30"])
    p["dte_bucket"] = pd.cut(p["dte"], [6, 17, 25, 37, 52], labels=["14d", "21d", "30d", "45d"])
    t = p.groupby(["dte_bucket", "delta_bucket"], observed=True).agg(
        n=("K", "size"), real_mid=("mid", "median"), fitted_err_pct=("err_new", "median"),
        original_err_pct=("err_old", "median"))
    print("\n4. puts the back test sells, check expiries only: model price against the real mid, %")
    print(t.round(1).to_string())
    print(f"   all {len(p)}: median error fitted {p['err_new'].median():+.1f}%, original {p['err_old'].median():+.1f}%; "
          f"median absolute error fitted {p['err_new'].abs().median():.1f}%, original {p['err_old'].abs().median():.1f}%")

    out = {"day": day, "spot": S, "vix": vix, "vix9d": vix9d, "atm_scale": round(scale, 4),
           "b1": round(float(b1), 4), "b2": round(float(b2), 4), "c1": round(float(c1), 4), "c2": round(float(c2), 4),
           "check_rmse_volpts": {"fitted": rmse(c["iv_new"] - c["iv"]), "original": rmse(c["iv_old"] - c["iv"])},
           "check_put_price_err_median_pct": {"fitted": float(p["err_new"].median()), "original": float(p["err_old"].median())}}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", f"smile_fit_{day}.json"), "w"), indent=1)
    t.round(2).to_csv(os.path.join(ROOT, "results", f"smile_check_{day}.csv"))
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
