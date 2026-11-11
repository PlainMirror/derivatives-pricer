"""Checks for the Leisen-Reimer tree, the fitted smile and the back test rules. No market data needed.

    python tests/test_lr_smile.py        # or: pytest tests
"""
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pricer  # noqa: E402
from pricer import MODEL, crr_tree, lr_tree  # noqa: E402


def bs(S, K, T, r, sig, q, call):
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if call:
        return S * math.exp(-q * T) * ndtr(d1) - K * math.exp(-r * T) * ndtr(d2)
    return K * math.exp(-r * T) * ndtr(-d2) - S * math.exp(-q * T) * ndtr(-d1)


CASES = [(400, 360, 30 / 365, 0.03, 0.18, 0.012), (400, 400, 14 / 365, 0.03, 0.14, 0.012),
         (400, 430, 45 / 365, 0.03, 0.12, 0.012), (100, 80, 1.0, 0.05, 0.35, 0.0)]


# ------------------------------------------------------------------ the tree
def test_lr_matches_black_scholes_closely():
    for S, K, T, r, sig, q in CASES:
        for call in (False, True):
            ref = bs(S, K, T, r, sig, q, call)
            got = lr_tree(S, K, T, r, sig, q, 201, american=False, call=call)
            assert abs(got - ref) < 2e-4 * max(ref, 0.05), (S, K, call, got, ref)


def test_lr_beats_crr_at_four_times_the_steps():
    """201 Leisen-Reimer steps against 801 CRR steps, European, same contracts."""
    for S, K, T, r, sig, q in CASES:
        ref = bs(S, K, T, r, sig, q, False)
        lr = abs(lr_tree(S, K, T, r, sig, q, 201, american=False) - ref)
        crr = abs(crr_tree(S, K, T, r, sig, q, 801, american=False) - ref)
        assert lr < crr, (K, lr, crr)


def test_lr_converges_smoothly():
    """The error shrinks steadily with N instead of zigzagging: each step up in N cuts it."""
    S, K, T, r, sig, q = 400, 372, 30 / 365, 0.03, 0.17, 0.012
    ref = bs(S, K, T, r, sig, q, False)
    errs = [abs(lr_tree(S, K, T, r, sig, q, n, american=False) - ref) for n in (25, 51, 101, 201)]
    assert all(b < a for a, b in zip(errs, errs[1:])), errs
    assert errs[0] / errs[-1] > 30, errs                 # roughly 1/N^2: 8x the steps, ~64x smaller


def test_even_steps_are_made_odd():
    S, K, T, r, sig, q = CASES[0]
    assert lr_tree(S, K, T, r, sig, q, 200) == lr_tree(S, K, T, r, sig, q, 201)


def test_american_lr_agrees_with_a_fine_crr_tree():
    for S, K, T, r, sig, q in CASES:
        for call in (False, True):
            a = lr_tree(S, K, T, r, sig, q, 201, american=True, call=call)
            b = crr_tree(S, K, T, r, sig, q, 2000, american=True, call=call)
            assert abs(a - b) < 2e-3 * max(b, 0.5), (K, call, a, b)


def test_early_exercise_in_the_lr_tree():
    S, T, r, sig = 100.0, 1.0, 0.05, 0.25
    assert lr_tree(S, 110, T, r, sig, 0.0, 201, True) > lr_tree(S, 110, T, r, sig, 0.0, 201, False) + 0.05
    am = lr_tree(S, 90, T, r, sig, 0.0, 201, True, call=True)
    eu = lr_tree(S, 90, T, r, sig, 0.0, 201, False, call=True)
    assert abs(am - eu) < 1e-9                           # no dividend: never exercise a call early
    deep = lr_tree(100.0, 60.0, 1.0, 0.01, 0.25, 0.08, 201, True, call=True)
    assert abs(deep - 40.0) < 1e-9                       # high dividend: exercise now


def test_degenerate_cases_fall_back_cleanly():
    v = lr_tree(767.0, 400.0, 7 / 365, 0.03, 0.01, 0.012, 201)   # p rounds to 1 here
    assert 0.0 <= v < 1e-9
    assert lr_tree(100.0, 105.0, 0.0, 0.03, 0.2, 0.0, 201) == 5.0


# ----------------------------------------------------------------- the smile
def test_smile_is_the_atm_vol_at_the_forward_and_continuous():
    S, T, r, q, a = 767.0, 30 / 365, 0.03, 0.012, 0.12
    F = S * math.exp((r - q) * T)
    assert abs(pricer.iv(S, F, T, a, 15.0, r, q) - a) < 1e-12
    lo, hi = pricer.iv(S, F * (1 - 1e-9), T, a, 15.0, r, q), pricer.iv(S, F * (1 + 1e-9), T, a, 15.0, r, q)
    assert abs(lo - hi) < 1e-8


def test_smile_has_the_equity_skew():
    S, T, r, q, a = 767.0, 30 / 365, 0.03, 0.012, 0.12
    vols = [pricer.iv(S, K, T, a, 15.0, r, q) for K in (650, 700, 740, 767, 790)]
    assert vols[0] > vols[1] > vols[2] > vols[3], vols          # puts dearer the further out
    assert vols[4] < vols[3]                                    # just above spot, cheaper than ATM


def test_smile_scales_with_sqrt_time():
    """The same standardised moneyness gives the same vol ratio at any expiry."""
    S, r, q, a = 767.0, 0.03, 0.012, 0.12
    out = []
    for T in (7 / 365, 30 / 365, 45 / 365):
        F = S * math.exp((r - q) * T)
        K = F * math.exp(-1.2 * a * math.sqrt(T))               # x = -1.2
        out.append(pricer.iv(S, K, T, a, 15.0, r, q) / a)
    assert max(out) - min(out) < 1e-12, out


def test_smile_is_held_flat_beyond_the_fitted_range():
    S, T, r, q, a = 767.0, 30 / 365, 0.03, 0.012, 0.12
    F = S * math.exp((r - q) * T)
    far = [pricer.iv(S, F * math.exp(x * a * math.sqrt(T)), T, a, 15.0, r, q) for x in (-4.0, -6.0, -10.0)]
    assert max(far) - min(far) < 1e-12, far


def test_strike_picker_hits_the_delta_with_the_fitted_smile():
    S, r, q, a = 767.0, 0.03, 0.012, 0.12
    for T in (14 / 365, 30 / 365, 45 / 365):
        for target in (0.15, 0.20, 0.30):
            for call in (False, True):
                K = pricer.strike_for_delta(S, T, r, q, a, 15.0, target, call=call)
                sig = pricer.iv(S, K, T, a, 15.0, r, q)
                assert abs(pricer.bs_abs_delta(S, K, T, r, q, sig, call) - target) < 1e-4
                assert (K > S) if call else (K < S)


# ---------------------------------------------------------------- back test
def _market(n=400, seed=0, vix=16.0):
    rng = np.random.default_rng(seed)
    px = 400 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=n), "spy_close": px,
                         "vix_close": vix, "vix9d_close": vix - 1.0, "spy_div": 0.0})


def test_cash_secured_puts_never_exceed_the_cash():
    df = _market()
    eq, trades, daily, info = pricer.backtest(df, capital=100_000, dte=30, target_delta=0.2,
                                              collateral="cash", record_daily=True)
    assert info["shares"] == 0 and len(trades) > 3
    for _, t in trades.iterrows():
        assert t["contracts"] * t["strike"] * 100 <= eq.max() * 1.0001


def test_rules_change_what_trades():
    df = _market(seed=1)
    base = pricer.backtest(df, dte=30, target_delta=0.2)[1]
    calm_only = pricer.backtest(df.assign(vix_close=25.0, vix9d_close=24.0), dte=30, target_delta=0.2)[1]
    assert len(base) > 0 and len(calm_only) == 0          # VIX 25 is above the cap of 19: never sells
    always = pricer.backtest(df.assign(vix_close=25.0, vix9d_close=24.0), dte=30, target_delta=0.2, entry="always")[1]
    assert len(always) > 0
    back = pricer.backtest(df.assign(vix9d_close=17.0), dte=30, target_delta=0.2, entry="always+term")[1]
    assert len(back) == 0                                  # VIX9D above VIX on every day: backwardation
    timed = pricer.backtest(df, dte=45, target_delta=0.2, time_exit=1 / 3)[1]
    assert (timed["reason"] != "expiry").all() and (timed["reason"] == "time_exit").any()
    assert timed["days_held"].max() <= 45 * 2 / 3 + 4      # closed with a third of the life left


def test_delta_stop_closes_a_losing_put():
    n = 60
    px = np.r_[np.full(10, 400.0), np.linspace(400, 340, n - 10)]   # a steady fall
    df = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=n), "spy_close": px,
                       "vix_close": 16.0, "vix9d_close": 15.0, "spy_div": 0.0})
    tr = pricer.backtest(df, dte=45, target_delta=0.2, stop="delta", stop_delta=0.5, entry="always")[1]
    assert (tr["reason"] == "stop").any(), tr


if __name__ == "__main__":
    assert MODEL["tree"] == "lr"
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
