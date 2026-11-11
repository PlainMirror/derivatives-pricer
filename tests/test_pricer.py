"""Checks for pricer.py: the CRR tree against Black-Scholes, the call-side fixes, and golden put prices.

Run with `python tests/test_pricer.py` (or `pytest tests`). No market data needed.
"""
import math
import os
import sys

from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

import pricer
from pricer import LEGACY, bin_tree

# These tests pin the ORIGINAL model (CRR tree, flat linear skew), so they call it explicitly.
iv = partial(pricer.iv, model=LEGACY)
strike_for_delta = partial(pricer.strike_for_delta, model=LEGACY)

S, R, Q = 400.0, 0.03, 0.012
VOLS = [0.12, 0.20, 0.30, 0.45]
DELTAS = [0.15, 0.20, 0.30]
DTES = [14, 21, 30, 45]          # 3 x 4 x 4 = 48 cells, same grid as the resume's validation


def bs_price(S, K, T, r, sig, q, call):
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if call:
        return S * math.exp(-q * T) * ndtr(d1) - K * math.exp(-r * T) * ndtr(d2)
    return K * math.exp(-r * T) * ndtr(-d2) - S * math.exp(-q * T) * ndtr(-d1)


def bs_abs_delta(S, K, T, r, sig, q, call):
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return math.exp(-q * T) * (ndtr(d1) if call else ndtr(-d1))


def grid(call):
    """Strike and skewed vol for every (vol, delta, DTE) cell, picked the way the backtest picks them."""
    for vol in VOLS:
        for delta in DELTAS:
            for dte in DTES:
                T = dte / 365.0
                K = strike_for_delta(S, T, R, Q, vol, vol * 100, delta, call=call)
                yield delta, T, K, iv(S, K, T, vol, vol * 100)


def max_rel_error(call, N):
    return max(abs(bin_tree(S, K, T, R, sig, Q, N, american=False, call=call)
                   / bs_price(S, K, T, R, sig, Q, call) - 1)
               for _, T, K, sig in grid(call))


# ------------------------------------------------------------- vs Black-Scholes
def test_european_put_matches_black_scholes():
    err = max_rel_error(call=False, N=50)
    assert err < 0.011, f"max put error {err:.4%} (measured 1.03% on Oct 17 2026)"


def test_european_call_matches_black_scholes():
    err = max_rel_error(call=True, N=50)
    assert err < 0.020, f"max call error {err:.4%} (measured 1.81% on Oct 17 2026)"


def test_error_shrinks_with_more_steps():
    for call in (False, True):
        err = max_rel_error(call=call, N=500)
        assert err < 0.0025, f"N=500 {'call' if call else 'put'} error {err:.4%}"


def test_put_call_parity_holds_exactly_in_the_tree():
    # p matches the forward, so a European C - P equals S e^(-qT) - K e^(-rT) up to rounding
    for _, T, K, sig in grid(call=False):
        c = bin_tree(S, K, T, R, sig, Q, 50, american=False, call=True)
        p = bin_tree(S, K, T, R, sig, Q, 50, american=False, call=False)
        assert abs((c - p) - (S * math.exp(-Q * T) - K * math.exp(-R * T))) < 1e-9


# -------------------------------------------------------------- early exercise
def test_american_call_without_dividends_equals_european():
    # never optimal to exercise a call early without dividends; the old call branch
    # compared against the put's intrinsic value and broke this
    for K in (80.0, 100.0, 120.0):
        am = bin_tree(100.0, K, 1.0, 0.05, 0.25, 0.0, 50, american=True, call=True)
        eu = bin_tree(100.0, K, 1.0, 0.05, 0.25, 0.0, 50, american=False, call=True)
        assert abs(am - eu) < 1e-12, (K, am, eu)


def test_deep_itm_call_with_high_dividend_exercises_early():
    am = bin_tree(100.0, 60.0, 1.0, 0.01, 0.25, 0.08, 50, american=True, call=True)
    eu = bin_tree(100.0, 60.0, 1.0, 0.01, 0.25, 0.08, 50, american=False, call=True)
    assert abs(am - 40.0) < 1e-9 and eu < am - 1.0, (am, eu)


def test_american_is_worth_at_least_european():
    for call in (False, True):
        for _, T, K, sig in grid(call):
            am = bin_tree(S, K, T, R, sig, Q, 50, american=True, call=call)
            eu = bin_tree(S, K, T, R, sig, Q, 50, american=False, call=call)
            assert am >= eu - 1e-12


def test_expiry_and_zero_vol_shortcuts():
    assert bin_tree(100.0, 105.0, 0.0, 0.03, 0.2, 0.0, 50, call=True) == 0.0
    assert bin_tree(110.0, 105.0, 0.0, 0.03, 0.2, 0.0, 50, call=True) == 5.0
    assert bin_tree(100.0, 105.0, 0.0, 0.03, 0.2, 0.0, 50, call=False) == 5.0
    fwd = 100.0 * math.exp((0.03 - 0.01) * 0.5)
    assert abs(bin_tree(100.0, 95.0, 0.5, 0.03, 0.0, 0.01, 50, call=True)
               - math.exp(-0.03 * 0.5) * (fwd - 95.0)) < 1e-12


# ---------------------------------------------------------------- strike picker
def test_strike_for_delta_hits_the_target():
    # 3 fixed-point iterations; puts land within 0.0014, calls within 0.0069 (the skew moves call vol more)
    for call, tol in ((False, 0.002), (True, 0.01)):
        for delta, T, K, sig in grid(call):
            got = bs_abs_delta(S, K, T, R, sig, Q, call)
            assert abs(got - delta) < tol, (call, delta, T, got)
            assert (K > S) if call else (K < S)


# ---------------------------------------------- put results unchanged by the fix
# (vol, delta, dte, strike, skewed vol, American put, European put), computed with
# pricer.py as it was before the Oct 17 2026 fix (_backup/pricer_before_fix_2026-10-17.py)
GOLDEN_PUTS = [
    (0.12, 0.15, 14, 390.41722023010266, 0.12436474518225603, 0.7759873557375926, 0.7733082452896016),
    (0.2, 0.3, 30, 388.94958819349887, 0.2084044415661155, 4.748364638102535, 4.7302604601291005),
    (0.3, 0.2, 21, 375.9551552403369, 0.32789760546400754, 3.667064230623633, 3.660024958481314),
    (0.45, 0.3, 45, 371.4018460588522, 0.5000714511711293, 14.87079695340031, 14.841049235021243),
    (0.2, 0.15, 45, 370.86803533783427, 0.22268547430243452, 2.554950396672439, 2.545809012131695),
]


def test_put_prices_unchanged_from_original():
    for vol, delta, dte, K0, sig0, am0, eu0 in GOLDEN_PUTS:
        T = dte / 365.0
        K = strike_for_delta(S, T, R, Q, vol, vol * 100, delta)
        sig = iv(S, K, T, vol, vol * 100)
        assert abs(K - K0) < 1e-9 and abs(sig - sig0) < 1e-12, (vol, delta, dte)
        assert abs(bin_tree(S, K, T, R, sig, Q, 50) - am0) < 1e-12
        assert abs(bin_tree(S, K, T, R, sig, Q, 50, american=False) - eu0) < 1e-12
    assert abs(bin_tree(100.0, 130.0, 1.0, 0.05, 0.25, 0.0, 50) - 30.066947875697636) < 1e-12
    assert bin_tree(100.0, 105.0, 0.0, 0.03, 0.2, 0.0, 50) == 5.0
    assert abs(bin_tree(100.0, 105.0, 0.5, 0.03, 0.0, 0.01, 50) - 3.935505739053364) < 1e-12


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
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
