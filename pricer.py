"""Binomial option pricer and a daily back test of selling SPY options.

    python pricer.py                      # 2010 to today: delta x DTE sweep, then one run in detail
    python pricer.py --legacy             # the original model: CRR tree, 50 steps, VIX as ATM vol, flat skew
    python pricer.py --collateral cash    # cash-secured puts instead of puts sold against a share position

Three parts:
  vol model   at-the-money vol from VIX9D and VIX, interpolated in total variance, scaled to the
              at-the-money level of a real SPY chain, with a smile fitted to that chain
  pricer      Leisen-Reimer binomial tree (201 steps) with early exercise; Cox-Ross-Rubinstein kept
  back test   sells a put (or a covered call) at a target delta and days to expiry, marks it to market
              every day, and closes it on a profit target, a stop, a time exit or expiry

Every option price in the back test comes from the vol model, not from quotes. smile_fit.py fits the
model to a real SPY chain and checks it on expiries it was not fitted on.
"""
import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
CACHE = os.path.join(ROOT, "cache")

# ------------------------------------------------------------------ settings
# Every modelling choice in one place. LEGACY reproduces the original model, which the golden
# tests in tests/test_pricer.py pin down.
MODEL = {"tree": "lr", "steps": 201, "smile": "fitted", "atm_scale": 0.812}
LEGACY = {"tree": "crr", "steps": 50, "smile": "linear", "atm_scale": 1.0}

# Linear skew of the original model: slope of vol against log-strike, at VIX 20.
SKEW_BASE = 0.30

# Fitted smile, in standardised moneyness x = ln(K/F) / (sigma_atm * sqrt(T)), one quadratic per side:
#     sigma(x) = sigma_atm * (1 + B1 x + B2 x^2)    below the forward (the put wing)
#     sigma(x) = sigma_atm * (1 + C1 x + C2 x^2)    above it
# The coefficients and the at-the-money scale above are set by smile_fit.py from the SPY chain at
# the 2026-09-24 close, using alternate expiries, and checked on the other half.
SMILE_B1, SMILE_B2 = -0.1532, 0.0246
SMILE_C1, SMILE_C2 = -0.2073, 0.1019
SMILE_X_RANGE = (-4.0, 3.0)   # beyond the fitted strikes the shape is held flat, not extrapolated
SMILE_FLOOR = 0.5             # vol never below half the at-the-money vol


# ---------------------------------------------------------------- vol model
def atm_vol(vix9d: float, vix: float, T_years: float) -> float:
    """At-the-money vol for a maturity between 9 and 30 days, read straight off VIX9D and VIX.

    Variance adds over time, so the interpolation is linear in total variance (vol^2 x days),
    not in vol. Outside 9 to 30 days the nearer index is used as is.
    """
    t = T_years * 365.0
    s9, s30 = vix9d / 100.0, vix / 100.0
    if t <= 9.0:
        return s9
    if t >= 30.0:
        return s30
    tv = s9 * s9 * 9.0 + (s30 * s30 * 30.0 - s9 * s9 * 9.0) * (t - 9.0) / 21.0
    return math.sqrt(tv / t)


def model_atm(vix9d: float, vix: float, T_years: float, model=None) -> float:
    """VIX prices a strip of options across all strikes, expensive downside puts included, so in a
    skewed market it sits above at-the-money vol. The scale corrects for that."""
    m = model or MODEL
    return m["atm_scale"] * atm_vol(vix9d, vix, T_years)


def skew_slope(vix_level: float) -> float:
    """The original linear skew: slope of vol against log-strike, steeper when VIX is high."""
    return SKEW_BASE * vix_level / 20.0


def iv(S: float, K: float, T: float, sigma_atm: float, vix_level: float,
       r: float = 0.03, q: float = 0.012, model=None) -> float:
    """Implied vol at strike K, given the at-the-money vol."""
    m = model or MODEL
    if m["smile"] == "linear":
        return max(sigma_atm - skew_slope(vix_level) * math.log(K / S), 0.01)
    if T <= 0:
        return max(sigma_atm, 0.01)
    F = S * math.exp((r - q) * T)
    x = math.log(K / F) / (sigma_atm * math.sqrt(T))
    x = min(max(x, SMILE_X_RANGE[0]), SMILE_X_RANGE[1])
    b1, b2 = (SMILE_B1, SMILE_B2) if x < 0 else (SMILE_C1, SMILE_C2)
    shape = max(1.0 + b1 * x + b2 * x * x, SMILE_FLOOR)
    return max(sigma_atm * shape, 0.01)


def bs_abs_delta(S, K, T, r, q, sig, call=False) -> float:
    """|Black-Scholes delta|, used to pick strikes and for the delta stop."""
    if T <= 0:
        return float((S > K) if call else (S < K))
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return math.exp(-q * T) * (ndtr(d1) if call else ndtr(-d1))


def strike_for_delta(S, T, r, q, sigma_atm, vix_level, target_delta, n_iter=None, call=False, model=None):
    """Strike whose Black-Scholes |delta|, at that strike's own smile vol, equals target_delta.

    The vol depends on the strike through the smile, so this iterates: guess a vol, solve for the
    strike, read the vol there, repeat. Each pass shrinks the error by a factor of about 0.3 at
    these deltas, so ten passes are far more than enough.
    """
    m = model or MODEL
    if n_iter is None:
        n_iter = 3 if m["smile"] == "linear" else 10
    z = ndtri(target_delta * math.exp(q * T))
    d1_target = z if call else -z
    sig, K = sigma_atm, S
    for _ in range(n_iter):
        K = S * math.exp((r - q + 0.5 * sig * sig) * T - d1_target * sig * math.sqrt(T))
        sig = iv(S, K, T, sigma_atm, vix_level, r, q, m)
    return K


# ------------------------------------------------------------------- pricer
def _short_cuts(S, K, T, r, sig, q, call):
    if T <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    if sig <= 0:
        fwd = S * math.exp((r - q) * T)
        payoff = max(fwd - K, 0.0) if call else max(K - fwd, 0.0)
        return math.exp(-r * T) * payoff
    return None


def crr_tree(S, K, T, r, sig, q, N, american=True, call=False):
    """Cox-Ross-Rubinstein: u = e^(sigma sqrt(dt)), d = 1/u, probability set from the forward."""
    sc = _short_cuts(S, K, T, r, sig, q, call)
    if sc is not None:
        return sc
    dt_step = T / N
    u = math.exp(sig * dt_step ** 0.5)
    d = 1.0 / u
    disc = math.exp(-r * dt_step)
    p = (math.exp((r - q) * dt_step) - d) / (u - d)
    assert 0.0 <= p <= 1.0, f"p={p:.4f} outside [0,1]. Increase N"

    j = np.arange(N + 1)
    if call:
        V = np.maximum(S * u ** (2 * j - N) - K, 0.0)
    else:
        V = np.maximum(K - S * u ** (2 * j - N), 0.0)

    for m in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            i = np.arange(m + 1)
            if call:
                V = np.maximum(V, S * u ** (2 * i - m) - K)
            else:
                V = np.maximum(V, K - S * u ** (2 * i - m))
    return float(V[0])


bin_tree = crr_tree   # the original name, kept for the golden tests


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt method 2: a binomial probability that matches N(z) closely for n steps."""
    a = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    return 0.5 + math.copysign(0.5, z) * math.sqrt(1.0 - math.exp(-a * a * (n + 1.0 / 6.0)))


def lr_tree(S, K, T, r, sig, q, N, american=True, call=False):
    """Leisen-Reimer (1996) binomial tree.

    CRR spaces its nodes by volatility alone, so the strike falls at a different place between two
    nodes every time N changes, and the error zigzags as N grows. Leisen-Reimer instead chooses the
    up-probabilities so that the tree's probabilities of finishing in the money match N(d2) and
    N(d1), via the Peizer-Pratt inversion. That centres the tree on the strike: the European price
    converges smoothly at about 1/N^2 instead of 1/N. It needs an odd number of steps.
    """
    sc = _short_cuts(S, K, T, r, sig, q, call)
    if sc is not None:
        return sc
    if N % 2 == 0:
        N += 1
    dt_step = T / N
    sq = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / sq
    d2 = d1 - sq
    p = _peizer_pratt(d2, N)
    p_star = _peizer_pratt(d1, N)
    if not (1e-10 < p < 1 - 1e-10 and 1e-10 < p_star < 1 - 1e-10):
        # Far from the money with tiny vol, d1 and d2 are so large that p rounds to 0 or 1 and the
        # tree has no width. The option is then worth its discounted forward intrinsic value to
        # many decimals, which CRR gets right, so it is used for these cases only.
        return crr_tree(S, K, T, r, sig, q, N, american=american, call=call)
    growth = math.exp((r - q) * dt_step)
    u = growth * p_star / p
    d = (growth - p * u) / (1.0 - p)
    disc = math.exp(-r * dt_step)

    j = np.arange(N + 1)
    ST = S * u ** j * d ** (N - j)          # j = number of up moves
    V = np.maximum(ST - K, 0.0) if call else np.maximum(K - ST, 0.0)
    Sn = ST
    for m in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            Sn = Sn[:-1] / d                # node prices one step earlier
            V = np.maximum(V, Sn - K) if call else np.maximum(V, K - Sn)
    return float(V[0])


def price_option(S, K, T, r, sig, q, call=False, american=True, model=None):
    m = model or MODEL
    tree = lr_tree if m["tree"] == "lr" else crr_tree
    return tree(S, K, T, r, sig, q, m["steps"], american=american, call=call)


# ----------------------------------------------------------------- backtest
def backtest(df, capital=500_000, dte=45, target_delta=0.50,
             profit_target=0.70, loss_mult=2.0, cost_per_side=0.01,
             r=0.03, q=0.012, record_daily=False, side="put",
             entry="vix_cap", vix_cap=None, time_exit=None, stop="loss", stop_delta=0.50,
             collateral="overlay", long_target=1.0, long_stop=0.5, model=None):
    """Sell one option series at a time, mark it every day, close it, and sell the next.

    entry       "vix_cap" sell only when VIX <= vix_cap (19 for puts, 22 for calls, the original rule)
                "always"  sell whenever flat
                "vrp"     sell only when model at-the-money vol is above 20-day realised vol
                "term"    skip days when VIX9D > VIX (short-dated fear above 30-day: backwardation)
                rules can be combined with "+", e.g. "vrp+term"
    stop        "loss"    close when the mark-to-market loss reaches loss_mult x the premium
                "delta"   close when the option's |delta| reaches stop_delta
    time_exit   close when this fraction of the option's life is left (None = hold to expiry)
    collateral  "overlay" hold SPY with the whole account and sell puts (or covered calls) on it
                "cash"    hold cash at the risk-free rate and sell as many puts as it secures
    side        "put" or "call" sells the option; "long_call" BUYS calls instead. A long call is sized
                to the same SPY exposure as the account (delta x 100 x spot per contract = equity), the
                rest stays in cash, and it closes at a gain of long_target or a loss of long_stop (as
                fractions of the premium paid), or on time_exit or expiry.
    """
    m = model or MODEL
    if side not in ("put", "call", "long_call"):
        raise ValueError(f"side must be 'put', 'call' or 'long_call', got {side!r}")
    long = side == "long_call"
    if long:
        collateral = "cash"
    if collateral not in ("overlay", "cash"):
        raise ValueError(f"collateral must be 'overlay' or 'cash', got {collateral!r}")
    if side == "call" and collateral == "cash":
        raise ValueError("a call is covered by shares, so it needs collateral='overlay'")
    if vix_cap is None:
        vix_cap = 19.0 if side == "put" else 22.0
    rules = set(entry.split("+"))
    unknown = rules - {"vix_cap", "always", "vrp", "term"}
    if unknown:
        raise ValueError(f"unknown entry rule(s): {sorted(unknown)}")
    if stop not in ("loss", "delta"):
        raise ValueError(f"stop must be 'loss' or 'delta', got {stop!r}")
    call = side in ("call", "long_call")

    df = df.sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(df["Date"]).to_numpy()
    spy = df["spy_close"].to_numpy(dtype=float)
    vix = df["vix_close"].to_numpy(dtype=float)
    n = len(df)
    div = df["spy_div"].to_numpy(float) if "spy_div" in df.columns else np.zeros(n)
    vix9d = df["vix9d_close"].to_numpy(float) if "vix9d_close" in df.columns else vix.copy()
    # realised vol over the last 20 sessions, today's close included (decisions are made at the close)
    rv20 = (pd.Series(np.log(spy)).diff().rolling(20).std() * math.sqrt(252)).to_numpy()

    if collateral == "overlay":
        shares = math.floor(capital / spy[0])
        cash = capital - shares * spy[0]
    else:
        shares, cash = 0.0, float(capital)
    contracts = 0

    pos = None
    equity = np.empty(n)
    trades, daily = [], []
    rf_d = (1 + r) ** (1 / 252) - 1

    def days_left(i):
        return (pos["expiry"] - dates[i]) / np.timedelta64(1, "D")

    def mark(i):
        """Model value and vol of the open option today."""
        T = max(days_left(i), 0) / 365.0
        sig = iv(spy[i], pos["strike"], T, model_atm(vix9d[i], vix[i], T, m), vix[i], r, q, m)
        return price_option(spy[i], pos["strike"], T, r, sig, q, call=call, model=m), sig, T

    def can_open(i):
        ok = True
        if "vix_cap" in rules:
            ok &= vix[i] <= vix_cap
        if "vrp" in rules:
            ok &= bool(np.isfinite(rv20[i])) and model_atm(vix9d[i], vix[i], dte / 365.0, m) > rv20[i]
        if "term" in rules:
            ok &= vix9d[i] <= vix[i]
        return ok

    def open_position(i):
        nonlocal cash, pos, contracts, shares
        T = dte / 365.0
        sig_atm = model_atm(vix9d[i], vix[i], T, m)
        K = strike_for_delta(spy[i], T, r, q, sig_atm, vix[i], target_delta, call=call, model=m)
        sig = iv(spy[i], K, T, sig_atm, vix[i], r, q, m)
        price = price_option(spy[i], K, T, r, sig, q, call=call, model=m)
        delta0 = bs_abs_delta(spy[i], K, T, r, q, sig, call)
        if long:
            # stock replacement: as much SPY exposure (delta-weighted) as the account is worth
            contracts = int(cash // (delta0 * 100 * spy[i]))
        elif collateral == "overlay":
            # re-level the share position to the whole account, then write one option per 100 shares
            target = math.floor((cash + shares * spy[i]) / spy[i])
            cash -= (target - shares) * spy[i]
            shares = target
            contracts = int(shares) // 100
        else:
            # every put fully secured by cash: strike x 100 per contract
            contracts = int(cash // (K * 100))
        if contracts == 0:
            return None
        premium = price * 100 * contracts
        fee = cost_per_side * premium
        cash += (-premium if long else premium) - fee
        pos = {"call": call, "entry_price": price, "strike": K, "open_date": dates[i],
               "expiry": dates[i] + np.timedelta64(dte, "D"), "premium": premium, "open_fee": fee,
               "entry_delta": delta0}
        return {"opened_strike": round(K, 2), "sold_px": round(price, 2), "premium_in_$": round(premium, 2)}

    def close_position(i, reason, val):
        nonlocal cash, pos
        close_val = val * 100 * contracts
        fee = cost_per_side * close_val
        cash += (close_val if long else -close_val) - fee
        gross = (close_val - pos["premium"]) if long else (pos["premium"] - close_val)
        net = gross - pos["open_fee"] - fee
        trades.append({"open": pd.Timestamp(pos["open_date"]).date(),
                       "close": pd.Timestamp(dates[i]).date(), "reason": reason,
                       "days_held": int((dates[i] - pos["open_date"]) / np.timedelta64(1, "D")),
                       "strike": round(pos["strike"], 2), "entry_delta": round(pos["entry_delta"], 3),
                       "entry_px": round(pos["entry_price"], 2), "exit_px": round(val, 2),
                       "vix_at_open": None, "vix_at_close": round(vix[i], 1),
                       "contracts": contracts, "gross_$": round(gross), "net_$": round(net)})
        pos = None
        return {"closed_reason": reason, "buyback_px": round(val, 2), "realized_pl_$": round(net)}

    for i in range(n):
        cash *= (1 + rf_d)
        if div[i] > 0 and shares:
            shares += (shares * div[i]) / spy[i]
        closed = opened = None
        val = None
        if pos is not None:
            val, sig, T = mark(i)
            left = days_left(i)
            if long:
                stopped = val <= (1 - long_stop) * pos["entry_price"]
            elif stop == "loss":
                stopped = (val - pos["entry_price"]) * 100 * contracts >= loss_mult * pos["premium"]
            else:
                stopped = left > 0 and bs_abs_delta(spy[i], pos["strike"], T, r, q, sig, call) >= stop_delta
            if stopped:
                closed = close_position(i, "stop", val)
            elif (val >= (1 + long_target) * pos["entry_price"]) if long else (val <= (1 - profit_target) * pos["entry_price"]):
                closed = close_position(i, "profit_target", val)
            elif time_exit is not None and 0 < left <= time_exit * dte:
                closed = close_position(i, "time_exit", val)
            elif left <= 0:
                closed = close_position(i, "expiry", val)
            if closed:
                val = None
        if pos is None and can_open(i):
            opened = open_position(i)
            if opened:
                val = pos["entry_price"]

        opt = val * 100 * contracts if pos is not None else 0.0
        equity[i] = cash + shares * spy[i] + (opt if long else -opt)

        if record_daily:
            row = {"date": pd.Timestamp(dates[i]).date(), "spy": round(spy[i], 2), "vix": round(vix[i], 1)}
            if pos is not None:
                row.update({"strike": round(pos["strike"], 2), "sold_at": round(pos["entry_price"], 2),
                            "option_val_now": round(val, 2),
                            "unreal_pl_$": round((val - pos["entry_price"]) * (1 if long else -1) * 100 * contracts),
                            "dte_left": int(days_left(i))})
            if closed:
                row.update(closed)
            if opened:
                row.update(opened)
            row["equity_$"] = round(equity[i])
            daily.append(row)

    eq = pd.Series(equity, index=pd.to_datetime(df["Date"]))
    cols = ["date", "spy", "vix", "strike", "sold_at", "option_val_now",
            "unreal_pl_$", "dte_left", "closed_reason", "buyback_px",
            "realized_pl_$", "opened_strike", "sold_px", "premium_in_$", "equity_$"]
    daily_df = pd.DataFrame(daily).reindex(columns=cols) if daily else pd.DataFrame(columns=cols)
    tr = pd.DataFrame(trades)
    if len(tr):
        tr = tr.drop(columns=["vix_at_open"])
    return eq, tr, daily_df, {"shares": shares, "contracts": contracts}


# ------------------------------------------------------------------ metrics
def sharpe_of(eq: pd.Series, r=0.03) -> float:
    ret = eq.pct_change().dropna()
    rf_d = (1 + r) ** (1 / 252) - 1
    return float((ret.mean() - rf_d) / ret.std() * math.sqrt(252))


def metrics(eq, df, capital=500_000, r=0.03):
    drip = (1 + df.get("spy_div", 0) / df["spy_close"]).cumprod()
    bench = pd.Series((df["spy_close"] * drip).to_numpy(float),
                      index=pd.to_datetime(df["Date"]))
    bench_final = capital * (bench.iloc[-1] / bench.iloc[0])
    both = pd.concat([eq.pct_change(), bench.pct_change()], axis=1).dropna()
    ret, bret = both.iloc[:, 0], both.iloc[:, 1]
    rf_d = (1 + r) ** (1 / 252) - 1
    ann = math.sqrt(252)

    sharpe = (ret.mean() - rf_d) / ret.std() * ann
    downside = ret[ret < rf_d] - rf_d
    sortino = (ret.mean() - rf_d) / math.sqrt((downside ** 2).mean()) * ann
    mdd = (eq / eq.cummax() - 1).min()

    ex_s, ex_b = ret - rf_d, bret - rf_d
    beta = np.cov(ex_s, ex_b)[0, 1] / np.var(ex_b)
    alpha_ann = (ex_s.mean() - beta * ex_b.mean()) * 252

    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "cagr_%":               round(((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, 2),
        "net_profit_$":         round(eq.iloc[-1] - capital),
        "total_return_%":       round((eq.iloc[-1] / capital - 1) * 100, 1),
        "excess_vs_spy_$":      round(eq.iloc[-1] - bench_final),
        "ann_alpha_%":          round(alpha_ann * 100, 2),
        "beta_vs_spy":          round(beta, 2),
        "sharpe":               round(sharpe, 2),
        "sortino":              round(sortino, 2),
        "max_drawdown_%":       round(mdd * 100, 1),
    }


# -------------------------------------------------------------------- sweep
def run_sweep(df, deltas, dtes, **backtest_kwargs):
    rows = {}
    for d in deltas:
        for t in dtes:
            eq, trades, _, _ = backtest(df, dte=t, target_delta=d, **backtest_kwargs)
            mt = metrics(eq, df)
            rc = (trades["reason"].value_counts(normalize=True)
                  if len(trades) else pd.Series(dtype=float))
            mt.update({
                "n_trades":       len(trades),
                "win_rate_%":     round((trades["net_$"] > 0).mean() * 100, 1) if len(trades) else np.nan,
                "pct_stop":       round(rc.get("stop", 0) * 100, 1),
                "pct_profit_tgt": round(rc.get("profit_target", 0) * 100, 1),
                "pct_time_exit":  round(rc.get("time_exit", 0) * 100, 1),
                "pct_expiry":     round(rc.get("expiry", 0) * 100, 1),
            })
            rows[(d, t)] = mt
            print(f"delta={d:.2f}  dte={t:3d}  sharpe={mt['sharpe']:5.2f}  "
                  f"sortino={mt['sortino']:5.2f}  mdd={mt['max_drawdown_%']:6.1f}%  "
                  f"trades={mt['n_trades']}", flush=True)
    res = pd.DataFrame(rows).T
    res.index.names = ["delta", "dte"]
    return res


def save_outputs(res, trades, daily, m, out_dir=OUT):
    """Sweep table, the detailed run's trades and daily blotter, and its metrics, as CSV."""
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "sweep.csv"))
    for col in ["sharpe", "sortino", "max_drawdown_%", "ann_alpha_%"]:
        res[col].unstack("dte").to_csv(os.path.join(out_dir, f"sweep_{col.strip('_%')}.csv"))
    d = daily.copy()
    d["cum_realized_pl_$"] = d["realized_pl_$"].fillna(0).cumsum()
    d.to_csv(os.path.join(out_dir, "daily_blotter.csv"), index=False)
    trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    pd.Series(m, name="value").rename_axis("metric").to_csv(os.path.join(out_dir, "metrics.csv"))
    print(f"\nSaved sweep, trades, blotter and metrics to {out_dir}")


def load_data(start="2010-01-01", end=None, cache=True):
    """Daily SPY, VIX and VIX9D closes and SPY dividends from Yahoo, cached once per day."""
    path = os.path.join(CACHE, f"spy_vix_{start}_{end or pd.Timestamp.today().date()}.parquet")
    if cache and os.path.exists(path):
        return pd.read_parquet(path)
    import yfinance as yf
    data = yf.download(["SPY", "^VIX", "^VIX9D"], start=start, end=end,
                       auto_adjust=False, actions=True, progress=False)
    close = data["Close"]
    core = close[["SPY", "^VIX"]].dropna(how="any")        # VIX9D stays OUT
    v9 = (close["^VIX9D"].reindex(core.index)
          .ffill(limit=5)
          .fillna(core["^VIX"]))                            # VIX9D starts in 2011: before that, VIX
    div = (data["Dividends"]["SPY"].reindex(core.index).fillna(0.0)
           if "Dividends" in data.columns.get_level_values(0)
           else pd.Series(0.0, index=core.index))
    out = pd.DataFrame({
        "Date":        core.index,
        "spy_close":   core["SPY"].to_numpy(float),
        "vix_close":   core["^VIX"].to_numpy(float),
        "vix9d_close": v9.to_numpy(float),
        "spy_div":     div.to_numpy(float),
    })
    n_fb = int(close["^VIX9D"].reindex(core.index).isna().sum())
    print(f"{out['Date'].iloc[0].date()} to {out['Date'].iloc[-1].date()}: {len(out)} days, "
          f"VIX9D missing on {n_fb} (VIX used instead)")
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        out.to_parquet(path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="exclusive; default is today")
    ap.add_argument("--side", default="put", choices=["put", "call", "long_call"])
    ap.add_argument("--delta", type=float, default=0.20, help="target delta of the detailed run")
    ap.add_argument("--dte", type=int, default=30, help="days to expiry of the detailed run")
    ap.add_argument("--legacy", action="store_true", help="original model: CRR, 50 steps, flat skew")
    ap.add_argument("--entry", default="vix_cap")
    ap.add_argument("--profit-target", type=float, default=0.70)
    ap.add_argument("--time-exit", type=float, default=None)
    ap.add_argument("--stop", default="loss", choices=["loss", "delta"])
    ap.add_argument("--collateral", default="overlay", choices=["overlay", "cash"])
    ap.add_argument("--atm-scale", type=float, default=None)
    a = ap.parse_args()
    if a.legacy:
        MODEL.update(LEGACY)
    if a.atm_scale is not None:
        MODEL["atm_scale"] = a.atm_scale
    rules = dict(side=a.side, entry=a.entry, profit_target=a.profit_target, time_exit=a.time_exit,
                 stop=a.stop, collateral=a.collateral)

    DELTAS = [0.15, 0.20, 0.30]                # |delta| of the option sold
    DTES = [14, 21, 30, 45]                    # calendar days

    df = load_data(a.start, a.end)
    print(f"model: {MODEL}; rules: {rules}")
    res = run_sweep(df, DELTAS, DTES, **rules)
    eq, trades, daily, info = backtest(df, dte=a.dte, target_delta=a.delta, record_daily=True, **rules)
    m = metrics(eq, df)
    m["n_trades"] = len(trades)
    print(f"\ndetailed run: delta {a.delta}, {a.dte} days: " + ", ".join(f"{k} {v}" for k, v in m.items()))
    save_outputs(res, trades, daily, m)
