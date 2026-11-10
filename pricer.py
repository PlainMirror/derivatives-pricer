"""Binomial option pricer and a daily back test of selling SPY options against a share position.

    python pricer.py                      # 2010 to today: delta x DTE sweep, then one run in detail
    python pricer.py --start 2022-01-01 --end 2022-12-31

Three parts:
  vol model   at-the-money vol from VIX9D and VIX, interpolated in total variance, plus a linear skew
  pricer      vectorised Cox-Ross-Rubinstein tree with early exercise
  back test   holds SPY, sells a put (or a covered call) at a target delta and days to expiry, marks it
              to market every day, and closes on a profit target, a loss cap or expiry

Every option price in the back test comes from the vol model, not from quotes. validate_surface.py
measures how far that model is from a real SPY chain.
"""
import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# ---------------------------------------------------------------- vol model
def atm_vol(vix9d: float, vix: float, T_years: float) -> float:
    """At-the-money vol for a maturity between 9 and 30 days.

    VIX9D and VIX are 9-day and 30-day implied vols. Variance adds over time, so
    the interpolation is linear in total variance (vol^2 x days), not in vol.
    Outside 9 to 30 days the nearer index is used as is.
    """
    t = T_years * 365.0
    s9, s30 = vix9d / 100.0, vix / 100.0
    if t <= 9.0:
        return s9
    if t >= 30.0:
        return s30
    tv = s9 * s9 * 9.0 + (s30 * s30 * 30.0 - s9 * s9 * 9.0) * (t - 9.0) / 21.0
    return math.sqrt(tv / t)


# The two assumptions of the vol model. The defaults are the original ones. validate_surface.py
# measures both against a real chain, and --atm-scale / --skew-base rerun the back test with other values.
ATM_SCALE = 1.0     # at-the-money vol as a fraction of the VIX-based number above
SKEW_BASE = 0.30    # slope of vol against log-strike when VIX is 20


def model_atm(vix9d: float, vix: float, T_years: float) -> float:
    return ATM_SCALE * atm_vol(vix9d, vix, T_years)


def skew_slope(vix_level: float) -> float:
    base_slope = SKEW_BASE           # assumed SPY skew; see SKEW_BASE
    vol_scaling = vix_level / 20.0   # steeper skew in high-vol regimes
    return base_slope * vol_scaling


def iv(S: float, K: float, T: float, sigma_atm: float, vix_level: float) -> float:
    mon = math.log(K / S)
    sig_K = sigma_atm - skew_slope(vix_level) * mon
    return max(sig_K, 0.01)


def strike_for_delta(S, T, r, q, sigma_atm, vix_level, target_delta, n_iter=3, call=False):
    # Black-Scholes delta is e^(-qT) N(d1) for a call and -e^(-qT) N(-d1) for a put.
    # target_delta is the absolute value, so only the sign of d1 differs.
    z = ndtri(target_delta * math.exp(q * T))
    d1_target = z if call else -z
    sig, K = sigma_atm, S
    for _ in range(n_iter):
        K = S * math.exp((r - q + 0.5 * sig * sig) * T - d1_target * sig * math.sqrt(T))
        sig = iv(S, K, T, sigma_atm, vix_level)
    return K


# ------------------------------------------------------------------- pricer
def bin_tree(S, K, T, r, sig, q, N, american=True,call = False):
    if T <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    if sig <= 0:
        fwd = S * math.exp((r - q) * T)
        payoff = max(fwd - K, 0.0) if call else max(K - fwd, 0.0)
        return math.exp(-r * T) * payoff

    dt_step = T / N
    u = math.exp(sig * dt_step ** 0.5)
    d = 1.0 / u
    disc = math.exp(-r * dt_step)
    p = (math.exp((r - q) * dt_step) - d) / (u - d)
    assert 0.0 <= p <= 1.0, f"p={p:.4f} outside [0,1]. Increase N"

    j = np.arange(N + 1)
    if call:
        V = np.maximum(S*u**(2*j - N) - K, 0.0)
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


# ----------------------------------------------------------------- backtest
def backtest(df, capital=500_000, dte=45, target_delta=0.50,
             profit_target=0.70, loss_mult=2.0, cost_per_side=0.01,
             r=0.03, q=0.012, N=50, record_daily = False, side="put"):

    if side not in ("put", "call"):
        raise ValueError(f"side must be 'put' or 'call', got {side!r}")
    df = df.sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(df["Date"]).to_numpy()
    spy = df["spy_close"].to_numpy(dtype=float)
    vix = df["vix_close"].to_numpy(dtype=float)
    n = len(df)
    div = df["spy_div"].to_numpy(float) if "spy_div" in df.columns else np.zeros(n)
    vix9d = df["vix9d_close"].to_numpy(float) if "vix9d_close" in df.columns else vix.copy()

    def atm_vol_at(i, T_years):
        return model_atm(vix9d[i], vix[i], T_years)

    shares = math.floor(capital / spy[0])
    contracts = shares // 100
    cash = capital - shares * spy[0]

    pos = None
    equity = np.empty(n)
    trades = []
    daily = []
    rf_d = (1+r)**(1/252)-1

    

    def value_now(i):
        T = max((pos["expiry"] - dates[i]) / np.timedelta64(1, "D"), 0) / 365.0
        sig = iv(spy[i], pos["strike"], T, atm_vol_at(i,T), vix[i])
        return bin_tree(spy[i], pos["strike"], T, r, sig, q, N, call=pos.get("call", False))

    def open_call(i, vix_high=22):
        # Covered call, mirroring open_put: re-level shares, pick the call strike
        # from the target delta, and flag the position so value_now prices a call.
        nonlocal cash, pos, contracts, shares
        if vix[i] > vix_high:
            return None
        frac = 1.0
        equity_now = cash + shares * spy[i]
        target = math.floor(equity_now * frac / spy[i])
        cash -= (target - shares) * spy[i]
        shares = target
        contracts = int(shares) // 100
        sig_atm = atm_vol_at(i, dte / 365.0)
        K = strike_for_delta(spy[i], dte / 365.0, r, q,
                             sig_atm, vix[i], target_delta, call=True)
        sig = iv(spy[i], K, dte / 365.0, sig_atm, vix[i])
        price = bin_tree(spy[i], K, dte / 365.0, r, sig, q, N, call=True)
        premium = price * 100 * contracts
        fee = cost_per_side * premium
        cash += premium - fee
        pos = {"call": True, "entry_price": price, "strike": K, "open_date": dates[i],
               "expiry": dates[i] + np.timedelta64(dte, "D"),
               "premium": premium, "open_fee": fee}
        return {"opened_strike": round(K, 2), "sold_px": round(price, 2), "premium_in_$": round(premium, 2)}

    def open_put(i, vix_high=19):
        nonlocal cash, pos, contracts,shares
        if vix[i]>vix_high:
            return None
        frac = 1.0
        equity_now = cash + shares*spy[i]
        target = math.floor(equity_now*frac/spy[i])
        cash -= (target-shares)*spy[i]
        shares = target
        contracts = int(shares)//100
        sig_atm = atm_vol_at(i,dte/365.0)
        K = strike_for_delta(spy[i], dte / 365.0, r, q,
                             sig_atm, vix[i], target_delta)
        sig = iv(spy[i], K, dte / 365.0, sig_atm, vix[i])
        price = bin_tree(spy[i], K, dte / 365.0, r, sig, q, N)
        premium = price * 100 * contracts
        fee = cost_per_side * premium
        cash += premium - fee
        pos = {"entry_price": price, "strike": K, "open_date": dates[i],
               "expiry": dates[i] + np.timedelta64(dte, "D"),
               "premium": premium, "open_fee": fee}
        
        return {"opened_strike":round(K,2),"sold_px":round(price,2),"premium_in_$":round(premium,2)}
        
        
    def close_position(i, reason):
        nonlocal cash, pos
        val = value_now(i)
        close_val = val * 100 * contracts
        fee = cost_per_side * close_val
        cash -= close_val + fee
        gross = pos["premium"] - close_val
        net = gross - pos["open_fee"] - fee
        trades.append({"open": pd.Timestamp(pos["open_date"]).date(),
                       "close": pd.Timestamp(dates[i]).date(), "reason": reason,
                       "entry_px": round(pos["entry_price"], 2),
                       "exit_px": round(val, 2),
                       "vix_at_close": round(vix[i], 1),
                       "gross_$": round(gross), "net_$": round(net)})
        pos = None
        return {"closed_reason":reason,"buyback_px":round(val,2),"realized_pl_$":round(net)}

    for i in range(n):

        cash *= (1 + rf_d) 
        if div[i] > 0:
            shares += (shares * div[i]) / spy[i]
        closed = opened = None
        if pos is not None:
            val = value_now(i)
            mtm_loss = (val - pos["entry_price"]) * 100 * contracts
            if mtm_loss >= loss_mult * pos["premium"]:
                closed = close_position(i, "loss_cap")
            elif val <= (1 - profit_target) * pos["entry_price"]:
                closed = close_position(i, "profit_target")
            elif (pos["expiry"] - dates[i]) / np.timedelta64(1, "D") <= 0:
                closed = close_position(i, "expiry")
        if pos is None:
            opened = open_put(i) if side == "put" else open_call(i)

        liab = value_now(i) * 100 * contracts if pos is not None else 0.0
        equity[i] = cash + shares * spy[i] - liab

        if record_daily:
            row = {"date": pd.Timestamp(dates[i]).date(),
                   "spy": round(spy[i], 2), "vix": round(vix[i], 1)}
            if pos is not None:
                cur = liab / (100 * contracts)
                row.update({
                    "strike": round(pos["strike"], 2),
                    "sold_at": round(pos["entry_price"], 2),
                    "put_val_now": round(cur, 2),
                    "unreal_pl_$": round((pos["entry_price"] - cur) * 100 * contracts),
                    "dte_left": int((pos["expiry"] - dates[i]) / np.timedelta64(1, "D")),
                })
            if closed:
                row.update(closed)
            if opened:
                row.update(opened)
            row["equity_$"] = round(equity[i])
            daily.append(row)

    eq = pd.Series(equity, index=pd.to_datetime(df["Date"]))
    cols = ["date", "spy", "vix", "strike", "sold_at", "put_val_now",
            "unreal_pl_$", "dte_left", "closed_reason", "buyback_px",
            "realized_pl_$", "opened_strike", "sold_px", "premium_in_$",
            "equity_$"]
    daily_df = (pd.DataFrame(daily).reindex(columns=cols)
                if daily else pd.DataFrame(columns=cols))
    
    return eq, pd.DataFrame(trades),daily_df, {"shares": shares, "contracts": contracts}


# ------------------------------------------------------------------ metrics
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

    return {
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
            m = metrics(eq, df)
            rc = (trades["reason"].value_counts(normalize=True)
                  if len(trades) else pd.Series(dtype=float))
            m.update({
                "n_trades":       len(trades),
                "win_rate_%":     round((trades["net_$"] > 0).mean() * 100, 1)
                                  if len(trades) else np.nan,
                "pct_loss_cap":   round(rc.get("loss_cap", 0) * 100, 1),
                "pct_profit_tgt": round(rc.get("profit_target", 0) * 100, 1),
                "pct_expiry":     round(rc.get("expiry", 0) * 100, 1),
            })
            rows[(d, t)] = m
            print(f"delta={d:.2f}  dte={t:3d}  sharpe={m['sharpe']:5.2f}  "
                  f"sortino={m['sortino']:5.2f}  mdd={m['max_drawdown_%']:6.1f}%  "
                  f"trades={m['n_trades']}")
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


def load_data(start="2010-01-01", end=None):
    """Daily SPY, VIX and VIX9D closes and SPY dividends from Yahoo."""
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
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="exclusive; default is today")
    ap.add_argument("--side", default="put", choices=["put", "call"])
    ap.add_argument("--delta", type=float, default=0.30, help="target delta of the detailed run")
    ap.add_argument("--dte", type=int, default=14, help="days to expiry of the detailed run")
    ap.add_argument("--atm-scale", type=float, default=ATM_SCALE)
    ap.add_argument("--skew-base", type=float, default=SKEW_BASE)
    a = ap.parse_args()
    ATM_SCALE, SKEW_BASE = a.atm_scale, a.skew_base

    DELTAS = [0.15, 0.20, 0.30]                # |delta| of the option sold
    DTES = [14, 21, 30, 45]                    # calendar days

    df = load_data(a.start, a.end)
    res = run_sweep(df, DELTAS, DTES, side=a.side)
    eq, trades, daily, info = backtest(df, dte=a.dte, target_delta=a.delta, record_daily=True, side=a.side)
    m = metrics(eq, df)
    m["n_trades"] = len(trades)
    print(f"\ndetailed run: delta {a.delta}, {a.dte} days: " + ", ".join(f"{k} {v}" for k, v in m.items()))
    save_outputs(res, trades, daily, m)
