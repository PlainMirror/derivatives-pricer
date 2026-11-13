# Derivatives Pricer and Back Tester

An American option pricer (Leisen-Reimer binomial tree) with a volatility smile fitted to a real SPY option chain, used to back test selling SPY puts and buying SPY calls from 2010 to 2026.

Research code. It places no orders and is not financial advice.

## How it works

| Part | What it does |
|---|---|
| Pricer | 201-step Leisen-Reimer tree with early exercise at every node. Cox-Ross-Rubinstein is kept for comparison (`--legacy`) |
| Vol model | at-the-money vol from VIX9D and VIX, interpolated in total variance and scaled to a real chain's level, plus a smile in standardised moneyness, one quadratic per side of the forward |
| Strike picker | finds the strike with a target delta, iterating because the vol depends on the strike |
| Back test | sells a put (on top of SPY, or cash-secured) or buys a call at a target delta and days to expiry, marks it daily, and closes on a profit target, a stop, a time exit or expiry |

**Tree accuracy** (`convergence.py`, European puts against Black-Scholes): at 201 steps Leisen-Reimer is within 0.0005%, against 0.08% to 0.15% for Cox-Ross-Rubinstein at the same steps, and its error shrinks smoothly as steps increase.

**Smile fit** (`smile_fit.py`): fitted on alternate expiries of the SPY chain at the September 24, 2026 close and checked on the others. Error on the held-back expiries is 1.05 vol points, against 3.19 for a flat VIX-based model, and the sold puts are priced within a median 8.8% of the market.

## Results, 2010 to November 2026, $500,000

| | Total return | CAGR | Sharpe | Max drawdown |
|---|---|---|---|---|
| best put on SPY (0.20 delta, 21 days) | 896% | 14.6% | 0.66 | -36.3% |
| best call (0.70 delta, 45 days) | 656% | 12.8% | 0.63 | -29.4% |
| SPY | 801% | 13.9% | 0.68 | -33.7% |

Across 12 settings each (delta x 14, 21, 30, 45 days), puts on SPY range from Sharpe 0.61 to 0.66 and calls from 0.35 to 0.63. Rules:

- **Puts:** sold when VIX is 19 or below, closed at 70% of the premium kept or a loss of twice the premium.
- **Calls:** sized to the account's SPY exposure, closed at +100%, -50%, or a third of their life left.

Six alternative entry and exit rules were tested against a rule fixed in advance (better Sharpe in both 2010-2017 and 2018-2026), and none passed (`rules_study.py`). Full grids are in `results/`.

## Limitations

- The smile comes from one calm day (VIX 15.7) and is applied to sixteen years.
- The fitted model still prices sold puts about 8% below the market, and 0.30-delta calls about 11% below, which biases the puts down and the calls up.
- Stops and targets are checked on daily closes.
- A flat 3% rate and 1.2% dividend yield throughout.
- No margin model for puts sold on top of a full share position.

## Running it

```bash
pip install -r requirements.txt
python tests/test_pricer.py && python tests/test_lr_smile.py
python convergence.py
python smile_fit.py cache/SPY_2026-09-24.json 2026-09-24 767.29   # needs a saved chain, not included
python rules_study.py
python calls_study.py
python pricer.py                          # sweep and one detailed run
python pricer.py --collateral cash        # cash-secured puts
```

| File | What it is |
|---|---|
| `pricer.py` | trees, vol model, strike picker, back test |
| `smile_fit.py`, `validate_surface.py` | the model against a real chain |
| `rules_study.py`, `calls_study.py`, `convergence.py` | the studies above |
| `tests/` | 25 tests, no market data needed |

Tested with Python 3.13.
