# Derivatives Pricer and Back Tester

I built this to price American options with a binomial tree and to back test a simple idea: hold SPY, and sell an out-of-the-money put against it every couple of weeks. It has three parts, a volatility model, the pricer and the back test, and a fourth script that checks the volatility model against real option prices. That check is the most useful thing here, because it showed the model I had been trusting was wrong in two ways.

Research code. It places no orders and is not financial advice.

## How it works

| Part | What it does |
|---|---|
| Vol model | at-the-money vol from VIX9D and VIX, interpolated in total variance between 9 and 30 days, plus a linear skew that steepens with VIX |
| Pricer | a 50-step Cox-Ross-Rubinstein tree, vectorised with numpy, with early exercise checked at every step |
| Strike picker | finds the strike with a target delta, iterating because the vol depends on the strike through the skew |
| Back test | holds SPY, sells a put (or a covered call) at a target delta and days to expiry, marks it to market every day, and closes on a 70% profit target, a loss cap of twice the premium, or expiry |

A few choices that matter:

- Total variance is what gets interpolated between VIX9D and VIX, not vol, since variance adds up over time.
- The tree's up-probability is set from the forward, so put-call parity holds in the tree to rounding error. A test checks that.
- No new put is sold when VIX is above 19, and the position is closed and re-opened on the same day's close.
- Costs are 1% of premium per side.

## Checking the pricer

`tests/test_pricer.py` needs no market data:

| Check | Result |
|---|---|
| European tree against Black-Scholes, over a 3 x 4 x 4 grid of delta, days and vol | largest error 1.03% for puts and 1.81% for calls at 50 steps, under 0.25% at 500 |
| put-call parity inside the tree | holds to 1e-9 |
| American call with no dividend | equal to the European call |
| deep in-the-money call with an 8% dividend yield | exercised early, worth intrinsic |
| strike picker | lands within 0.002 of the target delta for puts and 0.01 for calls |

## Back test, 2010 to November 2026

Sharpe ratio by the delta of the put sold and its days to expiry, from $500,000:

| Delta | 14 days | 21 days | 30 days | 45 days |
|---|---|---|---|---|
| 0.15 | 0.69 | 0.67 | 0.64 | 0.60 |
| 0.20 | 0.73 | 0.71 | 0.64 | 0.63 |
| 0.30 | 0.83 | 0.72 | 0.72 | 0.65 |

Looking at this table, the conclusion I first drew was that selling closer to the money and shorter-dated is better, with the 0.30-delta, 14-day put on top. That conclusion turned out to come from the pricing model, not from the market.

## Is the pricing model right?

Every option price above comes from the vol model. None is a quote. `validate_surface.py` takes one real SPY chain (the close on September 24, 2026, with SPY at 767.29, VIX9D 14.11 and VIX 15.67) and compares the model with the 782 put quotes between 7 and 52 days. The model is wrong in two ways that partly cancel:

**The level is too high.** The model treats VIX as an at-the-money vol. Real at-the-money vol was 11.2% to 12.8% across those expiries, against the model's 14.1% to 15.7%, so the model was about 3.4 points too high on every expiry. This makes sense once you remember what VIX is: it prices a strip of options across all strikes, including the expensive downside puts, so in a skewed market it sits above at-the-money vol.

**The skew is too flat.** The real slope of vol against log-strike was 1.31 (0.87 to 1.55 by expiry). The model assumed 0.24.

For the puts the back test sells, the net effect depends on the strike:

| Days | 0.15 delta | 0.20 delta | 0.30 delta |
|---|---|---|---|
| 14 | +8% | +29% | +36% |
| 21 | -2% | +23% | +31% |
| 30 | -10% | +14% | +27% |

(Model price against the real mid. Positive means the model charged more premium than the market was paying.)

Near the money the too-high level dominates and the model overcharges by a third. Further out the too-flat skew pulls the other way and the two errors nearly cancel. So the back test was collecting premium that wasn't there on exactly the puts it ranked best.

## The back test with the model matched to the real chain

`pricer.py --atm-scale 0.77 --skew-base 1.67` reruns everything with the level scaled to 77% of the VIX-based number and the skew set to the measured slope:

| Delta | 14 days | 21 days | 30 days | 45 days |
|---|---|---|---|---|
| 0.15 | 0.75 | 0.78 | 0.79 | 0.76 |
| 0.20 | 0.75 | 0.76 | 0.76 | 0.74 |
| 0.30 | 0.71 | 0.72 | 0.70 | 0.70 |

A few things stand out:

- The ranking is gone. Every cell is between 0.70 and 0.79, and the 0.30-delta, 14-day put falls from 0.83 to 0.71. "Sell closer to the money" was the flat skew talking.
- The overall result holds up. Selling puts against a share position still gives a Sharpe around 0.75, so the idea survives a more realistic price.
- For the 0.15-delta, 14-day put, the alpha against SPY is 1.6% a year with a beta of 1.08, and the worst drawdown is 36%. The short put adds to the share position, it does not hedge it.

## Limitations

1. One chain, one day. The two corrections come from a single calm session (VIX 15.7) and are applied to sixteen years. The skew steepens and the gap between VIX and at-the-money vol widens when markets are stressed, so this is a first correction, not a calibration. The real fix is a history of daily chains.
2. No margin. The puts are sold on top of a fully invested share position, which a real account would have to margin. The beta above 1 is that leverage.
3. Daily closes. The profit target and the loss cap are checked once a day, so a gap through the loss cap is closed at the next close, however far it has gone.
4. No assignment. An early-exercised short put is not modelled, although the price paid to close it is the American price.
5. A fixed 3% rate and 1.2% dividend yield for the whole period.
6. One strategy, one underlying, and the settings (70% profit target, twice-premium loss cap, VIX ceiling of 19) were not chosen out of sample.

## Running it

```bash
pip install -r requirements.txt
python tests/test_pricer.py
python pricer.py                                    # 2010 to today, original vol model
python pricer.py --atm-scale 0.77 --skew-base 1.67  # vol model matched to the real chain
python validate_surface.py cache/SPY_2026-09-24.json 2026-09-24 767.29
```

`pricer.py` downloads SPY, VIX and VIX9D from Yahoo and writes the sweep, the trades and a daily blotter to `out/`. `validate_surface.py` needs an option chain saved as an Alpaca snapshot, which isn't included here. Its output table is in `results/`.

| File | What it is |
|---|---|
| `pricer.py` | vol model, binomial tree, strike picker, back test and sweep |
| `validate_surface.py` | model prices against a real SPY chain |
| `tests/test_pricer.py` | tree against Black-Scholes, early exercise, strike picker |
| `results/` | the two sweeps and the validation table |

Tested with Python 3.13.
