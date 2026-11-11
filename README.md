# Derivatives Pricer and Back Tester

I built this to price American options with a binomial tree and to back test a simple idea: hold SPY, and sell an out-of-the-money put against it every couple of weeks. The first version said the idea worked, with a Sharpe ratio of 0.83 for the best setting. Most of this project ended up being about whether that number was real, and the answer is that it mostly came from how I was pricing the options.

Research code. It places no orders and is not financial advice.

## How it works

| Part | What it does |
|---|---|
| Pricer | a 201-step Leisen-Reimer binomial tree with early exercise checked at every node. The original 50-step Cox-Ross-Rubinstein (CRR) tree is kept for comparison |
| Vol model | at-the-money vol from VIX9D and VIX, interpolated in total variance and scaled to the level of a real SPY chain, with a smile fitted to that chain |
| Strike picker | finds the strike with a target delta, iterating because the vol depends on the strike through the smile |
| Back test | sells a put (or a covered call) at a target delta and days to expiry, marks it to market every day, and closes it on a profit target, a stop, a time exit or expiry. Puts are sold either on top of a full SPY position (`overlay`) or secured by cash (`cash`) |

`--legacy` reruns everything with the original model (CRR at 50 steps, VIX taken as the at-the-money vol, a flat linear skew).

## Why a Leisen-Reimer tree?

A CRR tree spaces its nodes by volatility alone, so the strike lands somewhere different between two nodes every time the number of steps changes. The error swings back and forth and only shrinks slowly. Leisen-Reimer chooses the up-probabilities so that the tree's chance of finishing in the money matches N(d2) and N(d1) (through the Peizer-Pratt inversion), which centres the tree on the strike. The price then converges smoothly, at about 1/N² instead of 1/N.

`convergence.py`, European puts the back test would sell, error against Black-Scholes in %:

| Steps | CRR, 14d 0.30 delta | CRR, 30d 0.20 delta | CRR, 45d 0.15 delta | LR, 14d | LR, 30d | LR, 45d |
|---|---|---|---|---|---|---|
| 25 | -1.61 | -2.81 | +1.75 | -0.008 | +0.009 | +0.031 |
| 51 | +0.58 | -0.48 | -2.14 | -0.002 | +0.002 | +0.008 |
| 101 | +0.41 | +0.40 | -0.04 | -0.0005 | +0.0006 | +0.002 |
| 201 | -0.08 | +0.05 | +0.15 | -0.0001 | +0.0001 | +0.0005 |
| 401 | +0.07 | +0.03 | -0.33 | 0.0000 | 0.0000 | +0.0001 |

The CRR column flips sign as the steps go up, and at 401 steps it is worse on the 45-day put than at 201. The old setting of 50 CRR steps was off by up to 2.8% on exactly the puts being sold. At 201 steps Leisen-Reimer is within 0.0005%, and a pricing takes about 1.3 milliseconds, so a 16-year back test still runs in a few seconds.

## Fitting the vol model to a real chain

The original model treated VIX as the at-the-money vol and used a flat linear skew, and when I first checked it against a real SPY chain it was too high at the money and its skew was about five times too flat (`validate_surface.py`). `smile_fit.py` replaces both with numbers fitted to the SPY chain at the September 24, 2026 close (SPY 767.29, VIX9D 14.11, VIX 15.67, 1,035 out-of-the-money quotes between 7 and 52 days). To have something to check it on, the 11 expiries are split alternately: 6 for the fit and 5 held back.

- **Level.** Real at-the-money vol was 0.81 of the VIX-based number (0.80 to 0.93 on the fitting expiries, 0.78 to 0.88 on the held-back ones). This makes sense once you remember what VIX is: it prices a strip of options across all strikes, including the expensive downside puts, so in a skewed market it sits above at-the-money vol.
- **Shape.** The smile is fitted in standardised moneyness, x = ln(K/F) / (σ_atm √T), as one quadratic on each side of the forward:

  σ(x) / σ_atm = 1 − 0.153x + 0.025x² below the forward, and 1 − 0.207x + 0.102x² above it.

  Dividing by σ√T is what lets one shape serve every expiry. A strike 5% below spot is deep in the wing for a one-week option and close to the money for a two-month one, and in these units the two line up.

On the 5 held-back expiries:

| | Fitted model | Original model |
|---|---|---|
| implied vol error, all strikes (RMSE, vol points) | 1.05 | 3.19 |
| the same with both given the real at-the-money vol (shape only) | 0.86 | 4.00 |
| price of the puts the back test sells, median error | -7.7% | +13.1% |
| the same, median absolute error | 8.8% | 17.5% |

So the original model overcharged for the puts it sold by about 13%, and the fitted one undercharges by about 8%, mostly on the 30-day, 0.15-delta puts (-26%), where its put wing is still too flat. (Note: my first fit was a single quadratic across both sides. It matched the whole smile well but the flatter call side pulled it, and it priced the sold puts 15% too cheap. I split it by side after seeing that, so the check is not fully clean for that one choice.)

## Did any rule change help? (`rules_study.py`)

I wrote down six changes to the original rules before running any of them, and one rule for keeping a change: the median Sharpe across all 12 settings (put delta 0.15, 0.20, 0.30 by 14, 21, 30, 45 days) has to beat the original rules in both 2010 to 2017 and 2018 to November 2026. This is the same rule I use on my other strategies, because a change that only helps in one half is usually fitted to that half.

| Change (on its own) | Median Sharpe 2010-17 | 2018-26 | Full | Worst drawdown |
|---|---|---|---|---|
| original: sell when VIX ≤ 19, take 70% of the premium, stop at a loss of 2x the premium | 0.79 | 0.54 | 0.64 | -37.7% |
| no VIX cap | 0.69 | 0.53 | 0.59 | -50.1% |
| no cap, but sell only when implied vol is above 20-day realised vol | 0.72 | 0.60 | 0.65 | -45.7% |
| no cap, but skip days when VIX9D is above VIX (backwardation) | 0.75 | 0.65 | 0.69 | -37.7% |
| take profit at 50% instead of 70% | 0.74 | 0.54 | 0.63 | -37.1% |
| close when a third of the option's life is left | 0.78 | 0.54 | 0.64 | -37.6% |
| stop when the put's delta reaches 0.50 instead of a 2x loss | 0.78 | 0.54 | 0.64 | -36.9% |
| SPY, buy and hold | 0.74 | 0.63 | 0.67 | -33.7% |

None of them passed, so the original rules stay. The closest was skipping backwardation days: it is the best row from 2018 on (0.65 against 0.54) and over the full period, but it is worse than the original in 2010 to 2017. Under the rule I set, that is a change that happens to suit recent years, so I didn't adopt it. Removing the VIX cap entirely was the worst idea, adding 12 points to the worst drawdown, since the cap is what keeps the strategy from selling puts into a crash.

## What the strategy actually earns

With the original rules and the fitted model, selling puts on top of a full SPY position:

| Delta | 14 days | 21 days | 30 days | 45 days |
|---|---|---|---|---|
| 0.15 | 0.65 | 0.65 | 0.62 | 0.62 |
| 0.20 | 0.64 | 0.66 | 0.63 | 0.61 |
| 0.30 | 0.66 | 0.65 | 0.64 | 0.61 |

(Sharpe, 2010 to November 2026. The original model gave 0.60 to 0.83 for the same grid.)

Every setting is at or slightly below SPY's own 0.67, with a beta of 1.08 to 1.19 and an alpha against SPY of -0.2% to -1.2% a year. So once the options are priced realistically, selling puts on a share position adds a little leverage and nothing else.

Taking the shares out shows the option trade on its own. Cash-secured puts (the cash earns the 3% rate, and each put is fully covered by it):

| Delta | Sharpe, 14 days | 21 days | 30 days | 45 days |
|---|---|---|---|---|
| 0.15 | 0.15 | 0.16 | -0.04 | -0.02 |
| 0.20 | 0.15 | 0.27 | 0.11 | 0.01 |
| 0.30 | 0.30 | 0.29 | 0.25 | 0.15 |

In 2010 to 2017 these were 0.14 to 0.94. From 2018 on every one of them is negative (-0.04 to -0.41). The later period has February 2018, March 2020, 2022 and April 2025, and about 17% to 21% of trades end on the stop. (Note: the CBOE PutWrite index, which sells at-the-money puts monthly with no stop, had a Sharpe of 0.39 over the same dates when measured against the same flat 3%.)

So the "edge" in the first version was mostly the pricing model overcharging for the puts it sold. Whatever is left is the variance risk premium, and it is small and unreliable at these deltas once the losses in sell-offs are counted.

## Checking the pricer

Two test files, no market data needed (25 tests):

- `tests/test_pricer.py` pins the original model: CRR against Black-Scholes over a 3 x 4 x 4 grid (largest error 1.03% for puts at 50 steps), put-call parity inside the tree, early exercise, the strike picker, and golden prices from before any change.
- `tests/test_lr_smile.py` covers the new parts:
  - Leisen-Reimer within 0.02% of Black-Scholes at 201 steps, more accurate than CRR at 801 steps, with errors that shrink with every increase in steps;
  - American Leisen-Reimer prices agree with a 2,000-step CRR tree;
  - the smile equals the at-the-money vol at the forward, is continuous there, has the equity skew, and has the same shape in standardised moneyness at every expiry;
  - the strike picker hits its delta to 1e-4 with the fitted smile;
  - cash-secured puts never exceed the cash, and each entry and exit rule changes what trades in the way it should.

## Limitations

1. One chain, one day. Both the at-the-money scale and the smile come from a single calm session (VIX 15.7) and are applied to sixteen years. The skew steepens and the gap between VIX and at-the-money vol changes when markets are stressed, so this is a first calibration, not a history of the surface. The fix is a daily record of SPY chains.
2. The fitted model still prices the sold puts about 8% below the market (26% at 30 days and 0.15 delta). That biases the back test against the strategy, while the original model biased it in favour by about 13%, so the true answer is somewhere between the two tables above.
3. Daily closes. Stops and targets are checked once a day, so a gap through the stop is closed at the next close, however far it has gone.
4. A flat 3% rate and 1.2% dividend yield for the whole period. Real T-bill rates were close to zero for most of 2010 to 2021, which flatters the cash-secured version's return but not its Sharpe, since both use the same 3%.
5. No margin in the overlay version, where puts are written on top of a fully invested share position. That is the leverage behind the beta above 1.
6. No assignment: an early-exercised short put isn't modelled, although the price paid to close it is the American price.

## Running it

```bash
pip install -r requirements.txt
python tests/test_pricer.py
python tests/test_lr_smile.py
python convergence.py                     # Leisen-Reimer against CRR
python smile_fit.py cache/SPY_2026-09-24.json 2026-09-24 767.29
python rules_study.py                     # the rule changes, both halves (about 2 minutes on 6 cores)
python pricer.py                          # the sweep and one detailed run with the fitted model
python pricer.py --collateral cash        # cash-secured puts
python pricer.py --legacy                 # the original model
```

`pricer.py` downloads SPY, VIX and VIX9D from Yahoo once a day and writes the sweep, trades and a daily blotter to `out/`. `smile_fit.py` and `validate_surface.py` need an option chain saved as an Alpaca snapshot, which isn't included here. Their output tables are in `results/`.

| File | What it is |
|---|---|
| `pricer.py` | the trees, the vol model, the strike picker, the back test and the sweep |
| `smile_fit.py` | fits the at-the-money level and the smile to a real chain, and checks them on held-back expiries |
| `validate_surface.py` | how far the original model was from the same chain |
| `rules_study.py` | the pre-registered rule changes, scored in both halves |
| `convergence.py` | the tree comparison above |
| `tests/` | the 25 tests |
| `results/` | the original model's sweep, the smile fit, and the rules study |

Tested with Python 3.13.
