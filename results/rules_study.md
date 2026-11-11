# Rules study

2010-01-04 to 2026-11-11, split at 2018-01-01. Model: {'tree': 'lr', 'steps': 201, 'smile': 'fitted', 'atm_scale': 0.812}.

## One change at a time

| variant | median_2010_17 | median_2018_26 | median_full | worst_dd | median_trades | median_win | kept |
|---|---|---|---|---|---|---|---|
| base | 0.786 | 0.537 | 0.639 | -37.7 | 372.0 | 0.812 | False |
| E1 no VIX cap | 0.688 | 0.532 | 0.592 | -50.1 | 538.5 | 0.821 | False |
| E2 implied above realised | 0.719 | 0.599 | 0.645 | -45.7 | 367.5 | 0.825 | False |
| E3 skip backwardation | 0.747 | 0.651 | 0.694 | -37.7 | 454.5 | 0.836 | False |
| X1 50% profit target | 0.74 | 0.54 | 0.63 | -37.1 | 514.0 | 0.847 | False |
| X2 time exit at 1/3 left | 0.779 | 0.544 | 0.642 | -37.6 | 395.0 | 0.792 | False |
| X3 delta stop at 0.50 | 0.78 | 0.54 | 0.643 | -36.9 | 395.5 | 0.827 | False |

Kept (median Sharpe up in both halves): none

## Combined

Final rules: `{'entry': 'vix_cap', 'profit_target': 0.7, 'time_exit': None, 'stop': 'loss', 'collateral': 'overlay'}`

| variant | median_2010_17 | median_2018_26 | median_full | worst_dd | median_trades | median_win |
|---|---|---|---|---|---|---|
| base | 0.786 | 0.537 | 0.639 | -37.7 | 372.0 | 0.812 |
| final | 0.786 | 0.537 | 0.639 | -37.7 | 372.0 | 0.812 |
| base, cash-secured | 0.69 | -0.188 | 0.151 | -12.6 | 372.0 | 0.812 |
| final, cash-secured | 0.69 | -0.188 | 0.151 | -12.6 | 372.0 | 0.812 |

### Sharpe by setting, final, full period

| delta | 14 | 21 | 30 | 45 |
|---|---|---|---|---|
| 0.15 | 0.65 | 0.65 | 0.62 | 0.62 |
| 0.2 | 0.64 | 0.66 | 0.63 | 0.61 |
| 0.3 | 0.66 | 0.65 | 0.64 | 0.61 |

### Max drawdown %, final

| delta | 14 | 21 | 30 | 45 |
|---|---|---|---|---|
| 0.15 | -35.9 | -35.9 | -35.7 | -34.8 |
| 0.2 | -36.2 | -36.3 | -36.1 | -35.1 |
| 0.3 | -36.9 | -36.9 | -36.7 | -37.7 |

### Sharpe by setting, final, cash-secured, full period

| delta | 14 | 21 | 30 | 45 |
|---|---|---|---|---|
| 0.15 | 0.15 | 0.16 | -0.04 | -0.02 |
| 0.2 | 0.15 | 0.27 | 0.11 | 0.01 |
| 0.3 | 0.3 | 0.29 | 0.25 | 0.15 |

### Max drawdown %, final, cash-secured

| delta | 14 | 21 | 30 | 45 |
|---|---|---|---|---|
| 0.15 | -7.9 | -7.6 | -8.3 | -6.3 |
| 0.2 | -10.7 | -9.3 | -9.7 | -8.8 |
| 0.3 | -12.4 | -12.6 | -12.6 | -11.6 |

## Benchmarks over the same dates

| | Sharpe 2010-17 | Sharpe 2018-26 | Sharpe full | total return % | max drawdown % |
|---|---|---|---|---|---|
| SPY | 0.74 | 0.63 | 0.67 | 788.86 | -33.72 |
| ^PUT | 0.53 | 0.34 | 0.39 | 275.34 | -30.04 |
