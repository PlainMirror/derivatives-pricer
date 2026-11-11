"""Leisen-Reimer against Cox-Ross-Rubinstein: pricing error against Black-Scholes by number of steps.

    python convergence.py

European puts the back test would sell (SPY 767, r 3%, q 1.2%), so the exact answer is known.
"""
import math

import pandas as pd
from scipy.special import ndtr

from pricer import crr_tree, lr_tree

S, R, Q = 767.0, 0.03, 0.012
CONTRACTS = [("30d, 0.20 delta", 735.0, 30 / 365, 0.145), ("14d, 0.30 delta", 755.0, 14 / 365, 0.13),
             ("45d, 0.15 delta", 715.0, 45 / 365, 0.16)]
STEPS = [25, 51, 101, 201, 401]


def bs_put(K, T, sig):
    d1 = (math.log(S / K) + (R - Q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-R * T) * ndtr(-d2) - S * math.exp(-Q * T) * ndtr(-d1)


rows = []
for name, K, T, sig in CONTRACTS:
    ref = bs_put(K, T, sig)
    for n in STEPS:
        rows.append({"contract": name, "steps": n,
                     "CRR error %": 100 * (crr_tree(S, K, T, R, sig, Q, n, american=False) / ref - 1),
                     "LR error %": 100 * (lr_tree(S, K, T, R, sig, Q, n, american=False) / ref - 1)})
t = pd.DataFrame(rows)
print(t.pivot(index="steps", columns="contract", values=["CRR error %", "LR error %"]).round(4).to_string())
