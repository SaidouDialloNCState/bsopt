from __future__ import annotations
from typing import Literal, Optional
import math
import numpy as np
from scipy.optimize import brentq
from .black_scholes import OptionParams, price as bs_price

Kind = Literal["call","put"]

def _no_arb_bounds(kind: Kind, S: float, K: float, r: float, T: float) -> tuple[float, float]:
    discK = K * math.exp(-r * T)
    if kind == "call":
        return max(0.0, S - discK), S
    else:
        return max(0.0, discK - S), discK

def implied_vol(kind: Kind, S: float, K: float, r: float, T: float, target_price: float,
                low: float = 1e-8, high: float = 5.0, xtol: float = 1e-12, rtol: float = 1e-10) -> float:
    """Return Black-Scholes implied vol for a European option. np.nan if out of bounds or no root."""
    lb, ub = _no_arb_bounds(kind, S, K, r, T)
    if not (lb - 1e-12 <= target_price <= ub + 1e-12):
        return float("nan")  # price violates no-arbitrage

    def f(sig: float) -> float:
        return bs_price(OptionParams(S, K, r, sig, T, kind)) - target_price

    # Ensure a sign change in the bracket; if not, expand 'high' up to 10.0
    f_low = f(low)
    f_high = f(high)
    tries = 0
    while f_low * f_high > 0 and high < 10.0 and tries < 10:
        high *= 1.5
        f_high = f(high)
        tries += 1
    if f_low * f_high > 0:
        return float("nan")  # no root found

    return float(brentq(f, low, high, xtol=xtol, rtol=rtol, maxiter=100))
