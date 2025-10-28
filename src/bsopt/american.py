from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Literal
from .black_scholes import OptionParams

Kind = Literal["call","put"]

def lsm_price(params: OptionParams, steps: int = 50, paths: int = 100_000, seed: int | None = None, basis: str = "poly") -> float:
    """
    Longstaff–Schwartz Bermudan/American pricer for GBM under risk-neutral measure.
    Basis: simple polynomials [1, S, S^2]. Works for call/put.
    Note: Early exercise for non-dividend call is suboptimal; still supported.
    """
    S0, K, r, sigma, T, kind = params.S, params.K, params.r, params.sigma, params.T, params.kind
    rng = np.random.default_rng(seed)
    dt = T / steps
    disc = math.exp(-r * dt)
    # --- simulate GBM paths (paths x (steps+1)) ---
    Z = rng.standard_normal((paths, steps))
    incr = (r - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * Z
    S = np.empty((paths, steps + 1), dtype=float)
    S[:, 0] = S0
    np.cumsum(incr, axis=1, out=incr)
    S[:, 1:] = S0 * np.exp(incr)

    if kind == "call":
        payoff = np.maximum(S - K, 0.0)
    else:
        payoff = np.maximum(K - S, 0.0)

    # Cashflows at maturity:
    cash = payoff[:, -1].copy()

    # Work backwards t = steps-1 ... 0
    for t in range(steps - 1, -1, -1):
        Y = cash * disc  # discount one step to time t
        itm = payoff[:, t] > 0
        if np.any(itm):
            X = S[itm, t]
            # polynomial basis [1, S, S^2] (stable enough for GBM)
            A = np.stack([np.ones_like(X), X, X * X], axis=1)
            # Regress discounted continuation Y on X
            beta, *_ = np.linalg.lstsq(A, Y[itm], rcond=None)
            cont = (beta[0] + beta[1] * X + beta[2] * X * X)
            ex_now = payoff[itm, t] > cont
            # exercise now: replace cashflow by immediate payoff at t
            Y[itm] = np.where(ex_now, payoff[itm, t], Y[itm])
        cash = Y  # move one step earlier

    price = float(cash.mean())  # at t=0 already
    return price
