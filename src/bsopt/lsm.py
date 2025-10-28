from __future__ import annotations
import math
import numpy as np
from .black_scholes import OptionParams

def lsm_price(params: OptionParams, paths: int = 100_000, steps: int = 50, seed: int = 42):
    S0, K, r, sigma, T, kind = params.S, params.K, params.r, params.sigma, params.T, params.kind
    assert kind in ("call", "put")
    rng = np.random.default_rng(seed)
    dt = T / steps
    nudt = (r - 0.5 * sigma * sigma) * dt
    sigsdt = sigma * math.sqrt(dt)

    # simulate GBM paths
    S = np.empty((paths, steps + 1), dtype=float)
    S[:, 0] = S0
    Z = rng.standard_normal((paths, steps))
    for t in range(steps):
        S[:, t + 1] = S[:, t] * np.exp(nudt + sigsdt * Z[:, t])

    def payoff(x: np.ndarray) -> np.ndarray:
        if kind == "call":
            return np.maximum(x - K, 0.0)
        else:
            return np.maximum(K - x, 0.0)

    cash = payoff(S[:, -1])              # value at maturity
    disc = math.exp(-r * dt)

    # backward induction (t = steps-1 ... 1)
    for t in range(steps - 1, 0, -1):
        St = S[:, t]
        imm = payoff(St)
        itm = imm > 0.0
        Y = cash * disc                  # discounted continuation back to t

        if np.any(itm):
            x = St[itm] / K
            A = np.vstack([np.ones_like(x), x, x * x]).T  # basis [1, S/K, (S/K)^2]
            beta, *_ = np.linalg.lstsq(A, Y[itm], rcond=None)

            x_all = St / K
            cont = beta[0] + beta[1] * x_all + beta[2] * x_all * x_all
            cont = np.maximum(cont, 0.0)  # guard against negative continuation
        else:
            cont = np.zeros_like(imm)

        # exercise only if ITM and immediate > continuation
        exercise = itm & (imm > cont)
        cash = np.where(exercise, imm, Y)

    price = cash.mean() * disc           # discount final step to t=0
    se = cash.std(ddof=1) * disc / math.sqrt(paths)
    return float(price), float(se)
