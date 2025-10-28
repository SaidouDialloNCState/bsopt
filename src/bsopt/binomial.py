from __future__ import annotations
from typing import Literal
import math
import numpy as np
from .black_scholes import OptionParams

Kind = Literal["call","put"]

def price_binomial(params: OptionParams, steps: int = 300, american: bool = False) -> float:
    S, K, r, sigma, T, kind = params.S, params.K, params.r, params.sigma, params.T, params.kind
    dt = T / steps
    if dt <= 0:
        return max(0.0, S-K) if kind == "call" else max(0.0, K-S)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)
    j = np.arange(steps + 1)
    ST = S * (u**j) * (d**(steps - j))
    values = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    for _ in range(steps, 0, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            pass
    return float(values[0])
