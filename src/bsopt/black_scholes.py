from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Dict
import math
from scipy.stats import norm

Kind = Literal["call", "put"]

@dataclass(frozen=True)
class OptionParams:
    S: float
    K: float
    r: float
    sigma: float
    T: float
    kind: Kind = "call"

def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan"), float("nan")
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return d1, d2

def price(params: OptionParams) -> float:
    S, K, r, sigma, T, kind = params.S, params.K, params.r, params.sigma, params.T, params.kind
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if kind == "call" else max(0.0, K - S)
        return intrinsic * math.exp(-r * T) if T > 0 else intrinsic
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    if kind == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def greeks(params: OptionParams) -> Dict[str, float]:
    S, K, r, sigma, T, kind = params.S, params.K, params.r, params.sigma, params.T, params.kind
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        delta = 1.0 if (kind == "call" and S > K) else (-1.0 if (kind == "put" and S < K) else 0.0)
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    pdf = norm.pdf(d1)
    sqrtT = math.sqrt(T)

    if kind == "call":
        delta = norm.cdf(d1)
        theta = -(S * pdf * sigma) / (2 * sqrtT) - r * K * math.exp(-r * T) * norm.cdf(d2)
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = -(S * pdf * sigma) / (2 * sqrtT) + r * K * math.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

    gamma = pdf / (S * sigma * sqrtT)
    vega = S * pdf * sqrtT  # per 1.00 vol

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho)}
