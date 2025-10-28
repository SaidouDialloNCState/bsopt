from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Iterable, Tuple
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares

Kind = Literal["call","put"]

def black_price(F: float, K: float, T: float, sigma: float, kind: Kind, disc: float = 1.0) -> float:
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        intrinsic = max(F - K, 0.0) if kind == "call" else max(K - F, 0.0)
        return disc * intrinsic
    v = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if kind == "call":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def sabr_iv_hagan(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    # Hagan 2002 lognormal volatility (beta in [0,1])
    if F <= 0 or K <= 0 or T <= 0: return float("nan")
    if F == K:  # ATM limit
        FB = F**(1 - beta)
        term = 1 + (((1 - beta)**2 / 24) * (alpha*alpha) / (FB*FB) + (rho * beta * nu * alpha) / (4 * FB) + ((2 - 3*rho*rho) * (nu*nu) / 24)) * T
        return (alpha / FB) * term
    z = (nu / alpha) * (F*K)**((1 - beta)/2) * math.log(F/K)
    xz = math.log((math.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho))
    FK = (F*K)**((1 - beta)/2)
    A = alpha / (FK * (1 + ((1 - beta)**2 / 24) * (math.log(F/K))**2 + ((1 - beta)**4 / 1920) * (math.log(F/K))**4))
    B = 1 + (((1 - beta)**2 / 24) * (alpha*alpha) / (FK*FK) + (rho*beta*nu*alpha) / (4*FK) + ((2 - 3*rho*rho) * (nu*nu) / 24)) * T
    return A * (z / xz) * B

@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float

def sabr_price_from_S(S: float, K: float, r: float, T: float, kind: Kind, p: SABRParams) -> float:
    disc = math.exp(-r*T)
    F = S / disc  # forward under Black (no q)
    iv = sabr_iv_hagan(F, K, T, p.alpha, p.beta, p.rho, p.nu)
    return black_price(F, K, T, iv, kind, disc)

def calibrate_sabr(F: float, Ks: Iterable[float], iv_targets: Iterable[float], T: float,
                   beta: float = 0.5, guess: Tuple[float,float,float] = (0.2, 0.0, 0.5)) -> SABRParams:
    Ks = np.asarray(list(Ks), float); iv_targets = np.asarray(list(iv_targets), float)
    def resid(x):
        alpha, rho, nu = x
        mod = np.array([sabr_iv_hagan(F, k, T, alpha, beta, rho, nu) for k in Ks])
        return mod - iv_targets
    lb = (1e-6, -0.999, 1e-6)
    ub = (5.0,   +0.999, 5.0)
    x0 = np.array(guess)
    sol = least_squares(resid, x0, bounds=(lb, ub), ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
    alpha, rho, nu = sol.x
    return SABRParams(alpha=float(alpha), beta=float(beta), rho=float(rho), nu=float(nu))
