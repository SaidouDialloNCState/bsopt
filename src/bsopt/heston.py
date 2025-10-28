from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Literal, Iterable
from scipy.optimize import least_squares
from .cos import cos_price

Kind = Literal["call","put"]

@dataclass(frozen=True)
class HestonParams:
    kappa: float   # mean reversion
    theta: float   # long-run var
    sigma: float   # vol of vol
    rho: float     # corr
    v0: float      # initial var

def cf_heston(u: np.ndarray, S0: float, r: float, T: float, p: HestonParams) -> np.ndarray:
    # Characteristic function of log S_T under risk-neutral
    kappa, theta, sigma, rho, v0 = p.kappa, p.theta, p.sigma, p.rho, p.v0
    iu = 1j * u
    d = np.sqrt((rho*sigma*iu - kappa)**2 + (sigma*sigma)*(iu + u*u))
    g = (kappa - rho*sigma*iu - d) / (kappa - rho*sigma*iu + d)
    expdT = np.exp(-d*T)
    C = (r * iu * T + (kappa*theta / (sigma*sigma)) * ((kappa - rho*sigma*iu - d)*T - 2.0*np.log((1 - g*expdT)/(1 - g))))
    D = ((kappa - rho*sigma*iu - d) / (sigma*sigma)) * ((1 - expdT)/(1 - g*expdT))
    return np.exp(C + D * v0 + iu * np.log(S0))

def price_heston_cos(S: float, K: float, r: float, T: float, kind: Kind, p: HestonParams, N: int = 512, L: float = 10.0) -> float:
    cf = lambda u: cf_heston(u, S, r, T, p)
    # crude cumulants for trunc box (c1, c2, c4) — use BS-like surrogates
    c1 = math.log(S) + (r - 0.5 * p.theta) * T
    c2 = p.theta * T
    c4 = 3.0 * (p.theta**2) * (T**2)
    return cos_price(cf, S, r, T, K, kind, N=N, L=L, cumulants=(c1, c2, c4))

def calibrate_heston(S: float, r: float, T: float, K: Iterable[float], iv_target: Iterable[float], kind: Kind = "call",
                     guess: HestonParams | None = None) -> HestonParams:
    Ks = np.asarray(list(K), float)
    ivs = np.asarray(list(iv_target), float)
    if guess is None: guess = HestonParams(2.0, 0.04, 0.5, -0.5, 0.04)
    def resid(x):
        p = HestonParams(*x)
        # price -> implied vol via Black is costly; compare prices against Black using target ivs
        from scipy.stats import norm
        disc = math.exp(-r*T)
        F = S / disc
        target_prices = np.array([disc * (F*norm.cdf((math.log(F/k)+0.5*ivs[i]*ivs[i]*T)/(ivs[i]*math.sqrt(T))) - k*norm.cdf((math.log(F/k)-0.5*ivs[i]*ivs[i]*T)/(ivs[i]*math.sqrt(T)))) for i,k in enumerate(Ks)])
        mod = np.array([price_heston_cos(S, k, r, T, kind, p, N=512) for k in Ks])
        return mod - target_prices
    lb = np.array([1e-6, 1e-6, 1e-6, -0.999, 1e-6])
    ub = np.array([20.0, 2.0, 5.0, 0.999, 2.0])
    x0 = np.array([guess.kappa, guess.theta, guess.sigma, guess.rho, guess.v0])
    sol = least_squares(resid, x0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
    return HestonParams(*map(float, sol.x))
