from __future__ import annotations
from typing import Optional, Dict, Literal, Tuple
import math
import numpy as np
from scipy.stats import qmc, norm
from .black_scholes import OptionParams

Kind = Literal["call","put"]

def _payoff(ST: np.ndarray, K: float, kind: Kind) -> np.ndarray:
    return np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)

def _simulate_ST_Z(S: float, r: float, sigma: float, T: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    Z = rng.standard_normal(n)
    ST = S * np.exp((r - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * Z)
    return ST, Z

def _simulate_ST_QMC(S: float, r: float, sigma: float, T: float, n: int, seed: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    eng = qmc.Sobol(d=1, scramble=True, seed=seed)
    U = eng.random(n).reshape(-1)
    # antithetic pairing improves Sobol too
    U = np.concatenate([U, 1.0 - U])[:n]
    U = np.clip(U, 1e-12, 1 - 1e-12)
    Z = norm.ppf(U)
    ST = S * np.exp((r - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * Z)
    return ST, Z

def mc_price(params: OptionParams, paths: int = 100_000, antithetic: bool = True, seed: Optional[int] = None) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    # standard MC used by legacy functions
    ST, _ = _simulate_ST_Z(params.S, params.r, params.sigma, params.T, paths, rng)
    if antithetic:
        ST = np.concatenate([ST, params.S*params.S/ ST])[:paths]  # crude antithetic (ignore if odd)
    disc_pay = math.exp(-params.r * params.T) * _payoff(ST, params.K, params.kind)
    est = disc_pay.mean()
    stderr = disc_pay.std(ddof=1) / math.sqrt(len(disc_pay))
    return float(est), float(stderr)

def mc_price_advanced(params: OptionParams, paths: int = 100_000, seed: Optional[int] = None, qmc: Optional[str] = None, control_variate: bool = False) -> Tuple[float, float]:
    """MC pricing with optional Sobol QMC and control variate C = e^{-rT} S_T (E[C]=S0)."""
    S, K, r, sigma, T, kind = params.S, params.K, params.r, params.sigma, params.T, params.kind
    disc = math.exp(-r * T)
    if qmc == "sobol":
        ST, Z = _simulate_ST_QMC(S, r, sigma, T, paths, seed)
    else:
        rng = np.random.default_rng(seed)
        ST, Z = _simulate_ST_Z(S, r, sigma, T, paths, rng)

    P = _payoff(ST, K, kind)
    X = disc * P
    if control_variate:
        C = disc * ST
        mu_C = S  # E[e^{-rT} S_T] = S0 under risk-neutral
        # optimal coefficient
        b = float(np.cov(X, C, ddof=1)[0,1] / np.var(C, ddof=1))
        Y = X - b * (C - mu_C)
        est = float(Y.mean())
        stderr = float(Y.std(ddof=1) / math.sqrt(len(Y)))
    else:
        est = float(X.mean())
        stderr = float(X.std(ddof=1) / math.sqrt(len(X)))
    return est, stderr

# Existing estimators (bump and pathwise/LR) kept for Greeks
def mc_greeks_bump(params: OptionParams, paths: int = 100_000, seed: Optional[int] = 7, bump_S: float = 1e-4, bump_sigma: float = 1e-4) -> Dict[str, float]:
    base, _ = mc_price_advanced(params, paths=paths, seed=seed)
    upS = OptionParams(params.S*(1+bump_S), params.K, params.r, params.sigma, params.T, params.kind)
    dnS = OptionParams(params.S*(1-bump_S), params.K, params.r, params.sigma, params.T, params.kind)
    upP, _ = mc_price_advanced(upS, paths=paths, seed=seed)
    dnP, _ = mc_price_advanced(dnS, paths=paths, seed=seed)
    delta = (upP - dnP) / (2 * params.S * bump_S)
    gamma = (upP - 2*base + dnP) / ((params.S * bump_S)**2)
    upV = OptionParams(params.S, params.K, params.r, params.sigma*(1+bump_sigma), params.T, params.kind)
    dnV = OptionParams(params.S, params.K, params.r, params.sigma*(1-bump_sigma), params.T, params.kind)
    upPV, _ = mc_price_advanced(upV, paths=paths, seed=seed)
    dnPV, _ = mc_price_advanced(dnV, paths=paths, seed=seed)
    vega = (upPV - dnPV) / (2 * params.sigma * bump_sigma)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}

def mc_greeks_pathwise_lr(params: OptionParams, paths: int = 100_000, seed: Optional[int] = 123, antithetic: bool = True) -> Dict[str, Tuple[float, float]]:
    S, K, r, sigma, T, kind = params.S, params.K, params.r, params.sigma, params.T, params.kind
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(paths)
    ST = S * np.exp((r - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * Z)
    disc = math.exp(-r * T)
    sqrtT = math.sqrt(T)
    m = math.log(S) + (r - 0.5 * sigma * sigma) * T
    y = np.log(ST) - m
    payoff = _payoff(ST, K, kind)
    I = (ST > K).astype(float) if kind == "call" else (ST < K).astype(float)
    sgn = 1.0 if kind == "call" else -1.0
    n = len(ST)
    inv_s2T = 1.0 / (sigma * sigma * T)

    delta_pw_samples = disc * sgn * I * (ST / S)
    vega_pw_samples  = disc * sgn * I * ST * (-sigma * T + sqrtT * Z)

    l1_S = y * (inv_s2T / S)
    l2_S = (1.0 - y) * (inv_s2T / (S * S))
    delta_lr_samples = disc * payoff * l1_S
    gamma_lr_samples = disc * payoff * (l1_S * l1_S + l2_S)
    vega_lr_samples  = disc * payoff * ( -1.0/sigma - y/sigma + (y*y) / (sigma**3 * T) )

    def ms(x):
        return float(x.mean()), float(x.std(ddof=1)/math.sqrt(n))

    return {
        "delta_pw": ms(delta_pw_samples),
        "vega_pw":  ms(vega_pw_samples),
        "delta_lr": ms(delta_lr_samples),
        "gamma_lr": ms(gamma_lr_samples),
        "vega_lr":  ms(vega_lr_samples),
    }

def _stratified_Z(n: int, rng: np.random.Generator) -> np.ndarray:
    U = (np.arange(n) + rng.random(n)) / n
    from scipy.stats import norm as _norm
    return _norm.ppf(np.clip(U, 1e-12, 1-1e-12))

def _moment_match(Z: np.ndarray) -> np.ndarray:
    return (Z - Z.mean()) / (Z.std(ddof=1) + 1e-12)

def mc_price_variants(params, paths=100_000, seed=None, stratified=False, moment_match=False):
    S, K, r, sigma, T, kind = params.S, params.K, params.r, params.sigma, params.T, params.kind
    rng = np.random.default_rng(seed)
    Z = _stratified_Z(paths, rng) if stratified else rng.standard_normal(paths)
    if moment_match:
        Z = _moment_match(Z)
    ST = S * np.exp((r - 0.5*sigma*sigma)*T + sigma*math.sqrt(T)*Z)
    disc = math.exp(-r*T)
    X = disc * _payoff(ST, K, kind)
    return float(X.mean()), float(X.std(ddof=1)/math.sqrt(len(X)))
