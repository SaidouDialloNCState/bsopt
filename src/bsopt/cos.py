from __future__ import annotations
import math
import numpy as np

def _truncation_from_cumulants(c1: float, c2: float, c4: float, L: float = 10.0):
    a = c1 - L * math.sqrt(abs(c2) + math.sqrt(abs(c4)))
    b = c1 + L * math.sqrt(abs(c2) + math.sqrt(abs(c4)))
    return a, b

def _chi_psi(a: float, b: float, c: float, d: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return chi_k, psi_k for interval [c,d] on [a,b]."""
    k = np.arange(N, dtype=float)
    u = k * math.pi / (b - a)                 # shape (N,)
    chi = np.zeros_like(u)
    psi = np.zeros_like(u)

    # k >= 1
    mask = k > 0
    u_m = u[mask]
    ca, cb = (c - a), (d - a)
    num_chi = (
        np.cos(u_m * cb) * np.exp(d) - np.cos(u_m * ca) * np.exp(c)
        + u_m * (np.sin(u_m * cb) * np.exp(d) - np.sin(u_m * ca) * np.exp(c))
    )
    chi[mask] = num_chi / (1.0 + u_m * u_m)
    psi[mask] = (np.sin(u_m * cb) - np.sin(u_m * ca)) / u_m

    # k = 0 limits
    chi[0] = np.exp(d) - np.exp(c)
    psi[0] = d - c
    return chi, psi

def cos_price(cf, S0: float, r: float, T: float, K: float, kind: str,
              N: int = 256, L: float = 10.0, cumulants: tuple[float,float,float] | None = None) -> float:
    """
    Fang–Oosterlee COS pricing for European options.
    cf(u): E[e^{i u X_T}] with X_T = log S_T.
    """
    x0 = math.log(S0)
    if cumulants:
        a, b = _truncation_from_cumulants(*cumulants, L=L)
    else:
        a, b = x0 - L, x0 + L

    k = np.arange(N, dtype=float)
    u = k * math.pi / (b - a)

    # payoff coefficients U_k
    lnK = math.log(K)
    if kind == "call":
        chi, psi = _chi_psi(a, b, lnK, b, N)             # integrate on [lnK, b]
        Uk = (chi - K * psi)
    else:
        chi, psi = _chi_psi(a, b, a, lnK, N)             # integrate on [a, lnK]
        Uk = (K * psi - chi)

    # 2/(b-a) factor and k=0 half-weight
    Uk *= 2.0 / (b - a)
    Uk[0] *= 0.5

    # density cosine coefficients: Re[ cf(u) * e^{-i u a} ]
    phi = cf(u) * np.exp(-1j * u * a)
    Re_phi = np.real(phi)

    disc = math.exp(-r * T)
    price = disc * np.sum(Re_phi * Uk)
    return float(price)
