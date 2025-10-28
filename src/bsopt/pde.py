from __future__ import annotations
import math
import numpy as np
from typing import Literal

Kind = Literal["call","put"]

def price_pde_cn(S: float, K: float, r: float, sigma: float, T: float, kind: Kind,
                 Smax_factor: float = 4.0, M: int = 200, N: int = 400,
                 american: bool = False, omega: float = 1.0, tol: float = 1e-8, itmax: int = 20000) -> float:

    Smax = Smax_factor * K
    dS = Smax / M
    dt = T / N
    grid_S = np.linspace(0.0, Smax, M + 1)

    # Terminal payoff
    if kind == "call":
        V = np.maximum(grid_S - K, 0.0)
    else:
        V = np.maximum(K - grid_S, 0.0)

    i = np.arange(1, M)  # 1..M-1
    alpha = 0.25 * dt * (sigma * sigma * i * i - r * i)
    beta  = -0.5 * dt * (sigma * sigma * i * i + r)
    gamma = 0.25 * dt * (sigma * sigma * i * i + r * i)

    A_diag  = 1.0 - beta
    A_lower = -alpha[1:]        # length M-2 (rows 2..M-2)
    A_upper = -gamma[:-1]       # length M-2 (rows 1..M-3)

    B_diag  = 1.0 + beta
    B_lower =  alpha[1:]        # length M-2
    B_upper =  gamma[:-1]       # length M-2

    for n in range(N):
        tau = n * dt  # time-to-maturity at this layer

        # Dirichlet boundaries at time tau
        if kind == "call":
            V0 = 0.0
            VM = Smax - K * math.exp(-r * tau)
        else:
            V0 = K if american else K * math.exp(-r * tau)  # American put: immediate exercise at S=0
            VM = 0.0

        V[0], V[-1] = V0, VM

        # Build RHS for interior nodes 1..M-1
        rhs = B_diag * V[1:M]
        rhs[1:]  += alpha[1:] * V[1:M-1]     # exclude V_M (added below)
        rhs[:-1] += gamma[:-1] * V[2:M]   # exclude V_0 (added below)
        # boundary contributions from neighbors
        rhs[0]  += alpha[0] * V[0]       # i=1 uses gamma_1 * V_0
        rhs[-1] += gamma[-1] * V[-1]     # i=M-1 uses alpha_{M-1} * V_M

        if american:
            # PSOR: enforce V >= payoff
            x = V[1:M].copy()
            payoff_now = (np.maximum(grid_S[1:M] - K, 0.0) if kind == "call"
                          else np.maximum(K - grid_S[1:M], 0.0))
            for _ in range(itmax):
                x_old = x.copy()
                # first row
                x[0] = max(payoff_now[0], (1 - omega) * x[0] + omega * (rhs[0] - A_upper[0] * x[1]) / A_diag[0])
                # interior
                for j in range(1, M - 2):
                    x[j] = max(payoff_now[j], (1 - omega) * x[j] + omega * (rhs[j]
                               - A_lower[j-1] * x[j-1] - A_upper[j] * x[j+1]) / A_diag[j])
                # last row
                x[-1] = max(payoff_now[-1], (1 - omega) * x[-1] + omega * (rhs[-1] - A_lower[-2] * x[-2]) / A_diag[-1])
                if np.linalg.norm(x - x_old, ord=np.inf) < tol:
                    break
            V[1:M] = x
        else:
            # Thomas algorithm
            c = A_upper.copy()
            d = rhs.copy()
            a = A_lower.copy()
            b = A_diag.copy()
            for j in range(1, M - 1):
                w = a[j - 1] / b[j - 1]
                b[j] -= w * c[j - 1]
                d[j] -= w * d[j - 1]
            x = np.zeros_like(d)
            x[-1] = d[-1] / b[-1]
            for j in range(M - 3, -1, -1):
                x[j] = (d[j] - c[j] * x[j + 1]) / b[j]
            V[1:M] = x

    return float(np.interp(S, grid_S, V))
