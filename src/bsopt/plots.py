from __future__ import annotations
import os
import matplotlib.pyplot as plt
import numpy as np
from .black_scholes import OptionParams, price, greeks

def _ensure_outdir():
    os.makedirs("outputs", exist_ok=True)

def price_vs_S(K=100, r=0.05, sigma=0.2, T=1.0, kind="call", Smin=20, Smax=180, n=200):
    _ensure_outdir()
    S_grid = np.linspace(Smin, Smax, n)
    vals = [price(OptionParams(S, K, r, sigma, T, kind)) for S in S_grid]
    plt.figure(); plt.plot(S_grid, vals)
    plt.title(f"Price vs Spot (K={K}, r={r}, sigma={sigma}, T={T}, kind={kind})")
    plt.xlabel("Spot S"); plt.ylabel("Option Price")
    out = f"outputs/price_vs_S_{kind}.png"; plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out

def price_vs_sigma(S=100, K=100, r=0.05, T=1.0, kind="call", smin=0.05, smax=0.6, n=200):
    _ensure_outdir()
    sig_grid = np.linspace(smin, smax, n)
    vals = [price(OptionParams(S, K, r, sigma, T, kind)) for sigma in sig_grid]
    plt.figure(); plt.plot(sig_grid, vals)
    plt.title(f"Price vs Volatility (S={S}, K={K}, r={r}, T={T}, kind={kind})")
    plt.xlabel("Volatility sigma"); plt.ylabel("Option Price")
    out = f"outputs/price_vs_sigma_{kind}.png"; plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out

def price_vs_T(S=100, K=100, r=0.05, sigma=0.2, kind="call", Tmin=1/365, Tmax=2.0, n=200):
    _ensure_outdir()
    T_grid = np.linspace(Tmin, Tmax, n)
    vals = [price(OptionParams(S, K, r, sigma, T, kind)) for T in T_grid]
    plt.figure(); plt.plot(T_grid, vals)
    plt.title(f"Price vs Time to Expiry (S={S}, K={K}, r={r}, sigma={sigma}, kind={kind})")
    plt.xlabel("Time to Expiry (years)"); plt.ylabel("Option Price")
    out = f"outputs/price_vs_T_{kind}.png"; plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out

def greek_vs_S(greek="delta", K=100, r=0.05, sigma=0.2, T=1.0, kind="call", Smin=20, Smax=180, n=200):
    _ensure_outdir()
    S_grid = np.linspace(Smin, Smax, n)
    vals = [greeks(OptionParams(S, K, r, sigma, T, kind))[greek] for S in S_grid]
    plt.figure(); plt.plot(S_grid, vals)
    plt.title(f"{greek.title()} vs Spot (K={K}, r={r}, sigma={sigma}, T={T}, kind={kind})")
    plt.xlabel("Spot S"); plt.ylabel(greek.title())
    out = f"outputs/{greek}_vs_S_{kind}.png"; plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out

def smile_from_bs(S=100, r=0.02, T=1.0, sigma_true=0.20, kind="call", kmin=60, kmax=140, n=25):
    """Create synthetic prices at sigma_true, invert to implied vol, and plot IV vs K."""
    from .black_scholes import OptionParams, price as bs_price
    from .iv import implied_vol
    _ensure_outdir()
    K_grid = np.linspace(kmin, kmax, n)
    prices = [bs_price(OptionParams(S, K, r, sigma_true, T, kind)) for K in K_grid]
    ivs = [implied_vol(kind, S, K, r, T, p) for K,p in zip(K_grid, prices)]
    plt.figure(); plt.plot(K_grid, ivs)
    plt.title(f"Implied Vol Smile (synthetic) — S={S}, r={r}, T={T}, kind={kind}")
    plt.xlabel("Strike K"); plt.ylabel("Implied Volatility")
    out = f"outputs/smile_{kind}.png"; plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out
