from __future__ import annotations
import click
from .black_scholes import OptionParams, price as bs_price, greeks as bs_greeks
from .binomial import price_binomial
from .monte_carlo import (
    mc_price,
    mc_price_advanced,
    mc_greeks_bump,
    mc_greeks_pathwise_lr,
    mc_price_variants,
)
from . import plots

@click.group()
def main():
    """bsopt: Black-Scholes, Greeks, MC (VR/Pathwise/LR), Binomial, PDE, SABR/Heston, and plots."""

# -------- Basic BS --------
@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
def price(kind, s, k, r, sigma, t):
    p = OptionParams(s, k, r, sigma, t, kind)
    click.echo(f"Price: {bs_price(p):.6f}")

@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
def greeks(kind, s, k, r, sigma, t):
    p = OptionParams(s, k, r, sigma, t, kind)
    for name, val in bs_greeks(p).items():
        click.echo(f"{name}: {val:.6f}")

# -------- MC pricing & Greeks --------
@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--paths", type=int, default=100000)
@click.option("--seed", type=int, default=42)
def mc(kind, s, k, r, sigma, t, paths, seed):
    p = OptionParams(s, k, r, sigma, t, kind)
    est, se = mc_price(p, paths=paths, seed=seed)
    click.echo(f"MC price: {est:.6f}  (stderr ~ {se:.6f})")

@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--paths", type=int, default=200000)
@click.option("--seed", type=int, default=123)
def mcgreeks(kind, s, k, r, sigma, t, paths, seed):
    p = OptionParams(s, k, r, sigma, t, kind)
    res = mc_greeks_pathwise_lr(p, paths=paths, seed=seed, antithetic=True)
    for name, (val, se) in res.items():
        click.echo(f"{name}: {val:.6f}  (stderr ~ {se:.6f})")

@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--paths", type=int, default=100000)
@click.option("--seed", type=int, default=7)
@click.option("--qmc", type=click.Choice(["none","sobol"]), default="none")
@click.option("--control-variate/--no-control-variate", default=False)
def mcadv(kind, s, k, r, sigma, t, paths, seed, qmc, control_variate):
    p = OptionParams(s, k, r, sigma, t, kind)
    est, se = mc_price_advanced(p, paths=paths, seed=seed, qmc=(None if qmc=="none" else qmc), control_variate=control_variate)
    click.echo(f"MC(adv) price: {est:.6f}  (stderr ~ {se:.6f})")

@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--paths", type=int, default=100000)
@click.option("--seed", type=int, default=7)
@click.option("--stratified/--no-stratified", default=False)
@click.option("--moment-match/--no-moment-match", default=False)
def mcvr(kind, s, k, r, sigma, t, paths, seed, stratified, moment_match):
    p = OptionParams(s, k, r, sigma, t, kind)
    est, se = mc_price_variants(p, paths=paths, seed=seed, stratified=stratified, moment_match=moment_match)
    click.echo(f"MC (VR) price: {est:.6f}  (stderr ~ {se:.6f})")

# -------- Binomial --------
@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--steps", type=int, default=300)
@click.option("--american/--european", default=False)
def binomial(kind, s, k, r, sigma, t, steps, american):
    from .binomial import price_binomial
    val = price_binomial(OptionParams(s, k, r, sigma, t, kind), steps=steps, american=american)
    typ = "American" if american else "European"
    click.echo(f"Binomial (CRR, {typ}) price: {val:.6f}")

# -------- IV + Smile --------
@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--price", "target_price", type=float, required=True)
def iv(kind, s, k, r, t, target_price):
    from .iv import implied_vol
    sigma = implied_vol(kind, s, k, r, t, target_price)
    click.echo(f"Implied vol: {sigma if sigma==sigma else float('nan'):.8f}")

@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--S", "s", type=float, default=100.0)
@click.option("--r", type=float, default=0.02)
@click.option("--T", "t", type=float, default=1.0)
@click.option("--sigma_true", type=float, default=0.20)
@click.option("--kmin", type=float, default=60.0)
@click.option("--kmax", type=float, default=140.0)
@click.option("--n", type=int, default=25)
def smile(kind, s, r, t, sigma_true, kmin, kmax, n):
    out = plots.smile_from_bs(S=s, r=r, T=t, sigma_true=sigma_true, kind=kind, kmin=kmin, kmax=kmax, n=n)
    click.echo(f"Saved: {out}")

# -------- SABR --------
@main.command(name="sabr_price")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--alpha", type=float, required=True)
@click.option("--beta", type=float, default=0.5)
@click.option("--rho", type=float, required=True)
@click.option("--nu", type=float, required=True)
def sabr_price_cmd(s, k, r, t, kind, alpha, beta, rho, nu):
    from .sabr import SABRParams, sabr_price_from_S
    p = SABRParams(alpha, beta, rho, nu)
    val = sabr_price_from_S(s, k, r, t, kind, p)
    click.echo(f"SABR price: {val:.6f}")

@main.command(name="sabr_calibrate")
@click.option("--F", "fwd", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--beta", type=float, default=0.5)
@click.option("--K", "ks", type=float, multiple=True, required=True)
@click.option("--iv", "ivs", type=float, multiple=True, required=True)
def sabr_calibrate_cmd(fwd, t, beta, ks, ivs):
    from .sabr import calibrate_sabr
    if len(ks) != len(ivs):
        raise click.ClickException("Must pass same number of --K and --iv")
    p = calibrate_sabr(fwd, ks, ivs, t, beta=beta)
    click.echo(f"alpha={p.alpha:.6f} beta={p.beta:.3f} rho={p.rho:.6f} nu={p.nu:.6f}")

# -------- Heston (COS) --------
@main.command(name="heston_price")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--kind", type=click.Choice(["call","put"]), default="call")
@click.option("--kappa", type=float, required=True)
@click.option("--theta", type=float, required=True)
@click.option("--sigma_v", type=float, required=True)
@click.option("--rho", type=float, required=True)
@click.option("--v0", type=float, required=True)
def heston_price_cmd(s, k, r, t, kind, kappa, theta, sigma_v, rho, v0):
    from .heston import HestonParams, price_heston_cos
    val = price_heston_cos(s, k, r, t, kind, HestonParams(kappa, theta, sigma_v, rho, v0))
    click.echo(f"Heston price: {val:.6f}")

@main.command(name="heston_calibrate")
@click.option("--S", "s", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--K", "ks", type=float, multiple=True, required=True)
@click.option("--iv", "ivs", type=float, multiple=True, required=True)
def heston_calibrate_cmd(s, r, t, ks, ivs):
    from .heston import calibrate_heston
    p = calibrate_heston(s, r, t, ks, ivs)
    click.echo(f"kappa={p.kappa:.6f} theta={p.theta:.6f} sigma={p.sigma:.6f} rho={p.rho:.6f} v0={p.v0:.6f}")

# -------- PDE CN/PSOR --------
@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="put")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--american/--european", default=False)
@click.option("--M", "m", type=int, default=200)
@click.option("--N", "n", type=int, default=400)
def pde(kind, s, k, r, sigma, t, american, m, n):
    from .pde import price_pde_cn
    val = price_pde_cn(s, k, r, sigma, t, kind, american=american, M=m, N=n)
    click.echo(f"PDE CN price: {val:.6f}")


@main.command()
@click.option("--kind", type=click.Choice(["call","put"]), default="put")
@click.option("--S", "s", type=float, required=True)
@click.option("--K", "k", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--steps", type=int, default=50)
@click.option("--paths", type=int, default=100000)
@click.option("--seed", type=int, default=42)
def lsm(kind, s, k, r, sigma, t, steps, paths, seed):
    """American option via Longstaff-Schwartz Monte Carlo."""
    from .lsm import lsm_price
    from .black_scholes import OptionParams
    val, se = lsm_price(OptionParams(s, k, r, sigma, t, kind), paths=paths, steps=steps, seed=seed)
    click.echo(f"LSM (American) price: {val:.6f}  (stderr ~ {se:.6f})")


@main.command()
@click.option("--S", "s", type=float, required=True)
@click.option("--r", type=float, required=True)
@click.option("--T", "t", type=float, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--kmin", type=float, required=True)
@click.option("--kmax", type=float, required=True)
@click.option("--n", type=int, default=200)
def density(s, r, t, sigma, kmin, kmax, n):
    """Export BS risk-neutral terminal density f_{S_T}(K) to CSV + PNG."""
    from .density import rn_density_bs
    csv_path, png_path = rn_density_bs(s, r, t, sigma, kmin, kmax, n)
    click.echo(f"Saved: {csv_path}, {png_path}")
