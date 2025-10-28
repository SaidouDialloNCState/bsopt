# Black-Scholes (bsopt)

A small quant lab:
- Closed-form Black–Scholes (calls/puts) with Greeks (Delta, Gamma, Vega; plus Theta/Rho).
- Monte Carlo (risk-neutral) with antithetic variates and bump-and-revalue Greeks.
- Cox–Ross–Rubinstein binomial tree (European by default).
- Plots vs spot (S), volatility (σ), and time to expiry (T).
- CLI tool: `bsopt`.

## Dev quickstart
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
pre-commit install
pytest -q


```

## CLI quick reference

```bash
# Closed-form
bsopt price  --kind call --S 100 --K 100 --r 0.05 --sigma 0.20 --T 1
bsopt greeks --kind call --S 100 --K 100 --r 0.05 --sigma 0.20 --T 1

# Monte Carlo / Binomial
bsopt mc       --kind call --S 100 --K 100 --r 0.05 --sigma 0.20 --T 1 --paths 200000 --seed 7
bsopt binomial --kind call --S 100 --K 100 --r 0.05 --sigma 0.20 --T 1 --steps 800

# Implied vol and synthetic smile
bsopt iv     --kind call --S 100 --K 100 --r 0.05 --T 1 --price 10.450584
bsopt smile  --kind call --S 100 --r 0.02 --T 1 --sigma_true 0.25

# Monte Carlo Greeks (Pathwise + Likelihood-Ratio)
bsopt mcgreeks --kind call --S 100 --K 100 --r 0.05 --sigma 0.20 --T 1 --paths 200000 --seed 7
