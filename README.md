# BSOPT – Options Research Toolbox
## Author: Saidou Diallo  
## LinkedIn: https://www.linkedin.com/in/saidoudialloquant/
### PROGRAMS USED
TABLEAU • GIT BASH • VS CODE • PLANTUML • PYTHON VENV • GITHUB • JUPYTER • PYTEST • GITHUB ACTIONS  
### LANGUAGES USED
PYTHON • BASH • MARKDOWN 
### SKILLS USED
DERIVATIVES MODELING • STOCHASTIC CALCULUS • NUMERICAL METHODS • MONTE CARLO SIMULATION • PDE SOLVERS • MODEL CALIBRATION • QUANTITATIVE SOFTWARE ENGINEERING • RESEARCH AUTOMATION • CI/CD • SCIENTIFIC PYTHON DEVELOPMENT  
## SUMMARY
BSOPT (Black-Scholes Options Toolbox) is a complete research and pricing framework for European and American derivatives, integrating analytical models, Monte Carlo engines, PDE solvers, and stochastic volatility calibration methods including SABR and Heston.  
I named it BSOPT because it began as a clean and transparent implementation of the Black–Scholes model but expanded into a modular options research system, connecting theoretical finance, numerical methods, and quantitative engineering in one unified codebase.  
This project is for quantitative researchers, traders, and financial engineers who need transparent, testable, and extensible pricing tools for derivatives research, calibration experiments, or model validation.  
BSOPT improves on common alternatives by combining analytical transparency, algorithmic depth, and production-quality engineering, enabling fast iteration, deep testing, and reproducible results for both academic and industry work.
## VALUE PROPOSITION
BSOPT provides a unified, auditable, and research-ready environment for option pricing and calibration across analytical, numerical, and stochastic models. It is designed for reproducible research, precision benchmarking, and extensible model development that supports quantitative finance workflows.
## ARCHITECTURE DIAGRAM
![BSOPT Architecture](https://raw.githubusercontent.com/SaidouDialloNCState/bsopt/main/diagrams/bsopt_abm.png)
## QUICKSTART (GIT BASH)
### STEP 1 — Clone & Set Up Environment
git clone https://github.com/SaidouDialloNCState/Black-Scholes.git
cd Black-Scholes
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
### STEP 2 — Run Test Prices
bsopt price --kind call --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1
bsopt lsm --kind put --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1 --paths 200000 --steps 50
bsopt pde --kind put --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1 --american

### STEP 3 — Calibration & Visualization
bsopt sabr_calibrate --data data/vol_surface.csv
bsopt heston_calibrate --data data/vol_surface.csv

bsopt density --S 100 --r 0.05 --T 1 --sigma 0.2 --kmin 30 --kmax 220 --n 400

# Modeling & Methods
BSOPT implements a layered architecture for derivative pricing and research.
It supports analytical (Black–Scholes), numerical (binomial, PDE), and stochastic (Monte Carlo, Heston, SABR) methods.
Monte Carlo components include pathwise and likelihood-ratio Greeks, variance reduction (antithetic, stratified, moment matching), and American option pricing via Longstaff–Schwartz regression.
PDE solvers (Crank–Nicolson) handle early exercise conditions via projection methods, while Fourier-COS transforms provide fast European option evaluation.# Engineering & Performance
The system includes robust fallback calibration, ensuring that output is always produced, even if fast kernels or transforms fail. GPU acceleration (via CuPy) and parallel batched pricing help scale calibration sweeps across large parameter grids. All components are structured into modular packages with unit tests, smoke tests, CLI tests, and timing utilities. The codebase is fully version-controlled, with separate safe scripts for reproducible research and deployment.
# Engineering & Performance
BSOPT is fully modular under src/bsopt, adhering to software-engineering best practices.
It features pytest-based validation, property-based tests for arbitrage and boundary conditions, and GitHub Actions CI/CD for reproducible verification.
The project includes type checking (mypy), formatting (black, ruff), and automated linting via pre-commit hooks.
All methods are benchmarked for numerical stability and convergence, ensuring consistent accuracy across analytical, PDE, and stochastic solvers.
# Features
BSOPT provides:
- Unified CLI + Python API for fast experimentation and reproducible pricing.
- Support for European and American options using analytical, binomial, PDE, and Monte Carlo methods.
- Advanced volatility model calibration (SABR & Heston).
- Risk-neutral density exports and volatility smile visualizations.
- Extensible design for integrating new stochastic models or AI/ML volatility surfaces.
- Automated testing, CI, and code-quality enforcement for research reproducibility.
The architecture separates core model logic, numerical engines, calibration layers, and visualization tools — ensuring clean scalability and fast prototyping for research or production.
