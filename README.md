# **bsopt — Options Research Toolbox**

**Author:** [Saidou Diallo](https://www.linkedin.com/in/saidoudialloquant/)  
**PROGRAMS USED:** Tableau · Git Bash · Visual Studio Code · GitHub Actions · Jupyter Notebook  
**LANGUAGES USED:** Python · Bash · YAML · Markdown  
**SKILLS USED:** Quantitative Finance · Derivatives Modeling · Monte Carlo Simulation · Numerical PDEs · Model Calibration · Data Visualization · Software Engineering · CI/CD  

---

## **SUMMARY**

**What I made:**  
`bsopt` (“Black-Scholes Options”) is a modern **options research and pricing toolbox**.  
It implements analytical, Monte Carlo, PDE, and transform-based pricing engines for European and American options, along with **SABR** and **Heston** model calibration.  
It includes a tested **CLI and Python API** for research reproducibility and comparative analysis across pricing methods.

**Why I named it what I did:**  
The project started as a clean, extensible implementation of the **Black–Scholes model**, but evolved into a broader framework for derivatives research — hence the name `bsopt` (short for *Black-Scholes Options*).

**Who this is for:**  
Quant researchers, traders, and developers who need a **transparent, auditable, and extensible** options pricing engine for experiments, benchmarking, or calibration tasks.

**What problem it solves:**  
Most existing libraries (like QuantLib) are **monolithic** or opaque for educational or research use.  
`bsopt` provides the **clarity, modularity, and numerical rigor** needed for experimentation, validation, and quantitative interviews.

**Why it’s better than “just using XYZ library”:**  
- 100% open, modular, and reproducible with CLI and CI integration.  
- Code and math directly traceable — ideal for research or demonstration.  
- Combines PDEs, Monte Carlo (LSM, Greeks, variance reduction), and transform methods in a single clean package.

---

## **VALUE PROPOSITION**

- A **research-grade derivatives toolkit** with analytical, numerical, and stochastic solvers unified under one consistent interface.  
- Designed for **transparency, extensibility, and reproducibility** — perfect for portfolio work, model comparison, or interview prep.

---

## **ARCHITECTURAL BLOCK DIAGRAM**

![BSOPT Architecture](https://raw.githubusercontent.com/SaidouDialloNCState/bsopt/main/diagrams/bsopt_abm.png)

---

## **QUICKSTART ON GIT BASH**

**STEP 1 — Clone and initialize**
```bash
mkdir -p ~/git/repository6
cd ~/git/repository6
git clone https://github.com/smdiallo_ncstate/Black-Scholes.git .
python -m venv .venv
source .venv/Scripts/activate
STEP 2 — Install and test

bash
Copy code
pip install -e .
pytest -q
STEP 3 — Run sample commands

bash
Copy code
bsopt price --kind call --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1
bsopt lsm   --kind put --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1 --paths 200000 --steps 50
bsopt sabr_calibrate --data data/vol_surface.csv

TECH SECTION
Methods / Models
Implements closed-form Black–Scholes, Cox–Ross–Rubinstein binomial trees, Crank–Nicolson PDE, Monte Carlo with pathwise/Likelihood-Ratio Greeks, and Longstaff–Schwartz regression (LSM) for American options.
Supports advanced model calibration for SABR and Heston volatility surfaces, with numerical stabilization and plotting tools.

Engineering
Fully modularized under src/bsopt/, type-checked and covered by pytest with CI/CD on GitHub Actions.
Includes variance reduction, benchmark tests, and boundary validation (American ≥ European ≤ Strike).
Uses pre-commit hooks (black, ruff, flake8, YAML/TOML linters) to enforce code and data hygiene.

Features

CLI + Python API (bsopt …) for reproducible research.

CSV + PNG export of risk-neutral density and volatility smiles.

Support for calibration visualization (SABR/Heston fits).

Ready for extension to local/stochastic volatility or machine-learning volatility surface modeling.
