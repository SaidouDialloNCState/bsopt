from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def rn_density_bs(S: float, r: float, T: float, sigma: float,
                  kmin: float, kmax: float, n: int = 200,
                  outdir: str = "outputs", basename: str = "density_bs"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    K = np.linspace(kmin, kmax, n)
    mu = math.log(S) + (r - 0.5 * sigma * sigma) * T
    var = sigma * sigma * T
    std = math.sqrt(var)
    # lognormal density of S_T under RN measure
    f = np.exp(-((np.log(K) - mu) ** 2) / (2 * var)) / (K * std * math.sqrt(2 * math.pi))
    df = pd.DataFrame({"K": K, "f_ST": f})
    csv_path = out / f"{basename}.csv"
    png_path = out / f"{basename}.png"
    df.to_csv(csv_path, index=False)
    plt.figure()
    plt.plot(K, f)
    plt.xlabel("K"); plt.ylabel("Risk-neutral density f_{S_T}(K)")
    plt.title("BS risk-neutral terminal density")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    return str(csv_path), str(png_path)
