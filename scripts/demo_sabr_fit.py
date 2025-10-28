import numpy as np, math
from bsopt.sabr import sabr_iv_hagan, calibrate_sabr
S=100; r=0.02; T=1.0; disc=math.exp(-r*T); F=S/disc
true = dict(alpha=0.25, beta=0.5, rho=-0.3, nu=0.8)
Ks = np.array([70,80,90,100,110,120,130], float)
ivs = np.array([sabr_iv_hagan(F,k,T,**true) for k in Ks])
p = calibrate_sabr(F, Ks, ivs, T, beta=true["beta"])
print("True:", true)
print("Fit: ", dict(alpha=p.alpha, beta=p.beta, rho=p.rho, nu=p.nu))
