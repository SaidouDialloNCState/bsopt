from bsopt.black_scholes import OptionParams, price as bs_price
from bsopt.heston import HestonParams, price_heston_cos

def test_cos_under_bs_limit():
    # Heston -> BS when sigma->0 and v0=theta=sigma^2 ~ const; approximately match
    S,K,r,T = 100,100,0.05,1.0
    p = HestonParams(kappa=5.0, theta=0.04, sigma=1e-6, rho=0.0, v0=0.04)
    c_cos = price_heston_cos(S,K,r,T,"call",p,N=256)
    c_bs  = bs_price(OptionParams(S,K,r,0.2,T,"call"))
    assert abs(c_cos - c_bs) < 2e-2
