from bsopt.black_scholes import OptionParams, price as bs_price
from bsopt.iv import implied_vol

def test_implied_vol_recovers_sigma():
    S,K,r,T,sig = 100.0, 100.0, 0.05, 1.0, 0.20
    p = bs_price(OptionParams(S,K,r,sig,T,"call"))
    est = implied_vol("call", S,K,r,T,p)
    assert abs(est - sig) < 1e-6
