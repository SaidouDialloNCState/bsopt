from bsopt.black_scholes import OptionParams, price as bs_price
from bsopt.lsm import lsm_price

def test_lsm_american_put_ge_euro():
    S,K,r,sigma,T = 100,100,0.05,0.20,1.0
    euro = bs_price(OptionParams(S,K,r,sigma,T,"put"))
    amer, _ = lsm_price(OptionParams(S,K,r,sigma,T,"put"), paths=60_000, steps=40, seed=123)
    assert amer >= euro - 0.05
    assert amer < 7.0
