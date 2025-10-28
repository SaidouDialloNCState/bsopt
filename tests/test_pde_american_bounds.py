import math
from bsopt.black_scholes import OptionParams, price as bs_price
from bsopt.pde import price_pde_cn

def test_american_put_between_euro_and_strike():
    S,K,r,sigma,T = 100,100,0.05,0.20,1.0
    euro_put = bs_price(OptionParams(S,K,r,sigma,T,"put"))
    amer_put = price_pde_cn(S,K,r,sigma,T,"put",american=True,M=160,N=320)
    assert amer_put >= euro_put - 1e-2
    assert amer_put <= K + 1e-6
    # sanity: shouldn't be outrageously large
    assert amer_put < 15.0
