import math
from bsopt.black_scholes import OptionParams, price

def test_put_call_parity():
    S,K,r,sigma,T = 100,100,0.03,0.2,1.0
    c = price(OptionParams(S,K,r,sigma,T,"call"))
    p = price(OptionParams(S,K,r,sigma,T,"put"))
    assert abs(c - p - (S - K*math.exp(-r*T))) < 1e-8

def test_monotonicity_in_sigma():
    S,K,r,T = 100,100,0.03,1.0
    p1 = price(OptionParams(S,K,r,0.10,T,"call"))
    p2 = price(OptionParams(S,K,r,0.30,T,"call"))
    assert p2 > p1
