from bsopt.black_scholes import OptionParams, price, greeks

def test_bs_call_atm_one_year():
    p = price(OptionParams(100,100,0.05,0.20,1.0,"call"))
    assert abs(p - 10.4506) < 1e-3

def test_greeks_presence():
    g = greeks(OptionParams(100,100,0.05,0.20,1.0,"call"))
    for k in ["delta","gamma","vega","theta","rho"]:
        assert k in g
