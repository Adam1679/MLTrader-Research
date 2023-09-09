"""
------------ filter variables ---------------
Used a market filter that filter out unfavorable market states.
1) must be positve.
2) must be comparable across products.
3) 

"""

from collections import OrderedDict

import numpy as np
import pandas as pd
from research.orderbook_strategies.utils.factor_analysis import factor_template
from research.orderbook_strategies.utils.helper import (
    cum,
    ewma,
    fast_roll_var,
    fcum,
    get_range_pos,
    vanish_thre,
    zero_divide,
)


# ----------- volatility factor template ----------------
# must be positve
class foctor_atr_period(factor_template):
    factor_name = "atr.period"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        return np.maximum(
            (data["max." + str(period)] - data["min." + str(period)]) / data["wpr"],
            np.ones_like(data["wpr"]) * 1e-10,
        )


class foctor_trend_index_period(factor_template):
    ## rolling trend index
    factor_name = "trend.index.period"

    params = OrderedDict([("period", np.power(2, range(9, 12)))])

    def formula(self, data, period):
        aa = zero_divide(
            abs(data["wpr"] - data["wpr"].shift(period)),
            data["max." + str(period)] - data["min." + str(period)],
        )
        aa[0:period] = 0
        aa = vanish_thre(aa, 1)
        return aa


class factor_adx(factor_template):
    # [0, 1]
    factor_name = "adx.1024.period"
    params = OrderedDict([("period", np.power(2, range(8, 12)))])

    def formula(self, data, period):
        atr = data["atr.4096"]
        up = data["max.1024"].diff(period)
        up.iloc[:period] = 0
        dn = -data["min.1024"].diff(period)
        dn.iloc[:period] = 0
        plus_dm = np.where((up > dn) & (up > 0), up, 0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0)
        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)
        _plus = ewma(plus_dm, period, adjust=True) / atr
        _minus = ewma(minus_dm, period, adjust=True) / atr
        s = _plus + _minus
        s[s == 0] = 1
        _adx = ewma(abs(_plus - _minus) / s, period, adjust=True)
        return _adx.values


class factor_market_efficiency(factor_template):
    # [0, 1]
    factor_name = "market_efficiency.period"
    params = OrderedDict([("period", np.power(2, range(8, 12)))])

    def formula(self, data, period):
        c = data["wpr"]
        move_speed = (c - c.shift(period)).abs()
        move_speed.iloc[:period] = 1
        volatility = (c - c.shift(1)).abs().rolling(period).sum()
        volatility.iloc[:period] = 1
        return move_speed / volatility


class foctor_std_period(factor_template):
    factor_name = "std.period"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        std = np.sqrt(fast_roll_var(data["wpr"], period))
        return std / data["wpr"]
