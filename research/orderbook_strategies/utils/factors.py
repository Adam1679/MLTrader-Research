from collections import OrderedDict

import numpy as np
from research.orderbook_strategies.utils.directional_factors import *
from research.orderbook_strategies.utils.filter_factors import *
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

from research.orderbook_strategies.utils.trades_factors import *
"""
方向，位置和形态
胜率，赔率和频率
"""
# ----------- targets factor template ----------------
# will be labels for the model
class foctor_ret_period(factor_template):
    factor_name = "ret.{period}"

    params = OrderedDict([("period", np.power(2, range(7, 12)))])

    def formula(self, data, period):
        return fcum(data["ret"], period).values


class foctor_ret_period_004(factor_template):
    factor_name = "ret.{period}.004"

    params = OrderedDict([("period", np.power(2, range(7, 12)))])

    def formula(self, data, period):
        return vanish_thre(fcum(data["ret"], period), 0.041).values


class foctor_ret_period_002(factor_template):
    factor_name = "ret.{period}.002"

    params = OrderedDict([("period", np.power(2, range(7, 12)))])

    def formula(self, data, period):
        return vanish_thre(fcum(data["ret"], period), 0.021).values


class foctor_ret_period_001(factor_template):
    factor_name = "ret.{period}.001"

    params = OrderedDict([("period", np.power(2, range(7, 12)))])

    def formula(self, data, period):
        return vanish_thre(fcum(data["ret"], period), 0.011).values
