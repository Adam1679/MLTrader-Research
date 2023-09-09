"""逐笔成交的因子


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
    htan,
)


class factor_last_buy_value_ratio(factor_template):

    factor_name = "trade.buy_power.period"

    params = OrderedDict([("period", np.power(2, range(5, 10)))])

    def formula(self, data, period):
        quote_qty = data["quantity"] * data["price"]
        quote_qty_buy = pd.Series(np.where(data["is_buyer_maker"], quote_qty, 0), index=data.index)
        quote_qty_sell = pd.Series(np.where(data["is_buyer_maker"], 0, quote_qty), index=data.index)
        cum_buy = cum(quote_qty_buy, period)
        cum_sell = cum(quote_qty_sell, period)
        buy_ratio = cum_buy / (cum_sell + cum_buy)
        return buy_ratio.values
    
# directional
# range [-1, 1]
class factor_last_buy_count_ratio(factor_template):

    factor_name = "trade.buy_power_count.period"

    params = OrderedDict([("period", np.power(2, range(10, 15)))])

    def formula(self, data, period):
        quote_qty_buy = pd.Series(np.where(data["is_buyer_maker"], np.ones_like(data["is_buyer_maker"]), np.zeros_like(data["is_buyer_maker"])), index=data.index)
        quote_qty_sell = pd.Series(np.where(data["is_buyer_maker"], np.zeros_like(data["is_buyer_maker"]), np.ones_like(data["is_buyer_maker"])), index=data.index)
        cum_buy = cum(quote_qty_buy, period)
        cum_sell = cum(quote_qty_sell, period)
        buy_ratio = 2 * cum_buy / (cum_sell + cum_buy) - 1
        return buy_ratio.values


class factor_large_trade_ratio(factor_template):

    factor_name = "trade.buy_power.period"

    params = OrderedDict([("period", np.power(2, range(10, 15)))])

    def formula(self, data, period):
        quote_qty = data["quantity"] * data["price"]
        yst_trade_mean = np.nanmean(quote_qty[~data['good']])
        yst_trade_std = np.nanstd(quote_qty[~data['good']])
        N = 1.8
        large_trade = pd.Series(np.where(quote_qty > yst_trade_mean + N * yst_trade_std, quote_qty, np.zeros_like(quote_qty)), index=data.index)
        large_buy = pd.Series(np.where(data["is_buyer_maker"] == True, large_trade, np.zeros_like(quote_qty)), index=data.index)
        large_sell = pd.Series(np.where(data["is_buyer_maker"] == False, large_trade, np.zeros_like(quote_qty)), index=data.index)
        cum_buy = cum(large_buy, period)
        cum_sell = cum(large_sell, period)
        buy_ratio = 2 * cum_buy / (cum_sell + cum_buy) - 1
        return buy_ratio.values
    
class factor_doublebuy_value_ratio(factor_template):

    factor_name = "trade.doublebuy.value.ratio.period"

    params = OrderedDict([("period", np.power(2, range(10, 15)))])

    def formula(self, data, period):
        quote_qty = data["quantity"] * data["price"]
        quote_qty_buy = pd.Series(np.where(data["is_buyer_maker"], quote_qty, 0), index=data.index)
        quote_qty_sell = pd.Series(np.where(data["is_buyer_maker"], 0, quote_qty), index=data.index)
        buy_diff = ewma(quote_qty_buy, period / 10) - ewma(quote_qty_buy, period)
        sell_diff = ewma(quote_qty_sell, period / 10) - ewma(quote_qty_sell, period)
        trade_mean = ewma(quote_qty, period)
        ratio = (buy_diff - sell_diff) / trade_mean
        return np.asarray(ratio)
    
class factor_last_buy_value_ratio_cross_more_positions(factor_template):

    factor_name = "trade.buy_power.more_position.12.period"

    params = OrderedDict([("period", np.power(2, range(10, 15)))])

    def formula(self, data, period):
        quote_qty = data["quantity"] * data["price"]
        quote_qty_buy = pd.Series(np.where(data["is_buyer_maker"], quote_qty, 0), index=data.index)
        quote_qty_sell = pd.Series(np.where(data["is_buyer_maker"], 0, quote_qty), index=data.index)
        cum_buy = cum(quote_qty_buy, period)
        cum_sell = cum(quote_qty_sell, period)
        buy_ratio = cum_buy / (cum_sell + cum_buy)
        buy_ratio_with_state = buy_ratio * data["more_position.12"]
        return np.asarray(buy_ratio_with_state)