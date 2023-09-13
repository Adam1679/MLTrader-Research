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


class foctor_range_pos_period(factor_template):
    ## any price is between minimum and maximum
    ## so if we use (price-min)/(max-min), the result is between [0,1]
    ## 0 is min, 1 is max
    ## then we subtract 0.5 from it
    ## then result is between -0.5 to 0.5
    ## and finally use ewma to take the average result over a range
    factor_name = "range.pos.{period}"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        return (
            get_range_pos(
                data["wpr"],
                data["min." + str(period)],
                data["max." + str(period)],
                period,
            ).values
            * 2
        )


# class foctor_range_period(factor_template):
#     factor_name = "range.{period}"

#     params = OrderedDict([("period", np.power(2, range(5, 12)))])

#     def formula(self, data, period):
#         return data["max." + str(period)] - data["min." + str(period)]


# ----------- volume factor template ----------------
class foctor_buy_power_period(factor_template):
    # [0, 1]
    factor_name = "buy_power.{period}"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        buy = cum(data["active.buy.qty"], period)
        sell = cum(data["active.sell.qty"], period)
        buy_power = buy / (buy + sell + 0.1)
        return buy_power.values


class factor_adx_directional(factor_template):
    factor_name = "adx.direc.1024.{period}"
    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        atr = data["atr.4096"]
        up = data["max.1024"].diff(period)
        up.iloc[:period] = 0
        dn = -data["min.1024"].diff(period)
        dn.iloc[:period] = 0
        plus_dm = np.where((up > dn) & (up > 0), up, 0)
        plus_dm = pd.Series(plus_dm, index=data.index)
        _plus = ewma(plus_dm, period, adjust=True) / atr

        minus_dm = np.where((dn > up) & (dn > 0), dn, 0)
        minus_dm = pd.Series(minus_dm, index=data.index)
        _minus = ewma(minus_dm, period, adjust=True) / atr
        s = _plus + _minus
        s[s == 0] = 1
        _adx_dire = ewma((_plus - _minus) / s, period, adjust=True)
        return _adx_dire.values


class factor_adx_directionalV2(factor_template):
    factor_name = "adx.direc2.1024.{period}"
    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        atr = data["atr.4096"]
        up = data["max.1024"].diff(period)
        up.iloc[:period] = 0
        dn = -data["min.1024"].diff(period)
        dn.iloc[:period] = 0
        plus_dm = np.where((up > dn) & (up > 0), up, 0)
        plus_dm = pd.Series(plus_dm, index=data.index)
        _plus = ewma(plus_dm, period, adjust=True) / atr

        minus_dm = np.where((dn > up) & (dn > 0), dn, 0)
        minus_dm = pd.Series(minus_dm, index=data.index)
        _minus = ewma(minus_dm, period, adjust=True) / atr
        s = _plus + _minus
        s[s == 0] = 1
        _adx_dire = (_plus - _minus) / s
        _adx_dire = np.digitize(_adx_dire, [-0.5, -0.01, 0.01, 0.5]) - 2
        _adx_dire = pd.Series(_adx_dire, index=data.index)
        _adx_dire = ewma(_adx_dire, period, adjust=True)
        return _adx_dire.values


class factor_rmi(factor_template):
    # evaluate relative momentum index
    # rang from [-1, 1]
    factor_name = "rmi.{period}"
    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        s = data["wpr"].diff(period)
        s.fillna(0, inplace=True)
        up = s.copy()
        up[up <= 0] = 0
        dn = -s.copy()
        dn[dn <= 0] = 0

        up = ewma(up, period, adjust=True)
        dn = ewma(dn, period, adjust=True)
        res = (up / (dn + up) - 0.5) * 2
        return res.values


class factor_volume_advantage(factor_template):
    # evaluate the volume advantage
    # rang from [-1, 1]
    factor_name = "volume.advantage.{period}"
    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        gain = data["wpr"] - data["wpr"].shift(1)
        long_v = data["quote_qty"].copy()
        short_v = data["quote_qty"].copy()
        long_v[gain < 0] = 0
        short_v[gain > 0] = 0
        long_v_ma = ewma(long_v, period, adjust=True)
        short_v_ma = ewma(short_v, period, adjust=True)
        adv = (zero_divide(long_v_ma, short_v_ma + long_v_ma + 0.1) - 0.5) * 2
        return adv.values



class foctor_ma_diff_period(factor_template):
    factor_name = "ma.dif.10.{period}"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        res = zero_divide(
            ewma(data["wpr"], round(period / 10), adjust=True)
            - ewma(data["wpr"], period, adjust=True),
            data["wpr"],
        ).values

        res = htan(res, 0, scale=120)
        return res


class foctor_nr_period(factor_template):
    ## to calculate the normalized return
    ## 3 parts: factor_name, params, formula
    ## 2^[10:13]=(1024,2048,4096)
    ## the idea is ret/|ret|
    ## then over a period, we divided by period in numerator and denominator
    ## (ret/period)/(|ret|/period)
    ## then we use ewma(ret)/ewma(|ret|) instead of mean return
    ## because calculate ewma is faster and easier
    ## but the first period items may not be correct for ewma
    ## so we use adjust=True
    ## but since there is avdivision, actually we don't need to use adjust=True
    ## they would be the same with or withour adjust=True
    factor_name = "nr.{period}"
    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        return zero_divide(
            ewma(data["ret"], period, adjust=True),
            ewma(data["ret"].abs(), period, adjust=True),
        ).values

class foctor_ret_period(factor_template):
    ## to calculate the normalized return
    ## 3 parts: factor_name, params, formula
    ## 2^[10:13]=(1024,2048,4096)
    ## the idea is ret/|ret|
    ## then over a period, we divided by period in numerator and denominator
    ## (ret/period)/(|ret|/period)
    ## then we use ewma(ret)/ewma(|ret|) instead of mean return
    ## because calculate ewma is faster and easier
    ## but the first period items may not be correct for ewma
    ## so we use adjust=True
    ## but since there is avdivision, actually we don't need to use adjust=True
    ## they would be the same with or withour adjust=True
    factor_name = "ret.{period}"
    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        return cum(data['ret'], period)

class foctor_kdj_j_period(factor_template):
    factor_name = "kdj.j.{period}"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        return ewma(
            ewma(
                (
                    zero_divide(
                        data["wpr"] - data["min." + str(period)],
                        data["max." + str(period)] - data["min." + str(period)],
                    )
                    - 0.5
                )
                * 2,
                round(period / 5),
                adjust=True,
            ),
            round(period / 5),
            adjust=True,
        ).values


class foctor_kdj_k_period(factor_template):
    factor_name = "kdj.k.{period}"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        return ewma(
            (
                zero_divide(
                    data["wpr"] - data["min." + str(period)],
                    data["max." + str(period)] - data["min." + str(period)],
                )
                - 0.5
            )
            * 2,
            round(period / 5),
            adjust=True,
        ).values


class foctor_dbook_period(factor_template):
    ## utilize the bid and ask quantity changes
    ## but the original quantity may not be stationary, i.e. it would have very large values
    ## usually from 0 to 200
    ## but can be up to several tens of thousand
    ## so we use the change of qty's direction instead
    ## it has only -1 0 1 so it's stionary, but it's dicrerte
    ## then we add ewma as fitler to make it more continuous
    ## we hope the signals are continuous values

    factor_name = "dbook.{period}"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        diff_bid_qty = data["bid.qty"] - data["bid.qty"].shift()
        diff_bid_qty[0] = 0
        diff_ask_qty = data["ask.qty"] - data["ask.qty"].shift()
        diff_ask_qty[0] = 0
        return ewma(
            np.sign(diff_bid_qty) - np.sign(diff_ask_qty), period, adjust=True
        ).values


class foctor_order_book_speard_diff_period(factor_template):
    factor_name = "dbook.spread.diff.{period}"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        bids_qty = 0
        asks_qty = 0
        for i in range(20):
            w = 1 - i / 20
            bids_qty = bids_qty + w * data[f"bid_{i}_v"] * data[f"bid_{i}_p"]
            asks_qty = asks_qty + w * data[f"ask_{i}_v"] * data[f"ask_{i}_p"]

        volume_s = bids_qty + asks_qty
        spread = bids_qty - asks_qty
        v = zero_divide(
            ewma(spread, round(period / 10), adjust=True)
            - ewma(spread, round(period), adjust=True),
            volume_s,
        ).values
        return htan(v)


class foctor_order_book_speard_period(factor_template):
    factor_name = "dbook.spread.{period}"

    params = OrderedDict([("period", np.power(2, range(5, 12)))])

    def formula(self, data, period):
        bids_qty = 0
        asks_qty = 0
        for i in range(20):
            w = 1 - i / 20
            bids_qty = bids_qty + w * data[f"bid_{i}_v"] * data[f"bid_{i}_p"]
            asks_qty = asks_qty + w * data[f"ask_{i}_v"] * data[f"ask_{i}_p"]

        spread = (bids_qty - asks_qty) / (bids_qty + asks_qty)
        return ewma(spread, period, adjust=True)