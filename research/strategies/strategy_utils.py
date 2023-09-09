from array import array
from asyncore import poll
from audioop import mul
from cmath import isnan
from os import popen
from symbol import pass_stmt

import numpy as np
import pandas as pd
import talib
from backtesting import Strategy
from backtesting.lib import _Array, resample_apply


class Forbidden:
    NO_TRADE = -2
    NO_LONG = 1
    NO_SHORT = -1
    ANY = 0


def shift(arr, n):
    arr = np.roll(arr, n)
    arr[:n] = np.nan
    return arr


def quantile_map(s: pd.Series, rolling_window, q):
    return s.rolling(rolling_window).quantile(quantile=q)


def get_weekly(arr):
    return arr


def get_volume_advantage(close, v, len):
    gain = close - shift(close, 1)
    long_v = v.copy()
    short_v = v.copy()
    long_v[gain <= 0] = 0
    short_v[gain >= 0] = 0
    long_v_ma = get_ma(long_v, len)
    short_v_ma = get_ma(short_v, len)
    adv = long_v_ma / (long_v_ma + short_v_ma)
    return adv


def clip(x, min=None, max=None):
    if min is not None and x < min:
        x = min
    if max is not None and x > max:
        x = max
    return x


def lowest(arr, window):
    d = pd.Series(arr).rolling(window).min().values
    return d


def highest(arr, window):
    d = pd.Series(arr).rolling(window).max().values
    return d


def RMI(arr: _Array, len, delay):
    s = arr.s
    s = s.diff(delay)
    up = s.copy()
    up[up <= 0] = 0
    dn = -s.copy()
    dn[dn <= 0] = 0

    up = (
        up.to_frame(name="up")
        .ewm(alpha=2 / (len + 1), min_periods=len)
        .mean()["up"]
        .values
    )
    dn = (
        dn.to_frame(name="dn")
        .ewm(alpha=2 / (len + 1), min_periods=len)
        .mean()["dn"]
        .values
    )
    res = 100 - 100 * dn / (dn + up)
    return res


def get_rsi(arr: _Array, len):
    s = arr.s
    s = s.diff(1)
    up = s.copy()
    up[up <= 0] = 0
    dn = -s.copy()
    dn[dn <= 0] = 0

    up = up.to_frame(name="up").ewm(alpha=1 / len, min_periods=len).mean()["up"].values
    dn = dn.to_frame(name="dn").ewm(alpha=1 / len, min_periods=len).mean()["dn"].values
    res = 100 - 100 * dn / (dn + up)
    return res


def get_vmap(self, close, volume, window):
    pass


def zigzag(array, left, right):
    res = np.full_like(array, np.nan)
    direction = np.full_like(array, 0)
    n = len(array)
    isHigh = isLow = False
    last_idx = 0
    for i in range(n):
        left_bound = max(i - left, 0)
        right_bound = min(i + right, n)
        if i + 1 < n and array[i] == array[i + 1]:
            continue
        if array[i] >= np.nanmax(array[left_bound:right_bound]):
            res[i] = array[i]
            direction[i] = 1
            if isHigh:
                offset = np.nanargmin(array[last_idx:i])
                assert offset > 0
                j = last_idx + offset
                res[j] = array[j]
                direction[j] = -1
            isHigh = True
            isLow = False
            last_idx = i

        elif array[i] <= np.nanmin(array[left_bound:right_bound]):
            res[i] = array[i]
            direction[i] = -1
            if isLow:
                offset = np.nanargmax(array[last_idx:i])
                assert offset > 0
                j = last_idx + offset
                res[j] = array[j]
                direction[j] = 1
            last_idx = i
            isLow = True
            isHigh = False

    return res, direction


def break_signal2(array, left, right):
    # 有个问题，这个指标不会更新的十分的及时
    res = np.zeros_like(array)
    zigzag_score, direction = zigzag(array, left, right)
    zigzag_score = shift(zigzag_score, right)
    direction = shift(direction, right)
    rs = []
    rs_directions = []
    last_idx = 0
    for i in range(len(array)):
        if not np.isnan(direction[i]) and direction[i] != 0:
            assert not np.isnan(zigzag_score[i])
            assert len(rs_directions) == 0 or rs_directions[-1] != direction[-1]
            if len(rs) == 4:
                rs.pop(0)
                rs_directions.pop(0)
            rs.append(zigzag_score[i])
            rs_directions.append(direction[i])
            last_idx = i

        if len(rs) < 4:
            continue
        if rs_directions[-1] > 0:
            # hlh l
            h2 = rs[-3]
            h1 = rs[-1]
            l2 = rs[-2]
            l1 = array[i]

        if rs_directions[-1] < 0:
            # lhl h
            h2 = rs[-2]
            h1 = array[i]
            l2 = rs[-3]
            l1 = rs[-1]

        if (rs_directions[-1] < 0) and h1 > h2 and l1 >= l2 and array[i - 1] <= h2:

            # 1) 新高不新低
            res[i] = 1

        elif (rs_directions[-1] > 0) and l1 < l2 and h1 <= h2 and array[i - 1] >= l2:
            # 2) 新低不新高
            res[i] = -1
        # elif rs_directions[-1] < 0 and array[i] < rs[-1] and i > last_idx+1:
        #     # 特殊 case，如果最新的是-1，然后新的 1 还没有来得及更新，价格就突破的支撑位。
        #     last_mid_high = np.nanargmax(array[last_idx: i]) + last_idx
        #     assert last_mid_high != last_idx
        #     if rs[-2] >= last_mid_high:
        #         # 2) 新低不新高
        #         res[i] = -1
        # elif rs_directions[-1] > 0 and array[i] > rs[-1] and i > last_idx+1:
        #     last_mid_low = np.nanargmin(array[last_idx: i]) + last_idx
        #     assert last_mid_low != last_idx
        #     if rs[-2] <= last_mid_low:
        #         # 2) 新低不新高
        #         res[i] = 1

    return res


def break_signal(array, left, right, window=500):
    res = np.zeros_like(array)
    # all patterns
    # 1) 新高不新低
    # 2) 新低不新高
    # 3) 新高也新低
    # 4) 不新高不新低
    for i in range(window, len(array)):
        zigzag_score, direction = zigzag(array[i - window + 1 : i + 1], left, right)
        nonzeros = np.nonzero(direction)[0]
        if len(nonzeros) >= 4:
            if direction[-1] > 0:
                # newest pivot high
                # lhlh
                l2 = zigzag_score[nonzeros[-4]]
                l1 = zigzag_score[nonzeros[-2]]
                h2 = zigzag_score[nonzeros[-3]]
                h1 = zigzag_score[nonzeros[-1]]

            elif direction[-1] < 0:
                # newest pivot low
                # hlhl
                h2 = zigzag_score[nonzeros[-4]]
                h1 = zigzag_score[nonzeros[-2]]
                l2 = zigzag_score[nonzeros[-3]]
                l1 = zigzag_score[nonzeros[-1]]
            elif direction[-1] == 0:
                if direction[nonzeros[-1]] > 0:
                    # hlh l
                    h2 = zigzag_score[nonzeros[-3]]
                    h1 = zigzag_score[nonzeros[-1]]
                    l2 = zigzag_score[nonzeros[-2]]
                    l1 = array[i]
                elif direction[nonzeros[-1]] < 0:
                    # lhl h
                    h2 = zigzag_score[nonzeros[-2]]
                    h1 = array[i]
                    l2 = zigzag_score[nonzeros[-3]]
                    l1 = zigzag_score[nonzeros[-1]]

            if (direction[-1] > 0) or (
                direction[-1] == 0 and direction[nonzeros[-1]] < 0
            ):
                if h1 > h2 and l1 >= l2 and array[i - 1] < h2:
                    # 1) 新高不新低
                    res[i] = 1
            elif (direction[-1] < 0) or (
                direction[-1] == 0 and direction[nonzeros[-1]] > 0
            ):
                if l1 < l2 and h1 <= h2 and array[i - 1] > l2:
                    # 2) 新低不新高
                    res[i] = -1

    return res


def pivot(array, left, right, isHigh, min_diff=0.0):
    res = np.full_like(array, np.nan)
    n = len(array)
    for i in range(n):
        if i < left:
            continue
        left_bound = i - left
        right_bound = i + right
        nanmax_left = np.nanmax(array[left_bound:i])
        nanmax_right = np.nanmax(array[i:right_bound])
        nanmin_left = np.nanmin(array[left_bound:i])
        nanmin_right = np.nanmin(array[i:right_bound])

        diff = array[i] * min_diff
        if (
            isHigh
            and array[i] >= nanmax_left
            and array[i] >= nanmax_right
            and min(array[i] - nanmin_left, array[i] - nanmin_right) >= diff
        ):
            res[i] = array[i]

        elif (
            not isHigh
            and array[i] <= nanmin_left
            and array[i] <= nanmin_right
            and min(nanmax_right - array[i], nanmax_left - array[i]) >= diff
        ):
            res[i] = array[i]

        else:
            res[i] = res[i - 1]
    return shift(res, right)


def nz(x, y=0, idx=None):
    if idx is not None and idx < 0:
        return y
    elif idx is not None:
        x = x[idx]
    return x if not np.isnan(x) else y


def get_VWAP(h, l, c, v, window):
    price = (h + l + c) / 3 * v
    sum_numerator = pd.Series(price).rolling(window).sum()
    sum_denominator = pd.Series(v).rolling(window).sum()
    return sum_numerator / sum_denominator


def get_autonomous_recursive_ma(close, length, gamma, zero_lag=False, sum_window=1000):
    """
    zl = false
    ma = 0.
    mad = 0.
    src_ = zl ? close + change(close,length_/2) : close
    ma := nz(mad[1],src_)
    d = cum(abs(src_[length_] - ma))/ bar_index * gamma
    mad := sma(sma(src_ > nz(mad[1],src_) + d ? src_ + d : src_ < nz(mad[1],src_) - d ? src_ - d : nz(mad[1],src_),length_),length_)
    mad_up = mad > mad[1]
    madup = mad > mad[1] ? #009688  : #f06292
    mad_f = mad/mad[1] > .999 and mad/mad[1] < 1.001
    """
    if zero_lag:
        if isinstance(close, pd.Series):
            dif = close.diff(length // 2)
            dif.fillna(0, inplace=True)
        else:
            dif = np.diff(close, n=length // 2)
            dif = np.concatenate([np.zeros(length // 2), dif])

        close = close + dif

    ma = np.zeros_like(close)
    mad = np.full_like(close, np.nan)
    x = np.full_like(close, np.nan)
    dx = np.full_like(close, np.nan)
    d = np.full_like(close, np.nan)
    cum_sum = 0
    for i in range(len(close)):
        if i >= length:
            ma[i] = mad[i - 1] if not np.isnan(mad[i - 1]) else close[i]
            cum_sum += abs(close[i - length] - ma[i])
            if i >= (sum_window + length):
                d[i] = (
                    np.nanmean(
                        np.abs(
                            close[i - length - sum_window : i - length]
                            - ma[i - sum_window : i]
                        )
                    )
                    * gamma
                )
            else:
                d[i] = np.nanmean(np.abs(close[: i - length] - ma[length:i])) * gamma
            # d[i] = cum_sum / i * gamma

            if close[i] > (nz(mad[i - 1], close[i]) + d[i]):
                x[i] = close[i] + d[i]

            elif close[i] < (nz(mad[i - 1], close[i]) - d[i]):
                x[i] = close[i] - d[i]

            else:
                x[i] = nz(mad[i - 1], close[i])

            dx[i] = np.nanmean(x[i - length : i])
            mad[i] = np.nanmean(dx[i - length : i])

    return mad


def getSuperTrend(high, low, close, atr_window, multiplier):
    hl = (high + low) / 2
    atr = get_atr(high, low, close, atr_window)
    final_upperband = hl + multiplier * atr
    final_lowerband = hl - multiplier * atr
    supertrend = [True] * len(high)
    for i in range(1, len(high)):
        curr, prev = i, i - 1
        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]
            # adjustment to the final bands
            if (
                supertrend[curr] == True
                and final_lowerband[curr] < final_lowerband[prev]
            ):
                final_lowerband[curr] = final_lowerband[prev]
            if (
                supertrend[curr] == False
                and final_upperband[curr] > final_upperband[prev]
            ):
                final_upperband[curr] = final_upperband[prev]
        # remove bands depending on the trend direction for visualization
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan

    return supertrend, final_upperband, final_lowerband


def hilbertTransform(src):
    src[np.isnan(src)] = 0.0
    return (
        0.0962 * src
        + 0.5769 * shift(src, 2)
        - 0.5769 * shift(src, 4)
        - 0.0962 * shift(src, 6)
    )


def getCloud(src, er_window):
    def hilbertTransform(array, i):
        a = array[i]
        b = nz(array, idx=i - 2)
        c = nz(array, idx=i - 4)
        d = nz(array, idx=i - 6)
        return 0.0962 * a + 0.5769 * b - 0.5769 * c - 0.0962 * d

    def computeComponent(array, period, i):
        return hilbertTransform(array, i) * period

    mesaPeriod = np.zeros_like(src)
    smooth = np.zeros_like(src)
    detrender = np.zeros_like(src)
    I1 = np.zeros_like(src)
    I2 = np.zeros_like(src)
    detrender = np.zeros_like(src)
    Q1 = np.zeros_like(src)
    Q2 = np.zeros_like(src)
    jI = np.zeros_like(src)
    jQ = np.zeros_like(src)
    Re = np.zeros_like(src)
    Im = np.zeros_like(src)
    phase = np.zeros_like(src)
    alpha = np.full_like(src, np.nan)
    PI = 2 * np.arcsin(1)
    er = market_efficiency(src, er_window)
    assert len(er) == len(src)
    for i in range(len(src)):
        if i == 0:
            continue
        mesaPeriodMult = 0.075 * mesaPeriod[i - 1] + 0.54
        smooth[i] = (
            4 * nz(src, idx=i)
            + 3 * nz(src, idx=i - 1)
            + 2 * nz(src, idx=i - 2)
            + nz(src, idx=i - 3)
        ) / 10
        detrender[i] = computeComponent(smooth, mesaPeriodMult, i)
        I1[i] = nz(detrender, idx=i - 3)
        Q1[i] = computeComponent(detrender, mesaPeriodMult, i)
        jI[i] = computeComponent(I1, mesaPeriodMult, i)
        jQ[i] = computeComponent(Q1, mesaPeriodMult, i)
        I2[i] = I1[i] - jQ[i]
        Q2[i] = Q1[i] + jI[i]
        I2[i] = 0.2 * I2[i] + 0.8 * nz(I2, idx=i - 1)
        Q2[i] = 0.2 * Q2[i] + 0.8 * nz(Q2, idx=i - 1)
        Re[i] = I2[i] * nz(I2, idx=i - 1) + Q2[i] * nz(Q2, idx=i - 1)
        Im[i] = I2[i] * nz(Q2, idx=i - 1) - Q2[i] * nz(I2, idx=i - 1)
        Re[i] = 0.2 * Re[i] + 0.8 * nz(Re, idx=i - 1)
        Im[i] = 0.2 * Im[i] + 0.8 * nz(Im, idx=i - 1)
        if Re[i] != 0 and Im[i] != 0:
            mesaPeriod[i] = 2 * PI / np.arctan(Im[i] / Re[i])

        mesaPeriod[i] = clip(
            mesaPeriod[i],
            0.67 * nz(mesaPeriod, idx=i - 1),
            1.5 * nz(mesaPeriod, idx=i - 1),
        )
        mesaPeriod[i] = clip(mesaPeriod[i], 6, 50)
        mesaPeriod[i] = 0.2 * mesaPeriod[i] + 0.8 * nz(mesaPeriod, idx=i - 1)
        phase[i] = 0.0
        if I1[i] != 0:
            phase[i] = (180 / PI) * np.arctan(Q1[i] / I1[i])
        deltaPhase = nz(phase, idx=i - 1) - phase[i]
        deltaPhase = min(deltaPhase, 1.0)
        fastLimit = er[i]
        slowLimit = er[i] * 0.1
        alpha[i] = max(fastLimit / deltaPhase, slowLimit)
    return alpha


def calcADX(high, low, close, window):
    up = high - shift(high, 1)
    down = -(low - shift(low, 1))
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((up < down) & (down > 0), down, 0)
    atr = get_atr(high, low, close, window)
    _plus = 100 * rma(plus_dm, window) / atr
    _minus = 100 * rma(minus_dm, window) / atr
    s = _plus + _minus
    s[s == 0] = 1
    _adx = 100 * rma(abs(_plus - _minus) / s, window)
    return _plus, _minus, _adx


def calcADX_adx(high, low, close, window):
    up = high - shift(high, 1)
    down = -(low - shift(low, 1))
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((up < down) & (down > 0), down, 0)
    atr = get_atr(high, low, close, window)
    _plus = 100 * rma(plus_dm, window) / atr
    _minus = 100 * rma(minus_dm, window) / atr
    s = _plus + _minus
    s[s == 0] = 1
    _adx = 100 * rma(abs(_plus - _minus) / s, window)
    return _adx


def calcADX_plus(high, low, close, window):
    up = high - shift(high, 1)
    down = -(low - shift(low, 1))
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    atr = get_atr(high, low, close, window)
    _plus = 100 * rma(plus_dm, window) / atr
    return _plus


def calcADX_minus(high, low, close, window):
    up = high - shift(high, 1)
    down = -(low - shift(low, 1))
    minus_dm = np.where((up < down) & (down > 0), down, 0)
    atr = get_atr(high, low, close, window)
    _minus = 100 * rma(minus_dm, window) / atr
    return _minus


def getRF(open, per_, mult):
    wper = 2 * per_ - 1
    avrng = talib.SMA(open.s.diff(1), per_)
    _smoothrng = (
        talib.SMA(
            avrng,
            wper,
        )
        * mult
    )
    _filt = open.s
    cond1 = _filt > _filt.shift(1)  # 在上升
    # cond2 = (open.s - _smoothrng) < _filt.shift(1)
    # cond3 = (open.s + _smoothrng) > _filt.shift(1)
    _filt = np.where(
        cond1,
        np.maximum(_filt.shift(1), _filt - _smoothrng),
        np.minimum(_filt + _smoothrng, _filt.shift(1)),
    )

    _upward = np.zeros_like(open)
    _downward = np.zeros_like(open)
    for i in range(len(_upward)):
        _downward[i] = _downward[i - 1]
        _upward[i] = _upward[i - 1]
        if i > 1:
            if _filt[i] > _filt[i - 1]:
                _upward[i] = _upward[i - 1] + 1
                _downward[i] = 0
            elif _filt[i] < _filt[i - 1]:
                _upward[i] = 0
                _downward[i] = _downward[i - 1] + 1

    return _filt, _smoothrng, _downward, _upward


def market_efficiency(c, n):
    """
    取值0-1，0代表波动很大。

    """
    c: pd.Series = c.s
    move_speed = (c - c.shift(n)).abs()
    volatility = (c - c.shift(1)).abs().rolling(n).sum()
    return move_speed / volatility


def ma(array, n):
    return array.rolling(n).mean()


def rma(array: _Array, n):
    if isinstance(array, pd.DataFrame):
        return array.ewm(alpha=1 / n).mean().values
    return pd.DataFrame({"value": array}).ewm(alpha=1 / n).mean()["value"].values


def donchian_up(array, n):
    return shift(talib.MAX(array, n), 1)


def donchian_down(array, n):
    return shift(talib.MIN(array, n), 1)


def BBANDS_U(h, l, c, n_lookback, n_std):
    """Bollinger bands indicator"""
    hlc3 = (h + l + c) / 3
    mean, std = talib.SMA(hlc3, n_lookback), talib.STDDEV(hlc3, n_lookback)
    upper = mean + n_std * std
    return upper


def BBANDS_D(h, l, c, n_lookback, n_std):
    """Bollinger bands indicator"""
    hlc3 = (h + l + c) / 3
    mean, std = talib.SMA(hlc3, n_lookback), talib.STDDEV(hlc3, n_lookback)
    lower = mean - n_std * std
    return lower


def get_donchinan_signal(h, l, c, n):
    pre_low = shift(talib.MIN(l, n), 1)
    pre_high = shift(talib.MAX(h, n), 1)
    signal = np.zeros_like(c)
    signal[c > pre_high] = 1
    signal[c < pre_low] = -1
    return signal


def get_kufman_ma(c, window):
    return talib.KAMA(c, window)


def get_ma(c, window, ma_type="SMA"):
    if ma_type == "SMA":
        return talib.MA(c, window)
    elif ma_type == "EMA":
        return (
            c.s.to_frame().ewm(alpha=2 / (window + 1), min_periods=window).mean().values
        )

def get_boll_up(c, window, multiplier):
    ma = get_ma(c, window)
    std = get_std(c, window)
    return ma + multiplier * std

def get_boll_down(c, window, multiplier):
    ma = get_ma(c, window)
    std = get_std(c, window)
    return ma - multiplier * std

def get_volume_break_ratio(v, window):
    v_ma = get_ma(v, window)
    return v / v_ma


def get_macd(c, fast, slow):
    fast_ma = get_ma(c, fast)
    slow_ma = get_ma(c, slow)
    macd = fast_ma - slow_ma
    return macd


def get_bar_stats(o, h, l, c):
    return min(abs(h - l) / (abs(c - o) + 0.001), 20)


def get_macd_signal(c, fast, slow, signal_window):
    fast_ma = get_ma(c, fast)
    slow_ma = get_ma(c, slow)
    macd = fast_ma - slow_ma
    signal = get_ma(macd, signal_window)
    return signal


def get_std(c, window):
    return talib.STDDEV(c, window)


def get_ma_angle(c, window):
    c = c / c[np.isfinite(c)][0]
    a = talib.MA(c, window)
    angle = talib.LINEARREG_ANGLE(a, 2)
    return angle


def get_ma_slope_filter(c, long_window, short_window):
    slow_ma_angle = np.abs(get_ma_angle(c, long_window))
    fast_ma_angle = np.abs(get_ma_angle(c, short_window))
    signal = np.zeros_like(c)
    signal[(slow_ma_angle < 0.006) & (fast_ma_angle < 0.006)] = Forbidden.NO_TRADE
    signal[(np.isnan(slow_ma_angle) | np.isnan(fast_ma_angle))] = np.nan
    return signal


def get_double_ma_signal(c, long_window, short_window):
    """up-cross = 1, down-cross = -1"""
    slow_ma = get_ma(c, long_window)
    fast_ma = get_ma(c, short_window)
    pre_slow_ma = shift(slow_ma, 1)
    pre_fast_ma = shift(fast_ma, 1)
    signal = np.zeros_like(c)
    signal[(fast_ma > slow_ma) & (pre_fast_ma < pre_slow_ma)] = 1  # up-cross
    signal[(fast_ma < slow_ma) & (pre_fast_ma > pre_slow_ma)] = -1
    signal[np.isnan(slow_ma)] = np.nan
    return signal


def get_double_kaufman_ma_signal(c, long_window, short_window):
    """up-cross = 1, down-cross = -1"""
    slow_ma = get_kufman_ma(c, long_window)
    fast_ma = get_kufman_ma(c, short_window)
    pre_slow_ma = shift(slow_ma, 1)
    pre_fast_ma = shift(fast_ma, 1)
    signal = np.zeros_like(c)
    signal[(fast_ma > slow_ma) & (pre_fast_ma < pre_slow_ma)] = 1  # up-cross
    signal[(fast_ma < slow_ma) & (pre_fast_ma > pre_slow_ma)] = -1
    signal[np.isnan(slow_ma)] = np.nan
    return signal


def get_adx_filter(c, h, l, period):
    adx = talib.ADX(h, l, c, period)
    signal = np.zeros_like(c)
    signal[(adx < 2)] = Forbidden.NO_TRADE
    signal[np.isnan(adx)] = np.nan
    return signal


def get_trend_filter(array, long_window, short_window):
    long_ma1 = talib.SMA(array, long_window)
    short_ma1 = talib.SMA(array, short_window)
    signal = np.zeros_like(array)
    signal[long_ma1 > short_ma1] = Forbidden.NO_LONG
    signal[long_ma1 < short_ma1] = Forbidden.NO_SHORT
    signal[np.isnan(long_ma1)] = np.nan
    return signal


def get_vwap(array, weight, window):
    nom = (array.s * weight.s).rolling(window).sum()
    dnom = weight.s.rolling(window).sum()
    return nom / dnom


def get_rsi_filter(array, long_window):
    """oversold and overboung"""
    rsi = talib.RSI(array, long_window)
    signal = np.zeros_like(array)
    signal[rsi > 60] = Forbidden.NO_LONG
    signal[rsi < 40] = Forbidden.NO_SHORT
    return signal


def get_mfi(h, l, c, v, window):
    mfi = talib.MFI(h, l, c, v, timeperiod=window)
    return mfi


def get_mfi_filter(h, l, c, v, window):
    mfi = talib.MFI(h, l, c, v, timeperiod=window)
    signal = np.zeros_like(h)
    signal[(mfi <= 52) & (mfi >= 48)] = Forbidden.NO_TRADE
    return signal


def get_atr(h, l, c, n, normalize=False, ma_type="rma"):
    tr = np.maximum(h - l, abs(h - shift(c, 1)), abs(l - shift(c, 1)))
    if ma_type == "rma":
        atr = rma(tr, n)
    elif ma_type == "sma":
        atr = ma(tr, n)
    if normalize:
        atr = atr / c
    return atr


def get_unit(h, l, c, n, capital):
    atr = talib.ATR(h.s, l.s, c.s, n)
    return np.ceil(0.01 * capital / atr)


def equal_map(s):
    return s


def risk_adjust_position(entry_price, exit_price, total_cache, risk):
    assert 0.2 > risk > 0
    cost = total_cache * risk
    exp_lost = abs(entry_price - exit_price)
    pos = cost / exp_lost
    return int(pos)