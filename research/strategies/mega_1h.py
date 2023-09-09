from math import floor

from research.strategies.automl_strategies import *

"""
持仓一天到一周不等一个中等周期趋势跟踪策略，在BTC和ETH上效果还行，4h和1h都可以实践
"""


class Mega_1h(AutoMLStrategy):
    parameter_specs = [
        {
            "name": "adx_window",
            "type": "choice",
            "values": list(range(5, 100)),
            "is_ordered": True,
        },
        {
            "name": "sl",
            "type": "range",
            "bounds": (1e-3, 1e-1),
        },
        {
            "name": "tp1",
            "type": "range",
            "bounds": (1e-3, 1e-1),
        },
        {
            "name": "adx_th",
            "type": "range",
            "bounds": (10, 50),
        },
        {
            "name": "volume_f",
            "type": "range",
            "bounds": (1.0, 2.0),
        },
        {
            "name": "left",
            "type": "choice",
            "values": list(range(3, 10)),
            "is_ordered": True,
        },
        {
            "name": "right",
            "type": "choice",
            "values": list(range(3, 10)),
            "is_ordered": True,
        },
    ]
    parameters = [
        "sl",
        "tp1",
        # "vwap_rsi_window",
        "adx_th",
        "adx_window",
        "volume_f",
        "left",
        "right",
    ]
    constraints = []
    fixed_parameters = ["side", "max_sl", "bankroll", "risk_level"]

    def init(self):
        self.left = int(self.hparams.get("left", 10))
        self.right = int(self.hparams.get("right", 5))
        self.volume_f = float(self.hparams.get("volume_f", 2.1))
        self.volume_ma_window = int(self.hparams.get("volume_ma_window", 52))
        self.adx_window = int(self.hparams.get("adx_window", 13))
        self.adx_th = float(self.hparams.get("adx_th", 13.5))
        self.sl = float(self.hparams.get("sl", 0.07))
        self.tp1 = float(self.hparams.get("tp1", 0.045))
        self.tp_atr = float(self.hparams.get("tp_atr", 4))
        self.sl_atr = float(self.hparams.get("sl_atr", 2.5))
        self.atr_window = float(self.hparams.get("atr_window", 26))
        self.mad_window = int(self.hparams.get("mad_window", 15))
        self.mad_gamma = int(self.hparams.get("mad_gamma", 3))
        self.long_volume_period = int(self.hparams.get("long_volume_period", 33))
        self.rsi_window = int(self.hparams.get("rsi_window", 90))
        self.macd_signal_window = int(self.hparams.get("macd_signal_window", 21))
        self.macd_fast = int(self.hparams.get("macd_fast", 9))
        self.macd_slow = int(self.hparams.get("macd_slow", 12))
        self.rsi = self.I(get_rsi, self.data.Open, self.rsi_window, plot=True)
        pivot_price = (
            self.data.High + self.data.Low + 2 * self.data.Close
        ) / 4  # 使用这个带来触发了更多的交易次数
        self.pivot_high = self.I(
            pivot, pivot_price, self.left, self.right, True, plot=True, overlay=True
        )

        self.pivot_low = self.I(
            pivot, pivot_price, self.left, self.right, False, plot=True, overlay=True
        )

        self.MAD_FLAT_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MAD FLAT"
        )
        self.pivot_diff = self.I(
            lambda: self.pivot_high - self.pivot_low, plot=False, name="PIVOT DIFF"
        )
        self.zigzag = self.I(
            lambda: np.zeros_like(self.data.Close),
            plot=True,
            name="PIVOT FLAT",
            scatter=True,
            overlay=True,
        )
        data = (self.data.Close + self.data.Open + self.data.High + self.data.Low) / 4
        fast_ma = get_ma(data, self.macd_fast)
        slow_ma = get_ma(data, self.macd_slow)
        macd = fast_ma - slow_ma
        signal = get_ma(macd, self.macd_signal_window)
        self.macd = self.I(equal_map, macd, name="macd", plot=True)
        self.macd_signal = self.I(equal_map, signal, name="macd_signal", plot=True)
        self.mad = self.I(
            get_autonomous_recursive_ma,
            self.data.Close,
            self.mad_window,
            self.mad_gamma,
            zero_lag=True,
            plot=True,
            overlay=True,
        )
        self.ma_90 = self.I(get_ma, self.data.Close, 90, plot=True, overlay=True)
        self.market_efficiency = self.I(
            market_efficiency, self.data.Close, 5, name="market_efficiency", plot=True
        )
        plus, minus, adx = calcADX(
            self.data.High, self.data.Low, self.data.Close, self.adx_window
        )
        self.plus = self.I(equal_map, plus, name="adx_plus", plot=True)
        self.minus = self.I(equal_map, minus, name="adx_minus", plot=True)
        self.adx = self.I(equal_map, adx, name="adx", plot=True)

        self.R = 0.01
        self.volume_ma = self.I(
            get_ma, self.data.Volume, self.volume_ma_window, plot=True, overlay=False
        )

        self.RS_BREAK_ENTRY = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="RS_BREAK_ENTRY"
        )

        self.volume_adv = self.I(
            get_volume_advantage,
            self.data.Close,
            self.data.Volume,
            self.long_volume_period,
            plot=True,
        )

        self.atr = self.I(
            get_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_window,
            plot=True,
        )

        self.MACD_BREAK_COND = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MACD_BREAK_COND"
        )
        self.VOLUME_BREAK_COND = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VOLUME_BREAK_COND"
        )
        self.ADX_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ADX_SIGNAL"
        )

        self.break_even_point = None
        self.estimated_flat_num_bars = 20
        self.last_low = True
        self.open_price_move = None

    def next(self):
        self.zigzag[-1] = np.nan
        if self.last_low:
            if self.pivot_high[-1] != self.pivot_high[-2]:
                self.zigzag[-1] = self.pivot_high[-1]
                self.last_low = False
        else:
            if self.pivot_low[-1] != self.pivot_low[-2]:
                self.zigzag[-1] = self.pivot_low[-1]
                self.last_low = True

        self.MAD_FLAT_FILTER[-1] = abs(self.mad[-1] / self.mad[-2] - 1) < 0.001
        if self.data.Volume[-1] > self.volume_ma[-1] * self.volume_f:
            self.VOLUME_BREAK_COND[-1] = 1

        if self.macd_signal[-1] < self.macd[-1]:
            self.MACD_BREAK_COND[-1] = 1

        if self.macd_signal[-1] > self.macd[-1]:
            self.MACD_BREAK_COND[-1] = -1

        if self.plus[-1] > self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = 1

        elif self.plus[-1] < self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = -1

        full_pos = 0.99
        if (
            self.rsi[-1] < 70
            and self.VOLUME_BREAK_COND[-1] > 0  # 有用，filter了很多差的交易
            and self.ADX_SIGNAL[-1] > 0  # 有用，filter了很多差的交易，但是收益也降低，并且交易次数显著降低一半
        ):
            self.RS_BREAK_ENTRY[-1] = 1

        elif (
            self.rsi[-1] > 30
            and self.VOLUME_BREAK_COND[-1] > 0  # 有用，filter了很多差的交易
            and self.ADX_SIGNAL[-1] < 0  # 有用，filter了很多差的交易，但是收益也降低，并且交易次数显著降低一半
            # and self.market_efficiency > 0.6 # 增加了一点点胜率，提高了盈亏比。减少了一些交易，提高夏普
            # and self.pivot_diff[-1] / self.pivot_diff_ma[-1] > 1.2 # 不是能很好的define trend
            # and self.MAD_FLAT_FILTER == 0
            # and self.data.Close[-1] < self.ma_90[-1]
            # and diff_ma_t
        ):
            self.RS_BREAK_ENTRY[-1] = -1

        LONG = self.RS_BREAK_ENTRY[-1] > 0

        SHORT = self.RS_BREAK_ENTRY[-1] < 0

        if LONG and self.side != "SHORT_ONLY":
            sl = self.data.Close[-1] - self.sl_atr * self.atr[-1]
            tp = self.data.Close[-1] + self.tp_atr * self.atr[-1]
            if sl > 0 and not self.position:
                # self.break_even_point = 2 * self.data.Close[-1] - sl
                # 之后尝试一下怎么多阶段卖出
                self.buy(size=full_pos, sl=sl, tp=tp, limit=self.data.Close[-1])  # 40%
                self.open_price_move = abs(self.data.Close[-1] - self.data.Open[-1])

        elif SHORT and self.side != "LONG_ONLY":
            sl = self.data.Close[-1] + self.sl_atr * self.atr[-1]
            tp = self.data.Close[-1] - self.tp_atr * self.atr[-1]
            if sl > 0 and tp > 0 and not self.position:
                # self.break_even_point = 2 * self.data.Close[-1] - sl
                self.sell(size=full_pos, sl=sl, tp=tp, limit=self.data.Close[-1])
                self.open_price_move = abs(self.data.Close[-1] - self.data.Open[-1])
