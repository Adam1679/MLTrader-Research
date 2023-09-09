from research.strategies.automl_strategies import *

# 首先5min的回测很不准，因为止盈止损点位都不太对，其次交易频率过高，只有在无交易手续费的情况下会有收益。
# 其次胜率似乎不稳定，后半年胜率只有50%


class MACD_Scalping(AutoMLStrategy):
    parameter_specs = [
        {
            "name": "hist_lookback_window",
            "type": "choice",
            "values": list(range(5, 20)),
            "is_ordered": True,
        },
        {
            "name": "sma_window_1",
            "type": "choice",
            "values": list(range(20, 50)),
            "is_ordered": True,
        },
        {
            "name": "sma_window_2",
            "type": "choice",
            "values": list(range(100, 200)),
            "is_ordered": True,
        },
        {
            "name": "rsi_window",
            "type": "choice",
            "values": list(range(2, 12)),
            "is_ordered": True,
        },
        {
            "name": "rsi_th",
            "type": "range",
            "bounds": (15, 40),
        },
        {
            "name": "hist_th",
            "type": "range",
            "bounds": (15, 40),
        },
    ]
    parameters = [
        "hist_th",
        "rsi_th",
        "rsi_window",
        "sma_window_2",
        "sma_window_1",
        "hist_lookback_window",
    ]
    constraints = []
    fixed_parameters = ["side", "max_sl", "bankroll"]

    def init(self):
        self.bar_since_last_crossover = 0
        self.bar_since_last_crossunder = 0

        self.hist_th = float(self.hparams.get("hist_th", 25))
        self.hist_lookback_window = int(self.hparams.get("hist_lookback_window", 10))
        self.sma_window_1 = int(self.hparams.get("sma_window_1", 50))
        self.sma_window_2 = int(self.hparams.get("sma_window_2", 200))
        self.bar_since_cross_th = int(self.hparams.get("bar_since_cross_th", 350))
        self.rsi_window = int(self.hparams.get("rsi_window", 9))
        self.rsi_th = float(self.hparams.get("rsi_th", 30))

        self.sl = float(self.hparams.get("sl", 0.005))
        self.tp1 = float(self.hparams.get("tp1", 0.005))
        self.ma_1 = self.I(
            talib.EMA, self.data.Close, self.sma_window_1, plot=True, overlay=True
        )
        self.ma_2 = self.I(
            talib.EMA, self.data.Close, self.sma_window_2, plot=True, overlay=True
        )

        macd, macdsignal, macdhist = talib.MACD(
            self.data.Close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        self.hist = self.I(
            equal_map, macdhist, plot=True, overlay=False, name="MACD HIST"
        )
        self.rsi = self.I(
            get_rsi,
            self.data.Close,
            self.rsi_window,
            plot=True,
            overlay=False,
            name="RSI",
        )

        self.MA_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MA_SIGNAL"
        )
        self.bar_since_last_crossover = self.I(
            lambda: np.zeros_like(self.data.Close),
            plot=True,
            name="bar_since_last_crossover",
        )
        self.bar_since_last_crossunder = self.I(
            lambda: np.zeros_like(self.data.Close),
            plot=True,
            name="bar_since_last_crossunder",
        )

        self.MACD_FILTER_1 = self.I(
            lambda: np.zeros_like(self.data.Close),
            plot=True,
            name="MACD absolute filter",
        )

        self.MACD_FILTER_2 = self.I(
            lambda: np.zeros_like(self.data.Close),
            plot=True,
            name="MACD relative filter",
        )

    def next(self):
        if self.ma_1[-1] > self.ma_2[-1] and self.ma_1[-2] < self.ma_2[-2]:
            # crossover
            self.bar_since_last_crossover[-1] = 0
        else:
            self.bar_since_last_crossover[-1] = self.bar_since_last_crossover[-2] + 1

        if self.ma_1[-1] < self.ma_2[-1] and self.ma_1[-2] > self.ma_2[-2]:
            # crossover
            self.bar_since_last_crossunder[-1] = 0
        else:
            self.bar_since_last_crossunder[-1] = self.bar_since_last_crossunder[-2] + 1

        if self.ma_1[-1] > self.ma_2[-1]:
            self.MA_SIGNAL[-1] = 1

        if self.ma_1[-1] < self.ma_2[-1]:
            self.MA_SIGNAL[-1] = -1

        if self.hist[-1] > 0 and self.hist[-1] > self.hist_th:
            self.MACD_FILTER_1[-1] = -1

        if self.hist[-1] < 0 and abs(self.hist[-1]) > self.hist_th:
            self.MACD_FILTER_1[-1] = 1

        if self.hist[-1] > 0 and (
            self.hist[-1] > max(self.hist[-self.hist_lookback_window : -1])
            or self.MACD_FILTER_2[-2] < 0
        ):
            self.MACD_FILTER_2[-1] = -1

        if self.hist[-1] < 0 and (
            self.hist[-1] < min(self.hist[-self.hist_lookback_window : -1])
            or self.MACD_FILTER_2[-2] > 0
        ):
            self.MACD_FILTER_2[-1] = 1

        LONG = (
            self.MA_SIGNAL[-1] > 0
            and self.bar_since_last_crossover[-1] < self.bar_since_cross_th
            and self.MACD_FILTER_1[-1] > 0
            and self.MACD_FILTER_2[-1] > 0
            and self.rsi < self.rsi_th
        )
        SHORT = (
            self.MA_SIGNAL[-1] < 0
            and self.bar_since_last_crossunder[-1] < self.bar_since_cross_th
            and self.MACD_FILTER_1[-1] < 0
            and self.MACD_FILTER_2[-1] < 0
            and self.rsi > (100 - self.rsi_th)
        )

        if not self.position:
            if LONG and self.side != "SHORT_ONLY":
                tp = self.data.Close[-1] * (1 + self.tp1)
                sl = self.data.Close[-1] * (1 - self.sl)
                self.buy(size=0.99, sl=sl, tp=tp, limit=self.data.Close[-1])

            elif SHORT and self.side != "LONG_ONLY":
                # sl = self.data.Close[-1] * (1 + self.sl_atr_rate * self.atr[-1])
                tp = self.data.Close[-1] * (1 - self.tp1)
                sl = self.data.Close[-1] * (1 + self.sl)
                self.sell(size=0.99, sl=sl, tp=tp, limit=self.data.Close[-1])
