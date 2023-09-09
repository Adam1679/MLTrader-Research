from research.strategies.strategy_utils import *


class BBABDS_COPY_STRATEGY_2(Strategy):
    _global = None
    long_signal = None
    short_signal = None
    parameters = ["long_signal", "short_signal", "_global"]

    def init(self):
        # ADX
        adx_len = 21
        self.th = 20
        [DIPlusC, DIMinusC, ADXC] = calcADX(
            self.data.High, self.data.Low, self.data.Close, adx_len
        )
        self.plus_di = self.I(equal_map, DIPlusC, plot=False, name="plus_di")
        self.minus_di = self.I(equal_map, DIMinusC, plot=False, name="minus_di")
        self.adx = self.I(equal_map, ADXC, plot=True, name="ADX")
        # MACD
        fast_length = 15
        slow_length = 16
        signal_length = 26
        self.fast_ma = self.I(
            talib.SMA,
            self.data.Open,
            fast_length,
            plot=False,
            overlay=False,
            name="fast MA",
        )
        self.slow_ma = self.I(
            talib.SMA,
            self.data.Open,
            slow_length,
            plot=False,
            overlay=False,
            name="slow MA",
        )
        self.macd = self.I(
            lambda a, b: a - b,
            self.fast_ma,
            self.slow_ma,
            plot=True,
            overlay=False,
            name="macd",
        )
        self.macd_mean = self.I(
            talib.SMA, self.macd, signal_length, plot=True, name="macd mean"
        )

        # RF
        per_ = 15
        mult = 2.6
        filt, smoothrng, downward, upward = getRF(self.data.Open, per_, mult)
        self.hband = self.I(
            lambda: filt + smoothrng, plot=True, name="RF UBand", overlay=True
        )
        self.lband = self.I(
            lambda: filt - smoothrng, plot=True, name="RF LBand", overlay=True
        )
        self.upward = self.I(equal_map, upward, plot=False, name="upward")
        self.downward = self.I(equal_map, downward, plot=False, name="downward")
        # SAR

        Sst, Sinc, Smax = 0.5, 0.2, 0.4
        self.sar = self.I(
            talib.SAR,
            self.data.High,
            self.data.Low,
            acceleration=Sinc,
            maximum=Smax,
            plot=True,
        )
        # Volume
        self.volume_f, self.sma_length, self.volume_f1, self.sma_length1 = (
            3.2,
            20,
            1.9,
            22,
        )
        self.ma_volume = self.I(
            talib.SMA, self.data.Volume, self.sma_length, plot=False
        )
        self.ma_volume1 = self.I(
            talib.SMA, self.data.Volume, self.sma_length1, plot=False
        )

        # TP
        self.tp = self._global.MANAGER.take_profit
        self.sl = self._global.MANAGER.stop_loss

        # BBand
        self.per2 = 20
        self.dev2 = 2.0
        upperband, middleband, lowerband = talib.BBANDS(
            self.data.High, self.per2, self.dev2, self.dev2
        )
        self.hb2 = self.I(equal_map, upperband, plot=True, overlay=True, name="B UBand")
        self.lb2 = self.I(equal_map, lowerband, plot=True, overlay=True, name="B LBand")
        self.ma2 = self.I(equal_map, middleband, plot=False)

        self.trend_long = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="trend long"
        )
        self.trend_short = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="trend short"
        )
        self.ml_long = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ml long"
        )
        self.ml_short = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ml short"
        )

        rolling_window = (
            self._global.MANAGER.quantile_rolling_days
            * self._global.N_MINS_ONE_DAY
            // self._global.BAR_WINDOW
        )
        self.long_t = self.I(
            quantile_map,
            self.long_signal,
            rolling_window,
            1 - self._global.MANAGER.quantile_rate,
            overlay=False,
            plot=True,
        )
        self.short_t = self.I(
            quantile_map,
            self.short_signal,
            rolling_window,
            1 - self._global.MANAGER.quantile_rate,
            overlay=False,
            plot=True,
        )
        self.long_signal = self.I(equal_map, self.long_signal, plot=True, overlay=False)
        self.short_signal = self.I(
            equal_map, self.short_signal, plot=True, overlay=False
        )

    def next(self):
        self.ml_long[-1] = self.long_signal[-1] > self.long_t[-1]
        self.ml_short[-1] = self.short_signal[-1] > self.short_t[-1]

        L_macd = self.macd[-1] > self.macd_mean[-1]
        S_macd = self.macd[-1] < self.macd_mean[-1]

        L_adx = self.plus_di[-1] > self.minus_di[-1] and self.adx[-1] > self.th
        S_adx = self.plus_di[-1] < self.minus_di[-1] and self.adx[-1] > self.th

        L_sar = self.sar[-1] < self.data.Close[-1]
        S_sar = self.sar[-1] > self.data.Close[-1]

        Volume_condt1 = self.data.Volume[-1] > self.ma_volume1[-1] * self.volume_f1

        L_RF = self.data.High[-1] > self.hband[-1] and self.upward[-1] > 0
        S_RF = self.data.Low[-1] < self.lband[-1] and self.downward[-1] > 0

        # trend signal
        self.trend_long[-1] = L_adx and L_RF and L_macd and L_sar and Volume_condt1
        self.trend_short[-1] = S_adx and S_RF and S_macd and S_sar and Volume_condt1

        # longCondition = self.trend_long[-1] > 0 and self.ml_long[-1]
        # shortCondition = self.trend_short[-1] > 0 and self.ml_short[-1]
        longCondition = self.trend_long[-1] > 0 and self.ml_long[-1]
        shortCondition = self.trend_short[-1] > 0 and self.ml_short[-1]
        if not self.position:
            if longCondition:
                price = self.data.Close[-1]
                tp = price * (1 + self.tp)
                sl = price * (1 - self.sl)
                self.buy(size=0.99, tp=tp, sl=sl)
            elif shortCondition:
                price = self.data.Close[-1]
                tp = price * (1 - self.tp)
                sl = price * (1 + self.sl)
                self.sell(size=0.99, tp=tp, sl=sl)
        elif self.position.is_long and shortCondition:
            for trade in self.trades:
                trade.close()
            price = self.data.Close[-1]
            tp = price * (1 - self.tp)
            sl = price * (1 + self.sl)
            self.sell(size=0.99, tp=tp, sl=sl)
        elif self.position.is_short and longCondition:
            for trade in self.trades:
                trade.close()
            price = self.data.Close[-1]
            tp = price * (1 + self.tp)
            sl = price * (1 - self.sl)
            self.buy(size=0.99, tp=tp, sl=sl)
        else:
            for trade in self.trades:
                if (self.data.index[-1] - trade.entry_time) > pd.Timedelta("1h"):
                    trade.close()


class BBABDS_ML_STRATEGY(Strategy):
    _global = None
    long_signal = None
    short_signal = None
    parameters = ["long_signal", "short_signal", "_global"]
    max_hold_seconds = None
    stop_loss = None
    take_profit = None

    def init(self):
        # Precompute the two moving averages
        rolling_window = 7 * 24 * 60 // 15
        self.quantile_rate = 0.1
        self.bband_up = self.I(
            BBANDS_U,
            self.data.High,
            self.data.Close,
            self.data.Low,
            20,
            2,
            overlay=True,
            plot=True,
        )
        self.bband_down = self.I(
            BBANDS_D,
            self.data.High,
            self.data.Close,
            self.data.Low,
            20,
            2,
            overlay=True,
            plot=True,
        )

        self.long_t = self.I(
            quantile_map,
            self.long_signal,
            rolling_window,
            1 - self.quantile_rate,
            overlay=False,
            plot=True,
        )
        self.short_t = self.I(
            quantile_map,
            self.short_signal,
            rolling_window,
            self.quantile_rate,
            overlay=False,
            plot=True,
        )
        self.long_signal = self.I(equal_map, self.long_signal, plot=True, overlay=False)
        self.short_signal = self.I(
            equal_map, self.short_signal, plot=True, overlay=False
        )
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        self.tp = 0.009
        self.sl = 0.042

    def next(self):
        bband_L = (
            self.data.Close[-1] < self.bband_down[-1]
            and self.data.Close[-2] > self.bband_down[-2]
        )
        bband_S = (
            self.data.Close[-1] > self.bband_down[-1]
            and self.data.Close[-2] < self.bband_down[-2]
        )

        ML_L = self.long_signal[-1] > self.long_t[-1]
        ML_S = self.long_signal[-1] > self.short_t[-1]
        if not self.position:
            if bband_L and ML_L:
                self.buy(
                    size=0.99,
                    sl=self.data.Close[-1] * (1 - self.sl),
                    tp=self.data.Close[-1] * (1 + self.tp),
                )
            if bband_S and ML_S:
                self.sell(
                    size=0.99,
                    sl=self.data.Close[-1] * (1 + self.sl),
                    tp=self.data.Close[-1] * (1 - self.tp),
                )

        elif self.position.is_long:
            if self.data.Close[-1] > self.bband_up[-1]:
                for trade in self.trades:
                    trade.close()

        elif self.position.is_short:
            if self.data.Close[-1] < self.bband_down[-1]:
                for trade in self.trades:
                    trade.close()


class SUPER_TREND(Strategy):
    def init(self):
        # Precompute the two moving averages
        self.risk_factor = 3
        self.atr_window = 35
        self.atr_multiplier = 3
        self.fast_dema_window = 144
        self.slow_dema_window = 169

        supertrend, final_upperband, final_lowerband = getSuperTrend(
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_window,
            self.atr_multiplier,
        )
        self.super_trend = self.I(equal_map, supertrend, plot=True)
        self.final_upperband = self.I(
            equal_map, final_upperband, overlay=True, plot=True, name="hi"
        )
        self.final_lowerband = self.I(
            equal_map, final_lowerband, overlay=True, plot=True, name="lo"
        )
        self.fast_ma = self.I(
            talib.DEMA, self.data.Close, self.fast_dema_window, overlay=True, plot=True
        )
        self.slow_ma = self.I(
            talib.DEMA, self.data.Close, self.slow_dema_window, overlay=True, plot=True
        )
        self.MA_L = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MA L"
        )
        self.MA_S = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MA S"
        )
        self.TREND_L = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="TREND L"
        )
        self.TREND_S = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="TREND S"
        )

    def next(self):
        self.MA_L[-1] = self.data.Close[-1] > max(self.fast_ma[-1], self.slow_ma[-1])
        self.MA_S[-1] = self.data.Close[-1] < min(self.fast_ma[-1], self.slow_ma[-1])

        self.TREND_L[-1] = self.super_trend[-1] > self.super_trend[-2]
        self.TREND_S[-1] = self.super_trend[-1] < self.super_trend[-2]

        LONG = self.TREND_L[-1] and self.MA_L[-1]
        SHORT = self.TREND_S[-1] and self.MA_S[-1]
        if not self.position:
            if LONG:
                sl = min(self.fast_ma[-1], self.slow_ma[-1])
                tp = (
                    self.risk_factor * abs(self.data.Close[-1] - sl)
                    + self.data.Close[-1]
                )
                self.buy(tp=tp, sl=sl)

            elif SHORT:
                sl = max(self.fast_ma[-1], self.slow_ma[-1])
                tp = max(
                    self.data.Close[-1]
                    - self.risk_factor * abs(self.data.Close[-1] - sl),
                    0.001,
                )
                self.sell(tp=tp, sl=sl)

        elif self.position.is_long and SHORT:
            for trade in self.trades:
                trade.close()

        elif self.position.is_short and LONG:
            for trade in self.trades:
                trade.close()
