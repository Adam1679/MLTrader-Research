import numpy as np
import pandas as pd
import talib
from backtesting import Strategy

from research.strategies.strategy_utils import *


class FIXED_TIME_HOLDING_WITH_PRED_CHANGE(Strategy):
    signal = None
    config_manager: "ConfigManager" = None
    parameters = ["signal", "config_manager"]
    max_hold_seconds = None
    stop_loss = None
    take_profit = None
    fixed_parameters = []

    def init(self):
        # Precompute the two moving averages
        self.unit = int(self._broker._cash / self.data.Close[0] / 5)
        assert self.unit > 1.0
        print("ML STRATEGY ONE UNIT SIZE = ", self.unit)
        self.quantile_rate0 = self.config_manager.quantile_rate * 2
        self.quantile_rate1 = self.config_manager.quantile_rate
        self.quantile_rate2 = self.config_manager.quantile_rate / 2
        self.max_hold_seconds = self.config_manager.maximum_holding_time_in_seconds
        self.stop_loss = self.config_manager.stop_loss
        self.take_profit = self.config_manager.take_profit
        rolling_window = (
            self.config_manager.quantile_rolling_days
            * self.config_manager.N_MINS_ONE_DAY
            // self.config_manager.bar_window
        )
        self.ml_signal = self.I(equal_map, self.signal, overlay=False, plot=True)
        self.atr = self.I(
            get_atr, self.data.High, self.data.Low, self.data.Close, 672, plot=True
        )
        self.long_t0 = self.I(
            quantile_map,
            self.signal,
            rolling_window,
            1 - self.quantile_rate0,
            overlay=False,
            plot=True,
        )
        self.short_t0 = self.I(
            quantile_map,
            self.signal,
            rolling_window,
            self.quantile_rate0,
            overlay=False,
            plot=True,
        )
        self.long_t1 = self.I(
            quantile_map,
            self.signal,
            rolling_window,
            1 - self.quantile_rate1,
            overlay=False,
            plot=True,
        )
        self.short_t1 = self.I(
            quantile_map,
            self.signal,
            rolling_window,
            self.quantile_rate1,
            overlay=False,
            plot=True,
        )
        self.long_t2 = self.I(
            quantile_map,
            self.signal,
            rolling_window,
            1 - self.quantile_rate2,
            overlay=False,
            plot=True,
        )
        self.short_t2 = self.I(
            quantile_map,
            self.signal,
            rolling_window,
            self.quantile_rate2,
            overlay=False,
            plot=True,
        )
        daily_trend = (
            self.config_manager.N_MINS_ONE_DAY // self.config_manager.bar_window
        )
        self.ADX = self.I(
            talib.ADX,
            self.data.High,
            self.data.Low,
            self.data.Close,
            daily_trend,
            plot=True,
        )
        self.ML_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ML SIGNAL"
        )

    def next(self):
        current_date = self.data.index[-1]
        long_signal, short_signal = 0, 0
        if self.ml_signal >= self.long_t2[-1]:
            long_signal = 3
        elif self.ml_signal >= self.long_t1[-1]:
            long_signal = 2
        elif self.ml_signal >= self.long_t0[-1]:
            long_signal = 1
        elif self.ml_signal[-1] <= self.short_t2[-1]:
            short_signal = 3
        elif self.ml_signal[-1] <= self.short_t1[-1]:
            short_signal = 2
        elif self.ml_signal[-1] <= self.short_t0[-1]:
            short_signal = 1

        if long_signal > 0:
            self.ML_SIGNAL[-1] = long_signal

        elif short_signal > 0:
            self.ML_SIGNAL[-1] = -short_signal
        if self.data.index[-1].minute == 0:
            cur_pos = (
                int(sum(map(lambda trade: trade.size, self.trades)))
                if len(self.trades) > 0
                else 0
            )
            if self.position.is_long:
                if long_signal > 0:
                    diff = long_signal * self.unit - cur_pos
                    if diff > 0:
                        self.buy(size=diff, limit=self.data.Close[-1])
                    elif diff < 0:
                        self.sell(size=diff, limit=self.data.Close[-1])

                elif short_signal > 0:
                    diff = short_signal * self.unit + abs(cur_pos)
                    self.sell(size=diff, limit=self.data.Close[-1])

                else:
                    for trade in self.trades:
                        trade.close()

            elif self.position.is_short:
                if short_signal > 0:
                    diff = short_signal * self.unit - abs(cur_pos)
                    if diff > 0:
                        self.sell(size=diff, limit=self.data.Close[-1])
                    elif diff < 0:
                        self.buy(size=diff, limit=self.data.Close[-1])

                elif long_signal > 0:
                    diff = long_signal * self.unit + abs(cur_pos)
                    self.buy(size=diff, limit=self.data.Close[-1])
                else:
                    for trade in self.trades:
                        trade.close()
            else:
                if long_signal > 0:
                    diff = long_signal * self.unit
                    self.buy(size=diff, limit=self.data.Close[-1])

                elif short_signal > 0:
                    diff = short_signal * self.unit
                    self.sell(size=diff, limit=self.data.Close[-1])
