import numpy as np
import talib
from backtesting import Strategy
from backtesting.lib import cross, resample_apply, crossover

from research.strategies.strategy_utils import *


class AutoMLStrategy(Strategy):
    """
    range_parameter = {"type": "range", "bounds": (min, max)}
    choice_parameter = {"type": "choice", "values": [], "is_ordered": False, "name": "string"}
    """

    parameter_specs = []
    fixed_parameters = []
    parameters = []
    hparams = {}
    constraints = []
    side = "BOTH"
    risk_level = -1

    def __init__(self, broker, data, params):
        super().__init__(broker, data, {})
        self.hparams = params
        if "side" in params:
            self.side = params["side"]



class AutoML_SUPER_TREND(AutoMLStrategy):
    parameter_specs = [
        {"name": "risk_factor", "type": "range", "bounds": (0.5, 5)},
        {
            "name": "atr_window",
            "type": "choice",
            "values": list(range(5, 40)),
            "is_ordered": True,
        },
        {
            "name": "atr_multiplier",
            "type": "range",
            "bounds": (0.1, 5),
        },
        {
            "name": "fast_dema_window",
            "type": "choice",
            "values": list(range(5, 300)),
            "is_ordered": True,
        },
        {
            "name": "slow_dema_window",
            "type": "choice",
            "values": list(range(5, 500)),
            "is_ordered": True,
        },
    ]
    parameters = [
        "risk_factor",
        "atr_window",
        "atr_multiplier",
        "fast_dema_window",
        "slow_dema_window",
    ]
    constraints = ["slow_dema_window > fast_dema_window"]
    fixed_parameters = ["side", "max_sl"]

    def init(self):
        self.min_profit_multiplier = 2
        self.risk_factor = int(self.hparams.get("risk_factor", 3.674882930517197))
        self.atr_window = int(self.hparams.get("atr_window", 26))
        self.atr_multiplier = int(
            self.hparams.get("atr_multiplier", 0.37722657360136513)
        )
        self.fast_dema_window = int(self.hparams.get("fast_dema_window", 182))
        self.slow_dema_window = int(self.hparams.get("slow_dema_window", 269))
        self.long_ma = resample_apply(
            "4H", get_ma, self.data.Close, 60, name="LONG MA", plot=True
        )
        self.short_ma = resample_apply(
            "4H", get_ma, self.data.Close, 15, name="SHORT MA", plot=True
        )

        self.side = self.hparams.get("side", "BOTH")
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
        self.MA_L[-1] = (
            self.data.Close[-1] > max(self.fast_ma[-1], self.slow_ma[-1])
            and self.short_ma[-1] > self.long_ma[-1]
        )
        self.MA_S[-1] = (
            self.data.Close[-1] < min(self.fast_ma[-1], self.slow_ma[-1])
            and self.short_ma[-1] > self.long_ma[-1]
        )

        self.TREND_L[-1] = self.super_trend[-1] > self.super_trend[-2]
        self.TREND_S[-1] = self.super_trend[-1] < self.super_trend[-2]

        LONG = self.TREND_L[-1] and self.MA_L[-1]
        SHORT = self.TREND_S[-1] and self.MA_S[-1]
        if not self.position:
            if LONG and self.side != "SHORT_ONLY":
                sl = min(
                    self.fast_ma[-1],
                    self.slow_ma[-1],
                )
                tp = (
                    self.risk_factor * abs(self.data.Close[-1] - sl)
                    + self.data.Close[-1]
                )
                max_loss_rate = abs(self.data.Close[-1] - sl) / self.data.Close[-1]
                max_pos = max(min(0.99, self.max_sl / max_loss_rate), 0.01)
                if (abs(tp - self.data.Close[-1]) / self.data.Close[-1]) >= (
                    self.min_profit_multiplier * self._broker._commission
                ):
                    if tp > self.data.Close[-1] > sl > 0:
                        self.buy(tp=tp, sl=sl, limit=self.data.Close[-1], size=max_pos)

            elif SHORT and self.side != "LONG_ONLY":
                sl = max(
                    self.fast_ma[-1],
                    self.slow_ma[-1],
                )
                tp = self.data.Close[-1] - self.risk_factor * abs(
                    self.data.Close[-1] - sl
                )
                max_loss_rate = abs(self.data.Close[-1] - sl) / self.data.Close[-1]
                max_pos = max(min(0.99, self.max_sl / max_loss_rate), 0.01)
                if (abs(tp - self.data.Close[-1]) / self.data.Close[-1]) >= (
                    self.min_profit_multiplier * self._broker._commission
                ):
                    if 0 < tp < self.data.Close[-1] < sl:
                        self.sell(tp=tp, sl=sl, limit=self.data.Close[-1], size=max_pos)

        elif self.position.is_long and SHORT:
            for trade in self.trades:
                trade.close()

        elif self.position.is_short and LONG:
            for trade in self.trades:
                trade.close()


class AutoML_Swing(AutoMLStrategy):
    parameter_specs = [
        {
            "name": "adx_window",
            "type": "choice",
            "values": list(range(5, 100)),
            "is_ordered": True,
        },
        {
            "name": "adx_th",
            "type": "range",
            "bounds": (1, 100),
        },
        {
            "name": "volume_ma_window",
            "type": "choice",
            "values": list(range(2, 50)),
            "is_ordered": True,
        },
        {
            "name": "mad_window",
            "type": "choice",
            "values": list(range(2, 16)),
            "is_ordered": True,
        },
        {
            "name": "mad_gamma",
            "type": "range",
            "bounds": (-0.1, 10),
        },
        {
            "name": "rsi_window",
            "type": "choice",
            "values": list(range(2, 100)),
            "is_ordered": True,
        },
        {
            "name": "macd_signal_window",
            "type": "choice",
            "values": list(range(2, 100)),
            "is_ordered": True,
        },
        {
            "name": "vwap_rsi_window",
            "type": "choice",
            "values": list(range(2, 100)),
            "is_ordered": True,
        },
        {
            "name": "sl",
            "type": "range",
            "bounds": (0.001, 0.08),
        },
        {
            "name": "tp",
            "type": "range",
            "bounds": (0.001, 0.08),
        },
    ]
    parameters = [
        "adx_window",
        "adx_th",
        # "mad_window",
        # "mad_gamma",
        "sl",
        "tp",
        # "volume_ma_window",
        # "volume_f",
        # "rsi_window",
        # "macd_fast_window",
        # "macd_slow_window",
        # "macd_signal_window",
        "vwap_rsi_window",
    ]
    constraints = ["sl > tp"]
    fixed_parameters = ["side", "max_sl"]

    def init(self):
        self.adx_window = int(self.hparams.get("adx_window", 11))
        self.adx_th = float(self.hparams.get("adx_th", 30))
        self.mad_window = int(self.hparams.get("mad_window", 5))
        self.mad_gamma = float(self.hparams.get("mad_gamma", 4.0))
        self.volume_ma_window = float(self.hparams.get("volume_ma_window", 31))
        self.volume_f = float(self.hparams.get("volume_f", 1.6))
        self.rsi_window = float(self.hparams.get("rsi_window", 61))
        self.macd_fast_window = float(self.hparams.get("macd_fast_window", 5))
        self.macd_slow_window = float(self.hparams.get("macd_slow_window", 12))
        self.macd_signal_window = float(self.hparams.get("macd_signal_window", 8))
        self.vwap_rsi_window = float(self.hparams.get("vwap_rsi_window", 14))
        self.sl = float(self.hparams.get("sl", 0.04))
        self.tp = float(self.hparams.get("tp", 0.015))
        self.atr_window = 48
        self.mad = resample_apply(
            "4H",
            get_autonomous_recursive_ma,
            (self.data.High + self.data.Low + self.data.Close) / 3,
            self.mad_window,
            self.mad_gamma,
            zero_lag=True,
            plot=True,
            overlay=True,
        )

        plus, minus, adx = calcADX(
            self.data.High, self.data.Low, self.data.Close, self.adx_window
        )

        self.TREND_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="TREND SIGNAL"
        )
        self.REVERSE_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="REVERSE_SIGNAL"
        )
        self.MAD_FLAT_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MAD FLAT"
        )
        self.ADX_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ADX_SIGNAL"
        )
        self.SAR_DIRECTION = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="SAR_DIRECTION"
        )

        self.VOLUME_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VOLUME_FILTER"
        )
        self.RSI_SIGNAL_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="RSI_SIGNAL_FILTER"
        )
        self.VWAP_RSI_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VWAP_RSI_FILTER"
        )
        self.MACD_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MACD_SIGNAL"
        )
        self.plus = self.I(equal_map, plus, name="adx_plus", plot=True)
        self.minus = self.I(equal_map, minus, name="adx_minus", plot=True)
        self.adx = self.I(equal_map, adx, name="adx", plot=True)

        self.sar = self.I(
            talib.SAR, self.data.High, self.data.Low, name="SAR", plot=True
        )
        self.volume_ma = self.I(
            get_ma, self.data.Volume, self.volume_ma_window, plot=True, overlay=False
        )
        self.rsi = self.I(
            get_rsi, self.data.Close, self.rsi_window, plot=True, overlay=False
        )
        dif, slow_dif, _ = talib.MACD(
            self.data.Close,
            self.macd_fast_window,
            self.macd_slow_window,
            self.macd_signal_window,
        )

        self.macd = self.I(equal_map, dif, name="MACD", plot=True, overlay=False)
        self.slow_macd = self.I(
            equal_map, slow_dif, name="MACD SIGNAL", plot=True, overlay=False
        )

        self.vwap = self.I(
            get_vwap,
            (self.data.High + self.data.Low + self.data.Close) / 3,
            self.data.Volume,
            48,
            name="VWAP",
            plot=True,
            overlay=True,
        )
        self.vwap_rsi = self.I(
            get_rsi,
            self.vwap,
            self.vwap_rsi_window,
            plot=True,
            overlay=False,
            name="vwap_rsi",
        )
        self.atr = self.I(
            get_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_window,
            normalize=True,
            plot=True,
            overlay=False,
            name="ATR",
        )

    def next(self):
        self.MAD_FLAT_FILTER[-1] = abs(self.mad[-1] / self.mad[-2] - 1) > 0.001
        if self.plus[-1] > self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = 1
        elif self.plus[-1] < self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = -1

        if self.data.Close[-1] > self.sar[-1]:
            self.SAR_DIRECTION[-1] = 1.0

        elif self.data.Close[-1] < self.sar[-1]:
            self.SAR_DIRECTION[-1] = -1.0

        self.VOLUME_FILTER[-1] = (
            self.data.Volume[-1] > self.volume_f * self.volume_ma[-1]
        )
        if self.rsi[-1] < 70:
            self.RSI_SIGNAL_FILTER[-1] = 1.0

        elif self.rsi[-1] > 30:
            self.RSI_SIGNAL_FILTER[-1] = -1.0

        if 80 > self.vwap_rsi[-1] > self.vwap_rsi[-2] > 20:
            self.VWAP_RSI_FILTER[-1] = 1.0

        elif 20 < self.vwap_rsi[-1] < self.vwap_rsi[-2] < 80:
            self.VWAP_RSI_FILTER[-1] = -1.0

        if self.macd[-1] > self.slow_macd[-1]:
            self.MACD_SIGNAL[-1] = 1.0

        elif self.macd[-1] < self.slow_macd[-1]:
            self.MACD_SIGNAL[-1] = -1.0

        if (
            self.MAD_FLAT_FILTER[-1] == 0
            and self.SAR_DIRECTION[-1] > 0
            and self.VWAP_RSI_FILTER[-1] > 0
        ):
            self.TREND_SIGNAL[-1] = 1.0

        elif (
            self.MAD_FLAT_FILTER[-1] == 0
            and self.SAR_DIRECTION[-1] < 0
            and self.VWAP_RSI_FILTER[-1] < 0
        ):
            self.TREND_SIGNAL[-1] = -1.0

        if (
            self.ADX_SIGNAL[-1] > 0
            and self.SAR_DIRECTION[-1] > 0
            and self.VOLUME_FILTER[-1] > 0
            and self.RSI_SIGNAL_FILTER[-1] > 0
            and self.MACD_SIGNAL[-1] > 0
            and self.MAD_FLAT_FILTER[-1] == 0
        ):
            self.REVERSE_SIGNAL[-1] = 1.0

        elif (
            self.ADX_SIGNAL[-1] > 0
            and self.SAR_DIRECTION[-1] > 0
            and self.VOLUME_FILTER[-1] > 0
            and self.RSI_SIGNAL_FILTER[-1] > 0
            and self.MACD_SIGNAL[-1] > 0
            and self.MAD_FLAT_FILTER[-1] == 0
        ):
            self.REVERSE_SIGNAL[-1] = -1.0

        # exit point
        LONG = (
            self.TREND_SIGNAL[-1]
            > 0
            # or
            # self.REVERSE_SIGNAL[-1] > 0
        )

        SHORT = (
            self.TREND_SIGNAL[-1]
            < 0
            # or
            # self.REVERSE_SIGNAL[-1] < 0
        )
        LONG_EXIT = self.SAR_DIRECTION[-1] < 0
        SHORT_EXIT = self.SAR_DIRECTION[-1] > 0

        if not self.position:
            if LONG and self.side != "SHORT_ONLY":
                sl = self.data.Close[-1] * (1 - self.sl)
                tp = self.data.Close[-1] * (1 + self.tp)
                self.buy(tp=tp, limit=self.data.Close[-1], size=0.99, sl=sl)

            elif SHORT and self.side != "LONG_ONLY":
                sl = self.data.Close[-1] * (1 + self.sl)
                tp = self.data.Close[-1] * (1 - self.tp)
                self.sell(tp=tp, limit=self.data.Close[-1], size=0.99, sl=sl)


class AutoML_Swing_Test(AutoMLStrategy):
    parameter_specs = [
        {
            "name": "adx_window",
            "type": "choice",
            "values": list(range(5, 100)),
            "is_ordered": True,
        },
        {
            "name": "adx_th",
            "type": "range",
            "bounds": (1, 100),
        },
        {
            "name": "volume_ma_window",
            "type": "choice",
            "values": list(range(2, 50)),
            "is_ordered": True,
        },
        {
            "name": "mad_window",
            "type": "choice",
            "values": list(range(4, 16)),
            "is_ordered": True,
        },
        {
            "name": "mad_gamma",
            "type": "range",
            "bounds": (2, 5),
        },
        {
            "name": "rsi_window",
            "type": "choice",
            "values": list(range(2, 100)),
            "is_ordered": True,
        },
        {
            "name": "macd_signal_window",
            "type": "choice",
            "values": list(range(2, 100)),
            "is_ordered": True,
        },
        {
            "name": "vwap_rsi_window",
            "type": "choice",
            "values": list(range(2, 100)),
            "is_ordered": True,
        },
        {
            "name": "atr_window",
            "type": "choice",
            "values": list(range(12, 96)),
            "is_ordered": True,
        },
        {
            "name": "sl",
            "type": "range",
            "bounds": (0.001, 0.08),
        },
        {
            "name": "tp",
            "type": "range",
            "bounds": (0.001, 0.08),
        },
        {
            "name": "sl_atr_rate",
            "type": "range",
            "bounds": (0.1, 1.0),
        },
        {
            "name": "sar_max",
            "type": "range",
            "bounds": (0.1, 5.0),
        },
        {
            "name": "sar_acceleration",
            "type": "range",
            "bounds": (0.01, 2.0),
        },
    ]
    parameters = [
        # "adx_window",
        # "adx_th",
        "mad_window",
        "mad_gamma",
        # "sl",
        # "tp",
        # "volume_ma_window",
        # "volume_f",
        # "rsi_window",
        # "macd_fast_window",
        # "macd_slow_window",
        # "macd_signal_window",
        "vwap_rsi_window",
        "sl_atr_rate",
        "atr_window",
        "sar_acceleration",
        "sar_max",
    ]
    constraints = ["sar_max > sar_acceleration"]
    fixed_parameters = ["side", "max_sl"]

    def init(self):
        self.adx_window = int(self.hparams.get("adx_window", 11))
        self.adx_th = float(self.hparams.get("adx_th", 30))
        self.mad_window = int(self.hparams.get("mad_window", 13))
        self.mad_gamma = float(self.hparams.get("mad_gamma", 3))
        self.volume_ma_window = float(self.hparams.get("volume_ma_window", 31))
        self.volume_f = float(self.hparams.get("volume_f", 1.6))
        self.rsi_window = float(self.hparams.get("rsi_window", 61))
        self.macd_fast_window = float(self.hparams.get("macd_fast_window", 5))
        self.macd_slow_window = float(self.hparams.get("macd_slow_window", 12))
        self.macd_signal_window = float(self.hparams.get("macd_signal_window", 8))
        self.vwap_rsi_window = float(self.hparams.get("vwap_rsi_window", 14))
        self.sl = float(self.hparams.get("sl", 0.025))
        self.sl_atr_rate = float(self.hparams.get("sl_atr_rate", 1.2))
        self.tp = float(self.hparams.get("tp", 0.04))
        self.atr_window = int(self.hparams.get("atr_window", 48))
        self.sar_acceleration = int(self.hparams.get("sar_acceleration", 0.3))
        self.sar_max = int(self.hparams.get("sar_max", 0.35))
        self.profit_factor = int(self.hparams.get("profit_factor", 1.0))
        self.mad = self.I(
            get_autonomous_recursive_ma,
            # (self.data.High + self.data.Low + self.data.Close) / 3,
            self.data.Close,
            self.mad_window,
            self.mad_gamma,
            zero_lag=True,
            plot=True,
            overlay=True,
        )

        plus, minus, adx = calcADX(
            self.data.High, self.data.Low, self.data.Close, self.adx_window
        )

        self.TREND_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="TREND SIGNAL"
        )
        self.REVERSE_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="REVERSE_SIGNAL"
        )
        self.MAD_FLAT_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MAD FLAT"
        )
        self.ADX_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ADX_SIGNAL"
        )
        self.SAR_DIRECTION = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="SAR_DIRECTION"
        )

        self.VOLUME_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VOLUME_FILTER"
        )
        self.RSI_SIGNAL_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="RSI_SIGNAL_FILTER"
        )
        self.VWAP_RSI_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VWAP_RSI_FILTER"
        )
        self.MACD_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MACD_SIGNAL"
        )
        self.plus = self.I(equal_map, plus, name="adx_plus", plot=True)
        self.minus = self.I(equal_map, minus, name="adx_minus", plot=True)
        self.adx = self.I(equal_map, adx, name="adx", plot=True)

        self.sar = self.I(
            talib.SAR,
            self.data.High,
            self.data.Low,
            self.sar_acceleration,
            self.sar_max,
            name="SAR",
            plot=True,
        )
        self.volume_ma = self.I(
            get_ma, self.data.Volume, self.volume_ma_window, plot=True, overlay=False
        )
        self.rsi = self.I(
            get_rsi, self.data.Close, self.rsi_window, plot=True, overlay=False
        )

        dif, slow_dif, _ = talib.MACD(
            self.data.Close,
            self.macd_fast_window,
            self.macd_slow_window,
            self.macd_signal_window,
        )

        self.macd = self.I(equal_map, dif, name="MACD", plot=True, overlay=False)
        self.slow_macd = self.I(
            equal_map, slow_dif, name="MACD SIGNAL", plot=True, overlay=False
        )

        self.vwap = self.I(
            get_vwap,
            (self.data.High + self.data.Low + self.data.Close) / 3,
            self.data.Volume,
            48,
            plot=True,
            overlay=True,
        )
        self.vwap_rsi = self.I(
            get_rsi,
            self.vwap,
            self.vwap_rsi_window,
            plot=True,
            overlay=False,
            name="vwap_rsi",
        )
        self.atr = self.I(
            get_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_window,
            normalize=True,
            plot=True,
            overlay=False,
            name="ATR",
        )

    def next(self):
        self.MAD_FLAT_FILTER[-1] = abs(self.mad[-1] / self.mad[-2] - 1) < 0.001
        if self.plus[-1] > self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = 1
        elif self.plus[-1] < self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = -1

        if self.data.Close[-1] > self.sar[-1]:
            self.SAR_DIRECTION[-1] = 1.0

        elif self.data.Close[-1] < self.sar[-1]:
            self.SAR_DIRECTION[-1] = -1.0

        self.VOLUME_FILTER[-1] = (
            self.data.Volume[-1] > self.volume_f * self.volume_ma[-1]
        )
        if self.rsi[-1] < 70:
            self.RSI_SIGNAL_FILTER[-1] = 1.0

        elif self.rsi[-1] > 30:
            self.RSI_SIGNAL_FILTER[-1] = -1.0

        if 80 > self.vwap_rsi[-1] > self.vwap_rsi[-2] > 20:
            self.VWAP_RSI_FILTER[-1] = 1.0

        elif 20 < self.vwap_rsi[-1] < self.vwap_rsi[-2] < 80:
            self.VWAP_RSI_FILTER[-1] = -1.0

        if self.macd[-1] > self.slow_macd[-1]:
            self.MACD_SIGNAL[-1] = 1.0

        elif self.macd[-1] < self.slow_macd[-1]:
            self.MACD_SIGNAL[-1] = -1.0

        if (
            self.MAD_FLAT_FILTER[-1] > 0
            and self.SAR_DIRECTION[-1] > 0
            and self.VWAP_RSI_FILTER[-1] > 0
            and
            # self.vwap[-1] > self.vwap_long[-1]
            self.data.Close[-1] < self.mad[-1]
        ):
            self.TREND_SIGNAL[-1] = 1.0

        elif (
            self.MAD_FLAT_FILTER[-1] > 0
            and self.SAR_DIRECTION[-1] < 0
            and self.VWAP_RSI_FILTER[-1] < 0
            and self.data.Close[-1] > self.mad[-1]
        ):
            self.TREND_SIGNAL[-1] = -1.0

        if (
            self.ADX_SIGNAL[-1] > 0
            and self.SAR_DIRECTION[-1] > 0
            and self.VOLUME_FILTER[-1] > 0
            and self.RSI_SIGNAL_FILTER[-1] > 0
            and self.MACD_SIGNAL[-1] > 0
            and self.MAD_FLAT_FILTER[-1] == 0
        ):
            self.REVERSE_SIGNAL[-1] = 1.0

        elif (
            self.ADX_SIGNAL[-1] > 0
            and self.SAR_DIRECTION[-1] > 0
            and self.VOLUME_FILTER[-1] > 0
            and self.RSI_SIGNAL_FILTER[-1] > 0
            and self.MACD_SIGNAL[-1] > 0
            and self.MAD_FLAT_FILTER[-1] == 0
        ):
            self.REVERSE_SIGNAL[-1] = -1.0

        # exit point
        LONG = (
            self.TREND_SIGNAL[-1]
            > 0
            # or
            # self.REVERSE_SIGNAL[-1] > 0
        )

        SHORT = (
            self.TREND_SIGNAL[-1]
            < 0
            # or
            # self.REVERSE_SIGNAL[-1] < 0
        )

        if not self.position:
            if LONG and self.side != "SHORT_ONLY":
                # sl = self.data.Close[-1] * (1 - self.sl)
                sl = self.data.Close[-1] * (1 - self.sl_atr_rate * self.atr[-1])
                # tp = self.data.Close[-1] * (1 + self.tp)
                self.buy(size=0.99, sl=sl)

            elif SHORT and self.side != "LONG_ONLY":
                sl = self.data.Close[-1] * (1 + self.sl_atr_rate * self.atr[-1])
                # tp = self.data.Close[-1] * (1 - self.tp)
                self.sell(size=0.99, sl=sl)
        else:
            for trade in self.trades:
                if self.position.is_long:
                    sl = self.data.Close[-1] * (1 - self.sl_atr_rate * self.atr[-1])
                    trade.sl = max(sl, trade.sl)
                else:
                    sl = self.data.Close[-1] * (1 + self.sl_atr_rate * self.atr[-1])
                    trade.sl = min(sl, trade.sl)

        # elif self.position.is_long and LONG_EXIT:
        #     for trade in self.trades:
        #         trade.close()
        # elif self.position.is_short and SHORT_EXIT:
        #     for trade in self.trades:
        #         trade.close()


class AutoML_Swing_Test2(AutoMLStrategy):
    parameter_specs = [
        {
            "name": "adx_window",
            "type": "choice",
            "values": list(range(5, 100)),
            "is_ordered": True,
        },
        {
            "name": "adx_th",
            "type": "range",
            "bounds": (1, 100),
        },
        {
            "name": "volume_ma_window",
            "type": "choice",
            "values": list(range(2, 50)),
            "is_ordered": True,
        },
        {
            "name": "mad_window",
            "type": "choice",
            "values": list(range(4, 16)),
            "is_ordered": True,
        },
        {
            "name": "mad_gamma",
            "type": "range",
            "bounds": (2, 5),
        },
        {
            "name": "rsi_window",
            "type": "choice",
            "values": list(range(2, 100)),
            "is_ordered": True,
        },
        {
            "name": "macd_signal_window",
            "type": "choice",
            "values": list(range(2, 100)),
            "is_ordered": True,
        },
        {
            "name": "vwap_rsi_window",
            "type": "choice",
            "values": list(range(2, 100)),
            "is_ordered": True,
        },
        {
            "name": "atr_window",
            "type": "choice",
            "values": list(range(12, 96)),
            "is_ordered": True,
        },
        {
            "name": "sl",
            "type": "range",
            "bounds": (0.001, 0.08),
        },
        {
            "name": "tp",
            "type": "range",
            "bounds": (0.001, 0.08),
        },
        {
            "name": "sl_atr_rate",
            "type": "range",
            "bounds": (0.1, 1.0),
        },
        {
            "name": "tp_atr_rate",
            "type": "range",
            "bounds": (0.1, 1.0),
        },
        {
            "name": "sar_max",
            "type": "range",
            "bounds": (0.1, 5.0),
        },
        {
            "name": "sar_acceleration",
            "type": "range",
            "bounds": (0.01, 2.0),
        },
    ]
    parameters = [
        # "adx_window",
        # "adx_th",
        "mad_window",
        "mad_gamma",
        # "sl",
        # "tp",
        # "volume_ma_window",
        # "volume_f",
        # "rsi_window",
        # "macd_fast_window",
        # "macd_slow_window",
        # "macd_signal_window",
        "vwap_rsi_window",
        "sl_atr_rate",
        "tp_atr_rate",
        "atr_window",
        "sar_acceleration",
        "sar_max",
    ]
    constraints = []
    fixed_parameters = ["side", "max_sl"]

    def init(self):
        self.adx_window = int(self.hparams.get("adx_window", 11))
        self.adx_th = float(self.hparams.get("adx_th", 30))
        self.mad_window = int(self.hparams.get("mad_window", 13))
        self.mad_gamma = float(self.hparams.get("mad_gamma", 3))
        self.volume_ma_window = float(self.hparams.get("volume_ma_window", 31))
        self.volume_f = float(self.hparams.get("volume_f", 1.6))
        self.rsi_window = float(self.hparams.get("rsi_window", 61))
        self.macd_fast_window = float(self.hparams.get("macd_fast_window", 5))
        self.macd_slow_window = float(self.hparams.get("macd_slow_window", 12))
        self.macd_signal_window = float(self.hparams.get("macd_signal_window", 8))
        self.vwap_rsi_window = float(self.hparams.get("vwap_rsi_window", 14))
        self.sl = float(self.hparams.get("sl", 0.025))
        self.sl_atr_rate = float(self.hparams.get("sl_atr_rate", 1.2))
        self.tp_atr_rate = float(self.hparams.get("tp_atr_rate", 1.2))
        self.tp = float(self.hparams.get("tp", 0.04))
        self.atr_window = int(self.hparams.get("atr_window", 48))
        self.sar_acceleration = int(self.hparams.get("sar_acceleration", 0.3))
        self.sar_max = int(self.hparams.get("sar_max", 0.35))
        self.profit_factor = int(self.hparams.get("profit_factor", 1.0))
        self.mad = self.I(
            get_autonomous_recursive_ma,
            # (self.data.High + self.data.Low + self.data.Close) / 3,
            self.data.Close,
            self.mad_window,
            self.mad_gamma,
            zero_lag=True,
            plot=True,
            overlay=True,
        )

        plus, minus, adx = calcADX(
            self.data.High, self.data.Low, self.data.Close, self.adx_window
        )

        self.TREND_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="TREND SIGNAL"
        )
        self.REVERSE_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="REVERSE_SIGNAL"
        )
        self.MAD_FLAT_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MAD FLAT"
        )
        self.ADX_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ADX_SIGNAL"
        )
        self.SAR_DIRECTION = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="SAR_DIRECTION"
        )

        self.VOLUME_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VOLUME_FILTER"
        )
        self.RSI_SIGNAL_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="RSI_SIGNAL_FILTER"
        )
        self.VWAP_RSI_FILTER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VWAP_RSI_FILTER"
        )
        self.MACD_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MACD_SIGNAL"
        )
        self.plus = self.I(equal_map, plus, name="adx_plus", plot=True)
        self.minus = self.I(equal_map, minus, name="adx_minus", plot=True)
        self.adx = self.I(equal_map, adx, name="adx", plot=True)

        self.sar = self.I(
            talib.SAR,
            self.data.High,
            self.data.Low,
            self.sar_acceleration,
            self.sar_max,
            name="SAR",
            plot=True,
        )
        self.volume_ma = self.I(
            get_ma, self.data.Volume, self.volume_ma_window, plot=True, overlay=False
        )
        self.rsi = self.I(
            get_rsi, self.data.Close, self.rsi_window, plot=True, overlay=False
        )
        dif, slow_dif, _ = talib.MACD(
            self.data.Close,
            self.macd_fast_window,
            self.macd_slow_window,
            self.macd_signal_window,
        )

        self.macd = self.I(equal_map, dif, name="MACD", plot=True, overlay=False)
        self.slow_macd = self.I(
            equal_map, slow_dif, name="MACD SIGNAL", plot=True, overlay=False
        )

        self.vwap = self.I(
            get_vwap,
            (self.data.High + self.data.Low + self.data.Close) / 3,
            self.data.Volume,
            48,
            name="VWAP",
            plot=True,
            overlay=True,
        )
        self.vwap_long = self.I(
            get_vwap,
            (self.data.High + self.data.Low + self.data.Close) / 3,
            self.data.Volume,
            102,
            name="VWAP",
            plot=True,
            overlay=True,
        )
        self.vwap_rsi = self.I(
            get_rsi,
            self.vwap,
            self.vwap_rsi_window,
            plot=True,
            overlay=False,
            name="vwap_rsi",
        )
        self.atr = self.I(
            get_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_window,
            # normalize=True,
            plot=True,
            overlay=False,
            name="ATR",
        )

    def next(self):
        self.MAD_FLAT_FILTER[-1] = abs(self.mad[-1] / self.mad[-2] - 1) < 0.001
        if self.plus[-1] > self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = 1
        elif self.plus[-1] < self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = -1

        if self.data.Close[-1] > self.sar[-1]:
            self.SAR_DIRECTION[-1] = 1.0

        elif self.data.Close[-1] < self.sar[-1]:
            self.SAR_DIRECTION[-1] = -1.0

        self.VOLUME_FILTER[-1] = (
            self.data.Volume[-1] > self.volume_f * self.volume_ma[-1]
        )
        if self.rsi[-1] < 70:
            self.RSI_SIGNAL_FILTER[-1] = 1.0

        elif self.rsi[-1] > 30:
            self.RSI_SIGNAL_FILTER[-1] = -1.0

        if 80 > self.vwap_rsi[-1] > self.vwap_rsi[-2] > 20:
            self.VWAP_RSI_FILTER[-1] = 1.0

        elif 20 < self.vwap_rsi[-1] < self.vwap_rsi[-2] < 80:
            self.VWAP_RSI_FILTER[-1] = -1.0

        if self.macd[-1] > self.slow_macd[-1]:
            self.MACD_SIGNAL[-1] = 1.0

        elif self.macd[-1] < self.slow_macd[-1]:
            self.MACD_SIGNAL[-1] = -1.0

        if (
            self.MAD_FLAT_FILTER[-1] > 0
            and self.SAR_DIRECTION[-1] > 0
            and self.VWAP_RSI_FILTER[-1] > 0
            and
            # self.vwap[-1] > self.vwap_long[-1]
            self.data.Close[-1] < self.mad[-1]
        ):
            self.TREND_SIGNAL[-1] = 1.0

        elif (
            self.MAD_FLAT_FILTER[-1] > 0
            and self.SAR_DIRECTION[-1] < 0
            and self.VWAP_RSI_FILTER[-1] < 0
            and
            # self.vwap[-1] < self.vwap_long[-1] # try another close < mad
            self.data.Close[-1] > self.mad[-1]
        ):
            self.TREND_SIGNAL[-1] = -1.0

        if (
            self.ADX_SIGNAL[-1] > 0
            and self.SAR_DIRECTION[-1] > 0
            and self.VOLUME_FILTER[-1] > 0
            and self.RSI_SIGNAL_FILTER[-1] > 0
            and self.MACD_SIGNAL[-1] > 0
            and self.MAD_FLAT_FILTER[-1] == 0
        ):
            self.REVERSE_SIGNAL[-1] = 1.0

        elif (
            self.ADX_SIGNAL[-1] > 0
            and self.SAR_DIRECTION[-1] > 0
            and self.VOLUME_FILTER[-1] > 0
            and self.RSI_SIGNAL_FILTER[-1] > 0
            and self.MACD_SIGNAL[-1] > 0
            and self.MAD_FLAT_FILTER[-1] == 0
        ):
            self.REVERSE_SIGNAL[-1] = -1.0

        # exit point
        LONG = (
            self.TREND_SIGNAL[-1]
            > 0
            # or
            # self.REVERSE_SIGNAL[-1] > 0
        )

        SHORT = (
            self.TREND_SIGNAL[-1]
            < 0
            # or
            # self.REVERSE_SIGNAL[-1] < 0
        )

        if not self.position:
            if LONG and self.side != "SHORT_ONLY":
                # sl = self.data.Close[-1] * (1 - self.sl)
                sl = self.data.Close[-1] - self.sl_atr_rate * self.atr[-1]
                tp = self.data.Close[-1] + self.tp_atr_rate * self.atr[-1]
                if tp > self.data.Close[-1] > sl > 0:
                    self.buy(size=0.99, sl=sl, tp=tp, limit=self.data.Close[-1])

            elif SHORT and self.side != "LONG_ONLY":
                sl = self.data.Close[-1] + self.sl_atr_rate * self.atr[-1]
                tp = self.data.Close[-1] - self.tp_atr_rate * self.atr[-1]
                if 0 < tp < self.data.Close[-1] < sl:
                    self.sell(size=0.99, sl=sl, tp=tp, limit=self.data.Close[-1])
