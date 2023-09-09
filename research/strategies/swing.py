from backtesting.lib import crossover

from research.strategies.automl_strategies import *


class Swing(AutoMLStrategy):
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
        {
            "name": "vwap_rsi_window",
            "type": "choice",
            "values": list(range(15, 50)),
            "is_ordered": True,
        },
    ]
    parameters = [
        "sl",
        "tp1",
        "vwap_rsi_window",
        "adx_th",
        "adx_window",
        "volume_f",
        "left",
        "right",
    ]
    constraints = []
    fixed_parameters = ["side", "max_sl"]

    def init(self):
        self.Sacc = float(self.hparams.get("Sacc", 0.3))
        self.Smax = float(self.hparams.get("Smax", 0.35))
        self.vwap_rsi_window = int(self.hparams.get("vwap_rsi_window", 14))
        self.vwap_rsi_oversold = float(self.hparams.get("vwap_rsi_oversold", 29))
        self.vwap_rsi_overbought = float(self.hparams.get("vwap_rsi_overbought", 75))
        self.mad_window = int(self.hparams.get("mad_window", 13))
        self.mad_gamma = float(self.hparams.get("mad_gamma", 3.0))
        self.sl = float(self.hparams.get("sl", 0.04))
        self.tp1 = float(self.hparams.get("tp1", 0.015))
        self.tp2 = float(self.hparams.get("tp2", 0.045))
        self.tp3 = float(self.hparams.get("tp3", 0.014))

        self.mad = self.I(
            get_autonomous_recursive_ma,
            self.data.Close,
            self.mad_window,
            self.mad_gamma,
            plot=True,
            overlay=True,
        )

        self.sar = self.I(
            talib.SAR,
            self.data.High,
            self.data.Low,
            self.Sacc,
            self.Smax,
            name="SAR",
            plot=True,
        )

        self.vwap = self.I(
            get_vwap,
            self.data.Close,
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

        self.SAR_DIRECTION = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="SAR_DIRECTION"
        )
        self.VWAP_RSI_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VWAP_RSI_SIGNAL"
        )
        self.MAD_FLAT = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MAD_FLAT"
        )
        self.SAR_ENTRY_1 = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="SAR_ENTRY"
        )

    def next(self):
        if abs(self.mad[-1] / self.mad[-2]) < 1.01:
            self.MAD_FLAT[-1] = 1

        if (
            self.vwap_rsi[-2] < self.vwap_rsi_oversold
            and self.vwap_rsi[-1] > self.vwap_rsi_oversold
        ):
            self.VWAP_RSI_SIGNAL[-1] = 1

        elif (
            self.vwap_rsi[-2] > self.vwap_rsi_overbought
            and self.vwap_rsi[-1] < self.vwap_rsi_overbought
        ):
            self.VWAP_RSI_SIGNAL[-1] = -1

        if self.sar[-1] < self.data.Close[-1]:
            self.SAR_DIRECTION[-1] = 1

        elif self.sar[-1] > self.data.Close[-1]:
            self.SAR_DIRECTION[-1] = -1

        if (
            self.VWAP_RSI_SIGNAL[-1] > 0
            and self.SAR_DIRECTION[-1] > 0
            and self.MAD_FLAT > 0
        ):
            self.SAR_ENTRY_1[-1] = 1

        elif (
            self.VWAP_RSI_SIGNAL[-1] < 0
            and self.SAR_DIRECTION[-1] < 0
            and self.MAD_FLAT > 0
        ):
            self.SAR_ENTRY_1[-1] = -1

        LONG = self.SAR_ENTRY_1[-1] > 0

        SHORT = self.SAR_ENTRY_1[-1] < 0
        if not self.position:
            if LONG and self.side != "SHORT_ONLY":
                # sl = self.data.Close[-1] * (1 - self.sl)
                sl = self.data.Close[-1] * (1 - self.sl)
                tp1 = self.data.Close[-1] * (1 + self.tp1)
                tp2 = self.data.Close[-1] * (1 + self.tp2)
                tp3 = self.data.Close[-1] * (1 + self.tp3)
                max_amount = self._broker._cash / self.data.Close[-1] // 3
                self.buy(size=0.99, sl=sl, tp=tp1, limit=self.data.Close[-1])

            elif SHORT and self.side != "LONG_ONLY":
                max_amount = self._broker._cash / self.data.Close[-1] // 3
                # sl = self.data.Close[-1] * (1 + self.sl_atr_rate * self.atr[-1])
                sl = self.data.Close[-1] * (1 + self.sl)
                tp1 = self.data.Close[-1] * (1 - self.tp1)
                tp2 = self.data.Close[-1] * (1 - self.tp2)
                tp3 = self.data.Close[-1] * (1 - self.tp3)
                self.sell(size=0.99, sl=sl, tp=tp1, limit=self.data.Close[-1])
