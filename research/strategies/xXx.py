from backtesting.lib import crossover

from research.strategies.automl_strategies import *

""" 目前不太work， 尝试过5min，15min波段"""


class xXx(AutoMLStrategy):
    parameter_specs = []
    parameters = []
    constraints = []
    fixed_parameters = ["side", "max_sl"]

    def init(self):
        self.left = int(self.hparams.get("left", 4))
        self.right = int(self.hparams.get("right", 5))
        self.adx_window = int(self.hparams.get("adx_window", 18))
        self.adx_th = float(self.hparams.get("adx_th", 14))
        self.sma_window_1 = int(self.hparams.get("sma_window_1", 400))
        self.mad_window = int(self.hparams.get("mad_window", 15))
        self.mad_gamma = int(self.hparams.get("mad_gamma", 3))
        self.sl = float(self.hparams.get("sl", 0.04))
        self.tp1 = float(self.hparams.get("tp1", 0.007))
        self.tp2 = float(self.hparams.get("tp2", 0.045))
        self.tp3 = float(self.hparams.get("tp3", 0.014))
        self.long_volume_period = int(self.hparams.get("long_volume_period", 40))

        self.pivot_high = self.I(
            pivot, self.data.High, self.left, self.right, True, plot=True, overlay=True
        )
        self.pivot_low = self.I(
            pivot, self.data.Low, self.left, self.right, False, plot=True, overlay=True
        )

        plus, minus, adx = calcADX(
            self.data.High, self.data.Low, self.data.Close, self.adx_window
        )
        self.plus = self.I(equal_map, plus, name="adx_plus", plot=True)
        self.minus = self.I(equal_map, minus, name="adx_minus", plot=True)
        self.adx = self.I(equal_map, adx, name="adx", plot=True)

        self.ma_1 = self.I(
            get_ma, self.data.Close, self.sma_window_1, plot=True, overlay=True
        )

        self.mad = self.I(
            get_autonomous_recursive_ma,
            self.data.Close,
            self.mad_window,
            self.mad_gamma,
            zero_lag=True,
            plot=True,
            overlay=True,
        )
        self.RS_BREAK_ENTRY = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="RS_BREAK_ENTRY"
        )
        self.RS_BREAK_COND = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="RS_BREAK_COND"
        )
        self.ADX_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ADX_SIGNAL"
        )
        self.MA_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MA_SIGNAL"
        )
        self.MAD_FLAT = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MAD_FLAT"
        )
        self.volume_adv = self.I(
            get_volume_advantage,
            self.data.Close,
            self.data.Volume,
            self.long_volume_period,
            plot=True,
        )

    def next(self):
        if self.ma_1[-1] < self.data.Close[-1]:
            self.MA_SIGNAL[-1] = 1

        elif self.ma_1[-1] > self.data.Close[-1]:
            self.MA_SIGNAL[-1] = -1

        if (
            self.data.Close[-1] > self.pivot_high[-1]
            and self.data.Close[-2] < self.pivot_high[-2]
        ):
            self.RS_BREAK_COND[-1] = 1

        elif (
            self.data.Close[-1] < self.pivot_low[-1]
            and self.data.Close[-2] > self.pivot_low[-2]
        ):
            self.RS_BREAK_COND[-1] = -1

        if self.plus[-1] > self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = 1

        elif self.plus[-1] < self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = -1

        if abs(self.mad[-1] / self.mad[-2] - 1) < 0.001:
            self.MAD_FLAT[-1] = 1

        if (
            self.RS_BREAK_COND[-1] > 0
            and self.ADX_SIGNAL[-1] > 0
            and self.MA_SIGNAL[-1] > 0
            and self.MAD_FLAT[-1] == 0
            and self.volume_adv[-1] > 0.51
        ):
            self.RS_BREAK_ENTRY[-1] = 1

        elif (
            self.RS_BREAK_COND[-1] < 0
            and self.ADX_SIGNAL[-1] < 0
            and self.MA_SIGNAL[-1] < 0
            and self.MAD_FLAT[-1] == 0
            and self.volume_adv[-1] > 0.51
        ):
            self.RS_BREAK_ENTRY[-1] = -1

        LONG = self.RS_BREAK_ENTRY[-1] > 0

        SHORT = self.RS_BREAK_ENTRY[-1] < 0

        if LONG and self.side != "SHORT_ONLY":
            sl = self.data.Close[-1] * (1 - self.sl)
            # sl = self.data.Close[-1] * (1 - self.sl_atr_rate * self.atr[-1])
            tp1 = self.data.Close[-1] * (1 + self.tp1)
            tp2 = self.data.Close[-1] * (1 + self.tp2)
            tp3 = self.data.Close[-1] * (1 + self.tp3)
            max_amount = (self._broker._bankroll / self.data.Close[-1]) // 10
            need_money = max_amount * self.data.Close[-1]
            if max_amount > 0 and self._broker._cash > need_money:
                self.buy(size=0.99, tp=tp1, sl=sl, limit=self.data.Close[-1])

        elif SHORT and self.side != "LONG_ONLY":
            # sl = self.data.Close[-1] * (1 + self.sl_atr_rate * self.atr[-1])
            sl = self.data.Close[-1] * (1 + self.sl)
            tp1 = self.data.Close[-1] * (1 - self.tp1)
            tp2 = self.data.Close[-1] * (1 - self.tp2)
            tp3 = self.data.Close[-1] * (1 - self.tp3)
            max_amount = (self._broker._bankroll / self.data.Close[-1]) // 10
            need_money = max_amount * self.data.Close[-1]
            if max_amount > 0 and self._broker._cash > need_money:
                self.sell(size=max_amount, tp=tp1, sl=sl, limit=self.data.Close[-1])
