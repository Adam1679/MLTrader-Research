from math import floor

from research.strategies.automl_strategies import *

"""
持仓一天到一周不等一个中等周期趋势跟踪策略，在BTC和ETH上效果还行，4h
"""


class Maximalist_4h(AutoMLStrategy):
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
    fixed_parameters = ["side", "max_sl", "bankroll"]

    def init(self):
        self.left = int(self.hparams.get("left", 6))
        self.right = int(self.hparams.get("right", 3))
        self.volume_f = float(self.hparams.get("volume_f", 1.9))
        self.volume_ma_window = int(self.hparams.get("volume_ma_window", 40))
        self.adx_window = int(self.hparams.get("adx_window", 11))
        self.adx_th = float(self.hparams.get("adx_th", 12))
        # self.Sacc = float(self.hparams.get("Sacc", 0.1))
        # self.Smax = float(self.hparams.get("Smax", 0.9))
        # self.vwap_rsi_window = int(self.hparams.get("vwap_rsi_window", 48))
        # self.vwap_rsi_oversold = float(self.hparams.get("vwap_rsi_oversold", 29))
        # self.vwap_rsi_overbought = float(self.hparams.get("vwap_rsi_overbought", 75))
        self.sl = float(self.hparams.get("sl", 0.072))
        self.tp1 = float(self.hparams.get("tp1", 0.04))
        self.tp2 = float(self.hparams.get("tp2", 0.045))
        self.tp3 = float(self.hparams.get("tp3", 0.014))
        self.long_volume_period = int(self.hparams.get("long_volume_period", 20))

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

        self.RS_BREAK_COND = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="RS_BREAK_COND"
        )
        self.VOLUME_BREAK_COND = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VOLUME_BREAK_COND"
        )
        self.ADX_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ADX_SIGNAL"
        )
        self.risk = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="POSITION RISK"
        )
        self.init_cash = 10_000_000

    def next(self):
        if self.data.Close[-1] > self.pivot_high[-1]:
            self.RS_BREAK_COND[-1] = 1

        elif self.data.Close[-1] < self.pivot_low[-1]:
            self.RS_BREAK_COND[-1] = -1

        if self.data.Volume[-1] > self.volume_ma[-1] * self.volume_f:
            self.VOLUME_BREAK_COND[-1] = 1

        if self.plus[-1] > self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = 1

        elif self.plus[-1] < self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = -1

        if (
            self.RS_BREAK_COND[-1] > 0
            and self.VOLUME_BREAK_COND[-1] > 0  # 有用，filter了很多差的交易
            and self.ADX_SIGNAL[-1] > 0  # 有用，filter了很多差的交易，但是收益也降低，并且交易次数显著降低一半
            and self.volume_adv[-1] >= 0.51  # 特别有用
        ):
            self.RS_BREAK_ENTRY[-1] = 1

        elif (
            self.RS_BREAK_COND[-1] < 0
            and self.VOLUME_BREAK_COND[-1] > 0  # 有用，filter了很多差的交易
            and self.ADX_SIGNAL[-1] < 0  # 有用，filter了很多差的交易，但是收益也降低，并且交易次数显著降低一半
            and self.volume_adv[-1] <= 0.49
        ):
            self.RS_BREAK_ENTRY[-1] = -1

        LONG = self.RS_BREAK_ENTRY[-1] > 0

        SHORT = self.RS_BREAK_ENTRY[-1] < 0
        full_pos = floor(self.init_cash / self.data.Close[-1])
        if LONG and self.side != "SHORT_ONLY":
            sl = self.pivot_low[-1]
            # cash = self._broker._bankroll
            # max_loss_size = int(self.R * cash) // abs(self.data.Close[-1] - sl)
            tp1 = self.data.Close[-1] * (1 + self.tp1)
            if sl > 0 and not self.position:
                self.buy(size=full_pos, sl=sl, limit=self.data.Close[-1])

            # elif sl > 0 and len(self.trades) == 1 and self.position.is_long():
            #     entry_price = self.trades[0].entry_price
            #     mix_entry_price = (entry_price + )

        elif SHORT and self.side != "LONG_ONLY":
            # sl = self.data.Close[-1] * (1 + self.sl_atr_rate * self.atr[-1])
            # cash = self._broker._cash
            sl = self.pivot_high[-1]
            # max_loss_size = (self.R * cash) // abs(self.data.Close[-1] - sl)
            tp1 = self.data.Close[-1] * (1 - self.tp1)

            if sl > 0 and not self.position:
                self.sell(size=full_pos, sl=sl, limit=self.data.Close[-1])

        sl = None
        for trade in self.trades:
            if trade.size > 0:
                if sl is None:
                    sl = max(self.pivot_low[-1], trade.sl)
                else:
                    sl = max(self.pivot_low[-1], trade.sl, sl)
            else:
                if sl is None:
                    sl = min(self.pivot_high[-1], trade.sl)
                else:
                    sl = min(self.pivot_low[-1], trade.sl, sl)

        for trade in self.trades:
            trade.sl = sl

        # if len(self.trades) > 0:
        #     price = 0
        #     size = 0
        #     for trade in self.trades:
        #         price += trade.entry_price
        #         size += trade.size
        #     avg_price = price / len(self.trades)
        #     potential_loss = size * (self.data.Close[-1] - avg_price)
        #     risk = potential_loss / self._broker._bankroll
        #     self.risk[-1] = risk
