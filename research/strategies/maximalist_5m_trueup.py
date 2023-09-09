from math import floor

from research.strategies.automl_strategies import *

"""
持仓一天到一周不等一个中等周期趋势跟踪策略，在BTC和ETH上效果还行，4h和1h都可以实践
"""


class Maximalist_5m_trueup(AutoMLStrategy):
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
        self.volume_f = float(self.hparams.get("volume_f", 1.9))
        self.volume_ma_window = int(self.hparams.get("volume_ma_window", 33))
        self.adx_window = int(self.hparams.get("adx_window", 13))
        self.adx_th = float(self.hparams.get("adx_th", 13.5))
        self.long_volume_period = int(self.hparams.get("long_volume_period", 20))
        self.macd_signal_window = int(self.hparams.get("macd_signal_window", 21))
        self.macd_fast = int(self.hparams.get("macd_fast", 9))
        self.macd_slow = int(self.hparams.get("macd_slow", 12))
        self.rsi_window = int(self.hparams.get("rsi_window", 90))
        self.tp = float(self.hparams.get("tp", 4.5 / 100))
        self.sl = float(self.hparams.get("sl", 7 / 100))
        self.tp_atr = float(self.hparams.get("tp_atr", 4))
        self.sl_atr = float(self.hparams.get("sl_atr", 2.5))
        self.atr_window = float(self.hparams.get("atr_window", 52))
        self.training_keep_pnl_rate = float(
            self.hparams.get("training_keep_pnl_rate", 0.6)
        )
        self.atr = self.I(
            get_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_window,
            plot=True,
        )
        self.cooldown_signal_threshold = int(
            self.hparams.get("cooldown_signal_threshold", 1)
        )
        self.cooldown_signal_threshold_in_bars = int(
            self.hparams.get("cooldown_signal_threshold_in_bars", 48)
        )
        self.cooldown_trigger_threshold = float(
            self.hparams.get("cooldown_trigger_threshold", 0.05)
        )

        self.plus = self.I(
            calcADX_plus,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.adx_window,
            name="adx_plus",
            plot=True,
        )
        self.minus = self.I(
            calcADX_minus,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.adx_window,
            name="adx_minus",
            plot=True,
        )
        self.rsi = self.I(
            get_rsi,
            self.data.Open,
            self.rsi_window,
            name="adx_plus",
            plot=True,
        )
        self.adx = self.I(
            calcADX_adx,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.adx_window,
            name="adx",
            plot=True,
        )

        self.volume_ma_ratio = self.I(
            get_volume_break_ratio,
            self.data.Volume,
            self.volume_ma_window,
            plot=True,
            overlay=False,
        )

        self.macd = self.I(
            get_macd,
            self.data.OHLC,
            self.macd_fast,
            self.macd_slow,
            name="macd",
            plot=True,
        )
        self.macd_signal = self.I(
            get_macd_signal,
            self.data.OHLC,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal_window,
            name="macd_signal",
            plot=True,
        )
        self.volume_adv = self.I(
            get_volume_advantage,
            self.data.Close,
            self.data.Volume,
            self.long_volume_period,
            plot=True,
        )

        self.volume_adv = self.I(
            get_volume_advantage,
            self.data.Close,
            self.data.Volume,
            self.long_volume_period,
            plot=True,
        )
        self.RS_BREAK_ENTRY = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="RS_BREAK_ENTRY"
        )
        self.VOLUME_BREAK_COND = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VOLUME_BREAK_COND"
        )
        self.ADX_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="ADX_SIGNAL"
        )
        self.MACD_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MACD_SIGNAL"
        )
        self.VOLUME_ADV_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VOLUME_ADV_SIGNAL"
        )
        self.COOL_DOWN = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="COOL_DOWN"
        )
        self.break_even_point = None
        self.risk_level = (
            self.hparams["risk_level"] if self.hparams["risk_level"] > 0 else 1.0
        )
        self.open_price_move = None
        self.last_check_time = 0

        self.tp_price = None
        self.sl_price = None

    def next(self):

        if self.volume_ma_ratio[-1] > self.volume_f:
            self.VOLUME_BREAK_COND[-1] = 1

        if self.plus[-1] > self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = 1

        elif self.plus[-1] < self.minus[-1] and self.adx[-1] > self.adx_th:
            self.ADX_SIGNAL[-1] = -1

        if self.macd[-1] > self.macd_signal[-1]:
            self.MACD_SIGNAL[-1] = 1
        elif self.macd[-1] < self.macd_signal[-1]:
            self.MACD_SIGNAL[-1] = -1

        if self.volume_adv[-1] >= 0.51:
            self.VOLUME_ADV_SIGNAL[-1] = 1
        elif self.volume_adv[-1] <= 0.49:
            self.VOLUME_ADV_SIGNAL[-1] = -1
        full_pos = 0.99
        if (
            self.VOLUME_BREAK_COND[-1] > 0
            and self.ADX_SIGNAL[-1] > 0
            and self.VOLUME_ADV_SIGNAL[-1] > 0
            and self.MACD_SIGNAL[-1] > 0
            and self.rsi[-1] < 70
        ):
            self.RS_BREAK_ENTRY[-1] = 1

        elif (
            self.VOLUME_BREAK_COND[-1] > 0  # 有用，filter了很多差的交易
            and self.ADX_SIGNAL[-1] < 0  # 有用，filter了很多差的交易，但是收益也降低，并且交易次数显著降低一半
            and self.VOLUME_ADV_SIGNAL[-1] < 0  # 特别有用
            and self.MACD_SIGNAL[-1] < 0
            and self.rsi[-1] > 30
        ):
            self.RS_BREAK_ENTRY[-1] = -1

        LONG = self.RS_BREAK_ENTRY[-1] > 0
        SHORT = self.RS_BREAK_ENTRY[-1] < 0

        if LONG and self.side != "SHORT_ONLY" and not self.position:
            # sl = self.data.Close[-1] - self.sl_atr * self.atr[-1]
            # self.tp_price = self.data.Close[-1] + self.tp_atr * self.atr[-1]

            sl = self.data.Close[-1] - self.sl * self.data.Close[-1]
            self.tp_price = self.data.Close[-1] + self.tp * self.data.Close[-1]
            sp_rate = abs(sl - self.data.Close[-1]) / self.data.Close[-1]
            risk_adjusted_pos = min(self.risk_level / sp_rate, 0.99)
            if sl > 0:
                self.break_even_point = 2 * self.data.Close[-1] - sl
                self.buy(
                    size=risk_adjusted_pos,
                    sl=sl,
                    limit=self.data.Close[-1],
                    tp=self.tp_price,
                )  # 40%
                self.open_price_move = abs(self.data.Close[-1] - self.data.Open[-1])
            elif self.position.is_short:
                self.position.close()

        elif SHORT and self.side != "LONG_ONLY" and not self.position:
            # sl = self.data.Close[-1] + self.sl_atr * self.atr[-1]
            # self.tp_price = self.data.Close[-1] - self.tp_atr * self.atr[-1]

            sl = self.data.Close[-1] + self.sl * self.data.Close[-1]
            self.tp_price = self.data.Close[-1] - self.tp * self.data.Close[-1]
            sp_rate = abs(sl - self.data.Close[-1]) / self.data.Close[-1]
            risk_adjusted_pos = min(self.risk_level / sp_rate, 0.99)
            if sl > 0:
                self.break_even_point = 2 * self.data.Close[-1] - sl
                self.sell(
                    size=risk_adjusted_pos,
                    sl=sl,
                    limit=self.data.Close[-1],
                    tp=self.tp_price,
                )
                self.open_price_move = abs(self.data.Close[-1] - self.data.Open[-1])

            elif self.position.is_long:
                self.position.close()

        # if len(self.trades):
        #     for trade in self.trades:
        #         if trade.size > 0:
        #             pnl = self.data.Close[-1] - trade.entry_price
        #             self.keep_pnl_price = trade.entry_price + self.training_keep_pnl_rate * pnl
        #             if self.tp_price and self.data.Close[-1] > self.tp_price:
        #                 self.tp_price = None
        #                 trade.close(0.5)
        #             trade.sl = max(trade.sl, self.keep_pnl_price)

        #         elif trade.size < 0:
        #             pnl = trade.entry_price - self.data.Close[-1]
        #             self.keep_pnl_price = trade.entry_price - self.training_keep_pnl_rate * pnl
        #             if self.tp_price and self.data.Close[-1] < self.tp_price:
        #                 self.tp_price = None
        #                 trade.close(0.5)
        #             trade.sl = min(trade.sl, self.keep_pnl_price)

        # sl = None
        # for trade in self.trades:
        #     if trade.size > 0:
        #         if LONG:
        #             sl = self.pivot_high[-1]
        #         if sl is None:
        #             sl = max(self.pivot_low[-1], trade.sl)
        #         else:
        #             sl = max(self.pivot_low[-1], trade.sl, sl)
        #     else:
        #         if SHORT:
        #             sl = self.pivot_low[-1]
        #         if sl is None:
        #             sl = min(self.pivot_high[-1], trade.sl)
        #         else:
        #             sl = min(self.pivot_high[-1], trade.sl, sl)

        # for trade in self.trades:
        #     trade.sl = sl
        # for trade in self.trades:
        #     hold_bar = len(self.data) - trade.entry_bar
        #     move_since_open = self.data.Close[-1] - trade.entry_price
        #     move = self.data.Close[-1] - trade.entry_price
        #     if hold_bar <= 5:
        #         # 如果下一个bar下跌一半，则直接止损。有用。
        #         if trade.is_long:
        #             if move < 0 and abs(move) >= self.open_price_move * 0.5:
        #                 trade.close()
        #         elif trade.is_short:
        #             if move > 0 and abs(move) >= self.open_price_move * 0.5:
        #                 trade.close()
        #     if LONG and trade.is_short:
        #         trade.close()
        #     if SHORT and trade.is_long:
        #         trade.close()

        # 这样不行
        # if trade.is_long and move_since_open > 0:
        #     trade.sl = max(trade.sl, trade.entry_price)
        # if trade.is_short and move_since_open < 0:
        #     trade.sl = min(trade.sl, trade.entry_price)

        # 这样不行, 容易止盈，少了很多利润
        # if trade.is_long:
        #     if self.data.Close[-1] > self.pivot_high[-1]:
        #         trade.sl = max(trade.sl, self.pivot_high[-1])
        # if trade.is_short:
        #     if self.data.Close[-1] < self.pivot_high[-1]:
        #         trade.sl = min(trade.sl, self.pivot_low[-1])

        # 这样不行
        # for trade in self.trades:
        #     hold_bar = len(self.data) - trade.entry_bar
        #     if hold_bar == 1:
        #         a = abs(self.data.Close[-1] - self.data.Open[-1])
        #         b = abs(self.data.High[-1] - self.data.Low[-1])
        #         diff = a / (b+1e-10)
        #         move = self.data.Close[-1] - self.data.Open[-1]
        #         if trade.is_long:
        #             if move < 0 or diff < 0.01:
        #                 trade.close()
        #             if diff < 0.05: # 扎针
        #                 trade.close
        #         elif trade.is_short:
        #             if move > 0 or diff < 0.01:
        #                 trade.close()
        # 在break-even止盈的操作带来了收益
        # for trade in self.trades:
        #     if self.break_even_point is None:
        #         continue
        #     if trade.size > 0 and self.data.Close[-1] >= self.break_even_point:
        #         self.break_even_point = None
        #         trade.close(0.5)
        #     if trade.size < 0 and self.data.Close[-1] <= self.break_even_point:
        #         self.break_even_point = None
        #         trade.close(0.5)
