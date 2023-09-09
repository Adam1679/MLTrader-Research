from math import floor

from research.strategies.automl_strategies import *

"""
持仓一天到一周不等一个中等周期趋势跟踪策略，在BTC和ETH上效果还行，4h和1h都可以实践
"""


class Maximalist_5m(AutoMLStrategy):
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
    fixed_parameters = ["side", "max_sl", "bankroll"]

    def init(self):
        self.left = int(self.hparams.get("left", 10))
        self.right = int(self.hparams.get("right", 5))
        self.volume_f = float(self.hparams.get("volume_f", 1.9))
        self.volume_ma_window = int(self.hparams.get("volume_ma_window", 52))
        self.adx_window = int(self.hparams.get("adx_window", 11))
        self.adx_th = float(self.hparams.get("adx_th", 10))
        self.long_volume_period = int(self.hparams.get("long_volume_period", 20))
        self.macd_signal_window = int(self.hparams.get("macd_signal_window", 21))
        self.macd_fast = int(self.hparams.get("macd_fast", 9))
        self.macd_slow = int(self.hparams.get("macd_slow", 12))
        self.rsi_window = int(self.hparams.get("rsi_window", 20))
        self.bar_stats_tb = int(self.hparams.get("bar_stats_tb", 5))

        self.cooldown_signal_threshold = int(
            self.hparams.get("cooldown_signal_threshold", 1)
        )
        self.cooldown_trigger_threshold = int(
            self.hparams.get("cooldown_trigger_threshold", 0.05)
        )

        self.pivot_high = resample_apply(
            "30min",
            pivot,
            self.data.HLC2,
            self.left,
            self.right,
            True,
            plot=True,
            overlay=True,
        )

        self.pivot_low = resample_apply(
            "30min",
            pivot,
            self.data.HLC2,
            self.left,
            self.right,
            False,
            plot=True,
            overlay=True,
        )

        self.plus = resample_apply(
            "30min",
            calcADX_plus,
            [self.data.High, self.data.Low, self.data.Close],
            self.adx_window,
            name="adx_plus",
            plot=True,
        )
        self.minus = resample_apply(
            "30min",
            calcADX_minus,
            [self.data.High, self.data.Low, self.data.Close],
            self.adx_window,
            name="adx_minus",
            plot=True,
        )
        self.adx = resample_apply(
            "30min",
            calcADX_adx,
            [self.data.High, self.data.Low, self.data.Close],
            self.adx_window,
            name="adx",
            plot=True,
        )

        self.R = 0.01
        self.volume_ma_ratio = resample_apply(
            "30min",
            get_volume_break_ratio,
            self.data.Volume,
            self.volume_ma_window,
            plot=True,
            overlay=False,
            name="volume_ma_ratio",
        )

        self.macd = resample_apply(
            "30min",
            get_macd,
            self.data.OHLC,
            self.macd_fast,
            self.macd_slow,
            name="macd",
            plot=True,
            agg_dict={"ohlc": "last"},
        )
        self.macd_signal = resample_apply(
            "30min",
            get_macd_signal,
            self.data.OHLC,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal_window,
            name="macd_signal",
            plot=True,
            agg_dict={"ohlc": "last"},
        )
        self.macd_signal = resample_apply(
            "30min",
            get_macd_signal,
            self.data.OHLC,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal_window,
            name="macd_signal",
            plot=True,
            agg_dict={"ohlc": "last"},
        )
        self.bar_stats = self.I(
            get_bar_stats,
            self.data.Open,
            self.data.High,
            self.data.Low,
            self.data.Close,
            name="bar_stats",
            plot=True,
        )
        self.volume_adv = resample_apply(
            "30min",
            get_volume_advantage,
            [self.data.Close, self.data.Volume],
            self.long_volume_period,
            plot=True,
        )

        self.RS_BREAK_ENTRY = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="RS_BREAK_ENTRY"
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
        self.MACD_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="MACD_SIGNAL"
        )
        self.VOLUME_ADV_SIGNAL = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="VOLUME_ADV_SIGNAL"
        )
        self.break_even_point = None

        self.init_cash = 50_000_000
        self.open_price_move = None
        self.traded_hour = set()

    def next(self):

        if self.data.Close[-1] > self.pivot_high[-1]:
            self.RS_BREAK_COND[-1] = 1

        elif self.data.Close[-1] < self.pivot_low[-1]:
            self.RS_BREAK_COND[-1] = -1

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

        # full_pos = floor(self.init_cash / self.data.Close[-1])
        full_pos = 0.99
        if (
            self.RS_BREAK_COND[-1] > 0
            and self.VOLUME_BREAK_COND[-1] > 0  # 有用，filter了很多差的交易
            # and self.ADX_SIGNAL[-1] > 0  # 有用，filter了很多差的交易，但是收益也降低，并且交易次数显著降低一半
            # and self.VOLUME_ADV_SIGNAL[-1] > 0 # 特别有用
            and self.MACD_SIGNAL[-1] > 0
            and self.bar_stats[-1] < self.bar_stats_tb
            # and self.rsi[-1] < 70
            # and self.market_efficiency > 0.6
            # and self.pivot_diff[-1] / self.pivot_diff_ma[-1] > 1.2
            # and diff_ma_t
        ):
            self.RS_BREAK_ENTRY[-1] = 1

        elif (
            self.RS_BREAK_COND[-1] < 0
            and self.VOLUME_BREAK_COND[-1] > 0  # 有用，filter了很多差的交易
            # and self.ADX_SIGNAL[-1] < 0  # 有用，filter了很多差的交易，但是收益也降低，并且交易次数显著降低一半
            # and self.VOLUME_ADV_SIGNAL[-1] < 0
            and self.MACD_SIGNAL[-1] < 0
            and self.bar_stats[-1] < self.bar_stats_tb
            # and self.rsi[-1] > 30
            # and self.market_efficiency > 0.6 # 增加了一点点胜率，提高了盈亏比。减少了一些交易，提高夏普
            # and self.pivot_diff[-1] / self.pivot_diff_ma[-1] > 1.2 # 不是能很好的define trend
            # and self.data.Close[-1] < self.ma_90[-1]
            # and diff_ma_t
        ):
            self.RS_BREAK_ENTRY[-1] = -1
        round_index = self.data.index[-1].minute % 30 == 0
        cur_hour = int(self.data.index[-1].timestamp()) // 60 // 60

        LONG = self.RS_BREAK_ENTRY[-1] > 0 and cur_hour not in self.traded_hour

        SHORT = self.RS_BREAK_ENTRY[-1] < 0 and cur_hour not in self.traded_hour
        if LONG and self.side != "SHORT_ONLY":
            sl = self.pivot_high[-1]
            if sl > 0 and not self.position:
                self.break_even_point = 2 * self.data.Close[-1] - sl
                # 在过去的高点卖出结论不一定是最好的

                self.buy(size=full_pos, sl=sl, limit=self.data.Close[-1])  # 40%
                # self.sell(size=full_pos // 2, stop=True, limit=self.data.Close[-1])  # 40%
                self.open_price_move = abs(self.data.Close[-1] - self.data.Open[-6])
                self.traded_hour.add(cur_hour)

        elif SHORT and self.side != "LONG_ONLY":
            sl = self.pivot_low[-1]
            if sl > 0 and not self.position:
                self.break_even_point = 2 * self.data.Close[-1] - sl
                self.sell(size=full_pos, sl=sl, limit=self.data.Close[-1])
                self.open_price_move = abs(self.data.Close[-1] - self.data.Open[-6])
                self.traded_hour.add(cur_hour)

        if round_index:
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
                        sl = min(self.pivot_high[-1], trade.sl, sl)

            # for trade in self.trades:
            #     if trade.size > 0 and self.data.Close[-1] >= sl:
            #         trade.sl = sl
            #     if trade.size < 0 and self.data.Close[-1] <= sl:
            #         trade.sl = sl

            for trade in self.trades:
                hold_minutes = (
                    self.data.index[-1] - trade.entry_time
                ).total_seconds() // 60
                move = self.data.Close[-1] - trade.entry_price
                if hold_minutes <= 30 * 5:
                    # 如果下一个bar下跌一半，则直接止损。有用。
                    if trade.is_long:
                        if move < 0 and abs(move) >= self.open_price_move * 0.5:
                            trade.close()
                    elif trade.is_short:
                        if move > 0 and abs(move) >= self.open_price_move * 0.5:
                            trade.close()
                if LONG and trade.is_short:
                    trade.close()
                if SHORT and trade.is_long:
                    trade.close()

            # 在break-even止盈的操作带来了收益
            for trade in self.trades:
                if self.break_even_point is None:
                    continue
                if trade.size > 0 and self.data.Close[-1] >= self.break_even_point:
                    self.break_even_point = None
                    trade.close(0.5)
                if trade.size < 0 and self.data.Close[-1] <= self.break_even_point:
                    self.break_even_point = None
                    trade.close(0.5)
