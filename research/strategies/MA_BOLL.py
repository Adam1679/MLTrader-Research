from math import floor

from research.strategies.automl_strategies import *

"""
持仓一天到一周不等一个中等周期趋势跟踪策略，在BTC和ETH上效果还行，4h和1h都可以实践
"""


class MA_BOLL(AutoMLStrategy):
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

    def init(self):
        self.left = int(self.hparams.get("left", 12))
        self.right = int(self.hparams.get("right", 4))
        self.volume_f = float(self.hparams.get("volume_f", 1.9))
        self.volume_ma_window = int(self.hparams.get("volume_ma_window", 52))
        self.adx_window = int(self.hparams.get("adx_window", 11))
        self.adx_th = float(self.hparams.get("adx_th", 10))
        self.trailing_tp = float(self.hparams.get("trailing_tp", 0.02))
        self.long_volume_period = int(self.hparams.get("long_volume_period", 20))
        self.macd_signal_window = int(self.hparams.get("macd_signal_window", 21))
        self.macd_fast = int(self.hparams.get("macd_fast", 9))
        self.macd_slow = int(self.hparams.get("macd_slow", 12))
        self.rsi_window = int(self.hparams.get("rsi_window", 20))

        self.adx = self.I(
            calcADX_adx,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.adx_window,
            name="adx",
            plot=False,
        )

        self.volume_ma_ratio = self.I(
            get_volume_break_ratio,
            self.data.Volume,
            self.volume_ma_window,
            plot=False,
            overlay=False,
            name="volume_break_ratio",
        )

        self.volume_adv = self.I(
            get_volume_advantage,
            self.data.Close,
            self.data.Volume,
            self.long_volume_period,
            plot=False,
            name="volume_advantage",
        )

        self.ma70 = resample_apply("1H", get_ma, self.data.HLC2, 70, name="ma70", plot=True)
        self.ma80 = resample_apply("1H", get_ma, self.data.HLC2, 80, name="ma80", plot=True)
        self.ma90 = resample_apply("1H", get_ma, self.data.HLC2, 90, name="ma90", plot=True)
        
        self.ma70V2 = self.I(get_ma, self.data.HLC2, 70, plot=False)
        self.ma80V2 = self.I(get_ma, self.data.HLC2, 80, plot=False)
        self.ma90V2 = self.I(get_ma, self.data.HLC2, 90, plot=True)
        
        self.bool_multiplier = 2
        self.bool_up = self.I(get_boll_up, self.data.HLC2, 20, self.bool_multiplier, plot=True)
        self.bool_down = self.I(get_boll_down, self.data.HLC2, 20, self.bool_multiplier, plot=True)
        
        self.pivot_high = self.I(
            pivot,
            self.data.HLC2,
            self.left,
            self.right,
            True,
            plot=False,
            overlay=False,
            name="pivot_high",
        )
        self.pivot_low = self.I(
            pivot,
            self.data.HLC2,
            self.left,
            self.right,
            False,
            plot=False,
            overlay=False,
            name="pivot_low",
        )
        self.atr = self.I(get_atr, self.data.High, self.data.Low, self.data.Close, 14, plot=True)
        self.std = self.I(get_std, self.data.Close, 64, plot=True)
        
        self.LARGE_TREND_COND = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="LARGE_TREND_COND"
        )
        
        self.SMALL_TREND_COND = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="SMALL_TREND_COND"
        )
        self.TRIGGER = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="TRIGGER"
        )
        self.BOLL_SUPPORT = self.I(
            lambda: np.zeros_like(self.data.Close), plot=True, name="BOLL_SUPPORT"
        )
        
        self.break_even_point = None
        self.init_cash = self.hparams['args'].cash
        self.risk_level = self.hparams['args'].risk_level
        self.open_price_move = None
        self.moving_tp_keep_ratio = 0.356
        self.moving_tp_keep_enable = False
        self.current_profit = 0

    def next(self):
        if not self.position:
            self.current_profit = 0
            self.moving_tp_keep_enable = False
        if (self.ma70[-1] > self.ma80[-1] > self.ma90[-1]) and (self.ma70V2[-1] > self.ma80V2[-1] > self.ma90V2[-1]):
            self.LARGE_TREND_COND[-1] = 1

        elif (self.ma70[-1] < self.ma80[-1] < self.ma90[-1]) and (self.ma70V2[-1] < self.ma80V2[-1] < self.ma90V2[-1]):
            self.LARGE_TREND_COND[-1] = -1
            
        if (self.ma70V2[-1] > self.ma80V2[-1] > self.ma90V2[-1]):
            self.SMALL_TREND_COND[-1] = 1

        elif (self.ma70V2[-1] < self.ma80V2[-1] < self.ma90V2[-1]):
            self.SMALL_TREND_COND[-1] = -1
        
        up_support = abs(self.data.Close[-1] - self.bool_up[-1]) < 0.8 * self.atr[-1] or abs(self.data.Open[-1] - self.bool_up[-1]) < 0.8 * self.atr[-1]
        down_support = abs(self.data.Close[-1] - self.bool_down[-1]) < 0.8 * self.atr[-1] or abs(self.data.Open[-1] - self.bool_down[-1]) < 0.8 * self.atr[-1]
        if up_support and not down_support:
            self.BOLL_SUPPORT[-1] = -1
        if down_support and not up_support:
            self.BOLL_SUPPORT[-1] = 1
            
        if (self.data.Close[-1] - self.data.Open[-1]) > 1. * self.atr[-1]:
            self.TRIGGER[-1] = 1
            
        if (self.data.Open[-1] - self.data.Close[-1]) > 1. * self.atr[-1]:
            self.TRIGGER[-1] = -1
        
        LONG = self.LARGE_TREND_COND[-1] > 0 and self.TRIGGER[-1] > 0 and self.SMALL_TREND_COND[-1] > 0 and self.BOLL_SUPPORT[-1] > 0
        SHORT = self.LARGE_TREND_COND[-1] < 0 and self.TRIGGER[-1] < 0 and self.LARGE_TREND_COND[-1] > 0 and self.BOLL_SUPPORT[-1] < 0
        R = self.risk_level
        full_pos = 0.99
        if LONG and self.side != "SHORT_ONLY":
            sl = self.data.Open[-1] - 0.1 * self.atr[-1]
            if sl > 0 and not self.position:
                if R > 0:
                    adjust_pos = risk_adjust_position(self.data.Close[-1], sl, self.init_cash, R)
                    if adjust_pos * self.data.Close[-1] > self.init_cash:
                        adjust_pos = 0.99
                else:
                    adjust_pos = 0.99
                
                self.buy(size=adjust_pos, sl=sl, limit=self.data.Close[-1])  # 40%

        elif SHORT and self.side != "LONG_ONLY":
            sl = self.data.Open[-1] + 0.1 * self.atr[-1]
            if sl > 0 and not self.position:
                if R > 0:
                    adjust_pos = risk_adjust_position(self.data.Close[-1], sl, self.init_cash, R)
                    if adjust_pos * self.data.Close[-1] > self.init_cash:
                        adjust_pos = 0.99
                else:
                    adjust_pos = 0.99
                self.sell(size=full_pos, sl=sl, limit=self.data.Close[-1])

        # sl = None
        # for trade in self.trades:
        #     if trade.size > 0:
        #         if LONG:
        #             trade.sl = max(self.pivot_low[-1], trade.sl)
        #         trade.sl = min(self.data.Close[-1] * 0.998, trade.sl)
        #     else:
        #         if SHORT:
        #             trade.sl = min(self.pivot_high[-1], trade.sl)
        #         trade.sl = max(self.data.Close[-1] * 1.002, trade.sl)
        
        sl = None
        # 均线退出
        # for trade in self.trades:
        #     if trade.size > 0:
        #         sl = min(self.ma70[-1], self.ma80[-1], self.ma90[-1])
        #         if self.data.Close[-1] > sl:
        #             trade.sl = sl
        #         else:
        #             trade.close()
                    
        #     else:
        #         sl = max(self.ma70[-1], self.ma80[-1], self.ma90[-1])
        #         if self.data.Close[-1] < sl:
        #             trade.sl = sl
        #         else:
        #             trade.close()
        
        # 布林退出
        # for trade in self.trades:
        #     if trade.size > 0:
        #         if self.data.Close[-1] > self.bool_up[-1]:
        #             trade.close()
                    
        #     else:
        #         if self.data.Close[-1] < self.bool_down[-1]:
        #             trade.close()
        
        # 布林退出 + 移动止盈
        for trade in self.trades:
            if trade.size > 0:
                if self.data.Close[-1] > self.bool_up[-1] and self.moving_tp_keep_enable is False:
                    self.current_profit = abs(self.data.Close[-1] - trade.entry_price)
                    self.moving_tp_keep_enable = True
                if self.moving_tp_keep_enable is True:
                    self.current_profit = max(self.current_profit, abs(self.data.Close[-1] - trade.entry_price))
                    sl = trade.entry_price + self.current_profit * self.moving_tp_keep_ratio
                    if self.data.Close[-1] < sl:
                        trade.close()
                        self.moving_tp_keep_enable = False
            else:
                if self.data.Close[-1] < self.bool_down[-1] and self.moving_tp_keep_enable is False:
                    self.current_profit = abs(self.data.Close[-1] - trade.entry_price)
                    self.moving_tp_keep_enable = True
                if self.moving_tp_keep_enable is True:
                    self.current_profit = max(self.current_profit, abs(self.data.Close[-1] - trade.entry_price))
                    sl = trade.entry_price - self.current_profit * self.moving_tp_keep_ratio
                    if self.data.Close[-1] > sl:
                        trade.close()
                        self.moving_tp_keep_enable = False
                    