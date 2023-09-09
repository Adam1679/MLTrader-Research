from math import floor

from backtesting.lib import cross, crossover

from research.strategies.automl_strategies import *

"""
持仓一天到一周不等一个中等周期趋势跟踪策略，在BTC和ETH上效果还行，4h和1h都可以实践
"""


class Boo_5m(AutoMLStrategy):
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
        self.ma_window = int(self.hparams.get("ma_window", 5))
        self.mid_window = int(self.hparams.get("mid_window", 26))
        self.rsi_window = int(self.hparams.get("rsi_window", 14))
        self.sl = float(self.hparams.get("sl", 5e-3))
        self.tp = float(self.hparams.get("tp", 5e-3))

        self.mid = self.I(
            get_ma, self.data.Close, self.mid_window, plot=True, overlay=True
        )
        self.std = self.I(
            get_std, self.data.Close, self.mid_window, plot=False, overlay=False
        )
        self.up = self.I(lambda: self.mid + 2 * self.std, plot=True, overlay=True)
        self.down = self.I(lambda: self.mid - 2 * self.std, plot=True, overlay=True)
        self.rsi = self.I(get_rsi, self.data.Close, self.rsi_window, plot=True)

        self.ma = self.I(
            get_ma, self.data.Close, self.ma_window, plot=True, overlay=True
        )
        self.ma90 = self.I(get_ma, self.data.Close, 99, plot=True, overlay=True)
        self.ma25 = self.I(get_ma, self.data.Close, 25, plot=True, overlay=True)

        self.break_even_point = None

        self.init_cash = 50_000_000
        self.open_price_move = None
        self.traded_hour = set()

    def next(self):
        LONG = False
        SHORT = False
        if (
            crossover(self.data.Close, self.down)
            and crossover(self.rsi, 40)
            and self.ma25[-1] > self.ma90[-1]
        ):
            LONG = True
        if (
            crossover(self.up, self.data.Close)
            and crossover(60, self.rsi)
            and self.ma25[-1] < self.ma90[-1]
        ):
            SHORT = True

        full_pos = 0.99
        if LONG and self.side != "SHORT_ONLY":
            sl = (1 - self.sl) * self.data.Close[-1]
            tp = (1 + self.tp) * self.data.Close[-1]
            if sl > 0 and not self.position:
                self.buy(size=full_pos, sl=sl, limit=self.data.Close[-1], tp=tp)  # 40%

        elif SHORT and self.side != "LONG_ONLY":
            sl = (1 + self.sl) * self.data.Close[-1]
            tp = (1 - self.tp) * self.data.Close[-1]
            if sl > 0 and not self.position:
                self.sell(size=full_pos, sl=sl, tp=tp, limit=self.data.Close[-1])
