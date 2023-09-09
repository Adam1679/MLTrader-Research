import json
import os
import pickle
import sys
from collections import defaultdict

from vnpy.trader.config_manager import ConfigManager

sys.path.insert(0, "./")
sys.path.insert(0, "./research")
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting._stats import compute_stats
from tqdm.notebook import tqdm
from utility import *

from feature_transformer.signals_vec import *
from research.strategies import SUPER_TREND as STRA_CLS

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
lgb.register_logger(logger)
import warnings

from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, DateRangeSlider, HoverTool, PreText
from bokeh.palettes import Category20_20 as colors
from bokeh.plotting import figure

warnings.filterwarnings("ignore")
import shutil
from pathlib import Path

import yaml

from research.parser import arg_parser

trade_symbols = [
    "ETHUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "SOLUSDT",
    "BTCUSDT",
    "NEARUSDT",
    "AVAXUSDT",
]

DATA_ROOT = "/Users/bytedance/binance_data/csv_data"

P = Path(os.path.join(".vscode", datetime.now().isoformat("#", "seconds")))
DUMP_ROOT = P.absolute()
args = arg_parser.parse_args()
config_manager = ConfigManager(path=f"research/configs", config_name=args.CONFIG_NAME)
stra = config_manager.all_config["TRADER_MODEL"]["BACKTEST_TRADE_STRATEGY"]
BAR_WINDOW = config_manager.bar_window
trade_symbols = config_manager.symbols
train_start_date = datetime.fromisoformat(args.TRAIN_DATE_FROM)
train_end_date = datetime.fromisoformat(args.TRAIN_DATE_TO)
if __name__ == "__main__":
    print("DUMP_ROOT = ", DUMP_ROOT)
    if P.exists():
        shutil.rmtree(DUMP_ROOT)
    P.mkdir(parents=True, exist_ok=True)

    new_dfs = {}
    GLOBAL_DIVIDE_BASE = defaultdict(dict)
    print("load data ...", end="")
    data_dv = {}
    executor = ThreadPoolExecutor()
    for asset_name in trade_symbols:
        data_dv[asset_name] = executor.submit(
            read_symbol,
            asset_name,
            BAR_WINDOW,
            root=DATA_ROOT,
            fake=False,
            start_date=train_start_date,
            end_date=train_end_date,
        )
    executor.shutdown()

    for asset_name in trade_symbols:
        data = data_dv[asset_name].result()
        data = data.dropna()
        if data.size == 0:
            raise ValueError()

        data.index = pd.to_datetime(data.index, unit="s")
        new_dfs[asset_name] = data
    result_df = pd.concat(new_dfs, axis=0, names=["Asset_ID"])

    dv = {}
    for feature in result_df.columns:
        dv[feature] = result_df.pivot(
            index="timestamp", columns="Asset_ID", values=feature
        )
        dv[feature] = dv[feature].fillna(method="ffill")

    executor = ProcessPoolExecutor()
    results_by_symbol = {}
    for symbol in trade_symbols:
        results_by_symbol[symbol] = executor.submit(
            run_model,
            symbol,
            args.plot,
            args.BACKTEST_TRADE_STRATEGY,
            dv,
            DUMP_ROOT,
            _commission=0.0,
        )
    results_by_symbol = {k: v.result() for k, v in results_by_symbol.items()}
    executor.shutdown()

    calc_portfolio_results(
        results_by_symbol, trade_symbols, DUMP_ROOT, args.BACKTEST_TRADE_STRATEGY
    )
