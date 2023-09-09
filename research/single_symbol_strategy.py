import json
import os
import random
import sys
import traceback
from math import isinf, isnan

from research.strategies import *
from research.strategies.automl_strategies import AutoMLStrategy

sys.path.insert(0, "./")
sys.path.insert(0, "./research")
import logging
from datetime import datetime

import pandas as pd
from utility import read_symbol, run_model

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import warnings

warnings.filterwarnings("ignore")
import shutil
from pathlib import Path

from research.htuner_utility import *
from research.parser import arg_parser

DATA_ROOT = "/Users/bytedance/binance_data/csv_data"
BAR_WINDOW = None
SYMBOL = ""

if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--BACKTEST_TRADE_STRATEGY", default="")
    arg_parser.add_argument("--TRAIN_DATE_FROM", default="", type=str)
    arg_parser.add_argument("--TRAIN_DATE_TO", default="", type=str)
    arg_parser.add_argument("--plot", action="store_true")
    arg_parser.add_argument("--study_name", default="")
    arg_parser.add_argument("--bar-window", default=15, type=int)
    arg_parser.add_argument("--risk-level", default=-1, type=float)
    arg_parser.add_argument("--cash", default=1_000_000_000, type=float)
    arg_parser.add_argument(
        "--side", default="BOTH", type=str, choices=["BOTH", "LONG_ONLY", "SHORT_ONLY"]
    )
    arg_parser.add_argument("--optimize_symbol", default="")
    arg_parser.add_argument("--fee", default=1e-3, type=float)
    
    args = arg_parser.parse_args()

    DUMP_ROOT = Path(".vscode")

    print("load data ...", end="")
    new_dfs = {}
    BAR_WINDOW = args.bar_window
    train_start_date = datetime.fromisoformat(args.TRAIN_DATE_FROM)
    train_end_date = datetime.fromisoformat(args.TRAIN_DATE_TO)
    asset_name = args.optimize_symbol
    SYMBOL = args.optimize_symbol
    sub_df = read_symbol(
        asset_name,
        args.bar_window,
        root=DATA_ROOT,
        fake=False,
        start_date=train_start_date,
        end_date=train_end_date,
        future=True,
    )
    sub_df.index = sub_df.index.tz_localize("UTC")
    if sub_df.size == 0:
        raise ValueError()

    new_dfs[asset_name] = sub_df

    result_df = pd.concat(new_dfs, axis=0, names=["Asset_ID"])
    global_dv = {}
    for feature in result_df.columns:
        global_dv[feature] = result_df.pivot(
            index="timestamp", columns="Asset_ID", values=feature
        )
        global_dv[feature] = global_dv[feature].fillna(method="ffill")

    strategy_name = args.BACKTEST_TRADE_STRATEGY
    stra_cls: AutoMLStrategy = eval(strategy_name)
    if args.study_name == "":
        args.study_name = (
            asset_name
            + "_"
            + strategy_name
            + "_"
            + datetime.now().strftime("%Y%m%d%H%M%S")
        )

    dump_path = (
        DUMP_ROOT.joinpath(str(BAR_WINDOW))
        .joinpath(SYMBOL)
        .joinpath(args.study_name)
        .joinpath("base")
    )
    if not dump_path.exists():
        dump_path.mkdir(parents=True)
    run_model(
        args,
        SYMBOL,
        args.plot,
        args.BACKTEST_TRADE_STRATEGY,
        global_dv,
        dump_path,
    )
