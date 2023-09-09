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


def evaluate_strategy(
    args,
    study_id,
    trial_id,
    global_dv,
    root: Path,
    hparams,
    optimize_symbol,
    optimize_target,
    neptune_client=None,
    neptune_master=None,
    eval=False,
):

    dump_path = (
        root.joinpath(str(BAR_WINDOW))
        .joinpath(SYMBOL)
        .joinpath(str(study_id))
        .joinpath(str(trial_id))
    )
    if not dump_path.exists():
        dump_path.mkdir(parents=True)
    results_by_symbol = {}
    metrics = {}
    core_metrics = [
        "Return [%]",
        "Return (Ann.) [%]",
        "Expectancy [%]",
        "Profit Factor",
        "Win Rate [%]",
        "# Trades/Day",
        "# Trades/Day (long)",
        "# Trades/Day (short)",
        "Calmar Ratio",
        "Buy & Hold Return [%]",
        "Avg. Trade [%]",
        "alpha [%]",
        "Sharpe Ratio",
        "Max. Trade Duration",
        "Avg. Trade Duration",
    ]

    risk_metrics = [
        "Max. Drawdown [%]",
        "Avg. Drawdown [%]",
        "Worst Trade [%]",
        "Max. Drawdown Duration",
        "Avg. Drawdown Duration",
        "beta [%]",
        "Volatility (Ann.) [%]",
    ]

    try:
        results_by_symbol[asset_name] = run_model(
            args,
            asset_name,
            args.plot,
            args.BACKTEST_TRADE_STRATEGY,
            global_dv,
            dump_path,
            **hparams,
        )

        if eval:
            return
        all_metrics = results_by_symbol[optimize_symbol][1]
        for name in risk_metrics + core_metrics:
            metrics[name] = all_metrics[name]
        metrics["objective_value"] = all_metrics[optimize_target]
        for value in metrics.values():
            assert not isinstance(value, float) or (
                not isinf(value) and not isnan(value)
            ), value
        report_metric(study_id, trial_id, hparams, metrics=metrics)
        if neptune_client:
            neptune_client["metrics"] = metrics
            neptune_client["infeasible"] = False
        if neptune_master:
            for key, value in metrics.items():
                neptune_master[f"metrics/{key}"].log(value)

    except Exception:
        print(traceback.format_exc())
        if not eval:
            report_metric(study_id, trial_id, infeasible=True, hparams=hparams)
        if neptune_client:
            neptune_client["infeasible"] = True

    if neptune_client:
        neptune_client["hparams"] = hparams
        neptune_client["trial_id"] = str(trial_id)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    if not args.no_neptune:
        import neptune.new as neptune
    else:
        neptune_client = None

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
        future=args.future,
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
    if not args.eval:
        parameter_specs = [
            spec
            for spec in stra_cls.parameter_specs
            if spec["name"] in stra_cls.parameters
        ]
        study_id = htuner_register_study(
            parameter_specs,
            study_name=args.study_name,
            max_num_trials=args.max_num_trials,
            parameter_constraints=stra_cls.constraints,
        )
    else:
        tags = [
            asset_name,
            str(args.bar_window) + "m",
            strategy_name,
            datetime.now().strftime("%Y%m%d%H%M%S"),
        ]
        study_id = "_".join(tags)
    print(f"register study {study_id}")
    if not args.eval and not args.no_neptune:
        neptune_master = neptune.init(
            project="adamzhang1679/MLTradeAutoML",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MjJkMzM1Mi0yYjdiLTRhMjMtODkwZC1iOTczYzU2YjJmYmEifQ==",
            custom_run_id=f"{study_id}",
            tags=[
                str(study_id),
                "master",
                str(args.bar_window),
                str(args.bar_window),
                str(args.optimize_symbol),
            ],
        )
    else:
        neptune_master = None
    if args.eval:
        hparams = {}
        if args.study_id and args.trial_id:
            with open(
                os.path.join(
                    DUMP_ROOT,
                    args.study_id,
                    args.trial_id,
                    f"{args.optimize_symbol}_hparams.json",
                )
            ) as f:
                hparams = json.load(f)
        hparams["side"] = args.side
        hparams["risk_level"] = float(args.risk_level)
        evaluate_strategy(
            args,
            study_id,
            0,
            global_dv,
            DUMP_ROOT,
            hparams,
            optimize_target=args.optimize_target,
            optimize_symbol=args.optimize_symbol,
            eval=args.eval,
        )
        exit(0)

    while True:
        trial_id, hparams = suggest_trial(study_id, 0)
        if trial_id is None:
            break
        hparams["side"] = args.side
        if not args.no_neptune:
            neptune_client = neptune.init(
                project="adamzhang1679/MLTradeAutoML",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MjJkMzM1Mi0yYjdiLTRhMjMtODkwZC1iOTczYzU2YjJmYmEifQ==",
                custom_run_id=f"{study_id}.{trial_id}",
                tags=[str(study_id), str(args.bar_window), str(args.optimize_symbol)],
            )
        else:
            neptune_client = None

        evaluate_strategy(
            args,
            study_id,
            trial_id,
            global_dv,
            DUMP_ROOT,
            hparams,
            neptune_client=neptune_client,
            neptune_master=neptune_master,
            optimize_target=args.optimize_target,
            optimize_symbol=args.optimize_symbol,
        )
        if not args.no_neptune:
            neptune_client.stop()
