import json
import os
import sys
from collections import defaultdict

from research.utility import get_all_market_fear_index, get_timestamp_partition

sys.path.insert(0, "./")
sys.path.insert(0, "./research")
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm.notebook import tqdm
from utility import (
    calc_portfolio_results,
    read_symbol,
    run_model,
    save_lgb_booster,
    validate_one_symble,
)

from feature_transformer.utility import get_features_vec

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
lgb.register_logger(logger)
import warnings

from vnpy.trader.config_manager import ConfigManager

warnings.filterwarnings("ignore")
import shutil
from pathlib import Path

from research.parser import arg_parser

EPS = 1e-18


def pearson_eval(preds, train_data):
    """customized lgb evaluation method"""
    labels = np.nan_to_num(train_data.get_label())
    return "corr", pearsonr(labels, np.nan_to_num(preds))[0], True


args = arg_parser.parse_args()
if args.study_name != "":
    P = Path(os.path.join(".vscode", args.study_name))
else:
    P = Path(os.path.join(".vscode", datetime.now().isoformat("#", "seconds")))
DUMP_ROOT = P.absolute()
DATA_ROOT = "/Users/bytedance/binance_data/csv_data"

config_manager = ConfigManager(path="config/configs", config_name=args.CONFIG_NAME)
ALL_CONFIG = config_manager.all_config
ALL_CONFIG_PATH = str(Path("config/configs").joinpath(args.CONFIG_NAME))
BAR_WINDOW = config_manager.bar_window
SYMBOLS = config_manager.symbols
N_MINS_ONE_DAY = 60 * 24
N_SECS_ONE_DAY = N_MINS_ONE_DAY * 60
N_BAR = config_manager.pred_minutes // BAR_WINDOW
N_BARS_ONE_DAY = N_MINS_ONE_DAY // BAR_WINDOW
GLOBAL_DIVIDE_BASE = defaultdict(dict)
FEATURE_NAMES = None
ASSET_NAME_TO_ID = {v: i for i, v in enumerate(config_manager.symbols)}
ASSET_ID_TO_NAME = {v: k for k, v in ASSET_NAME_TO_ID.items()}
WEIGHTS = pd.Series(1 / len(SYMBOLS), index=SYMBOLS)
TEST_DATE_FROM = args.TEST_DATE_FROM
TEST_DATE_TO = args.TEST_DATE_TO
TRAIN_START = config_manager.train_date_from
TEST_START = int(pd.Timestamp(TEST_DATE_FROM).to_pydatetime().timestamp())
TRAIN_END = TEST_START - config_manager.train_test_gap_in_days * N_SECS_ONE_DAY
EVAL_RESULTS = defaultdict(dict)


def get_weekly_return(close_data, rule):
    close_data.index = pd.to_datetime(close_data.index, unit="s")
    weekly_date = close_data.resample(rule, label="left", closed="right").last()
    weekly_return = weekly_date.pct_change(1).shift(-1)
    weekly_return = weekly_return.reindex(index=close_data.index)
    weekly_return.fillna(method="ffill", inplace=True)
    weekly_return.index = weekly_return.index.astype(int) // 1_000_000_000
    return weekly_return


if __name__ == "__main__":
    print("DUMP_ROOT = ", DUMP_ROOT)
    GLOBAL_INFO = {"post_process": {}, "pre_process": {}}
    if P.exists():
        shutil.rmtree(DUMP_ROOT)
    P.mkdir(parents=True, exist_ok=True)
    if len(args.TRADE_STRATEGYS) != 0:
        for stra in args.TRADE_STRATEGYS:
            assert stra in globals()
    for s in config_manager.symbols:
        assert s in config_manager.symbols, str(s) + "_" + str(config_manager.symbols)
    new_dfs = {}

    print("load data ...", end="")
    for symbol_name in config_manager.symbols:
        df_for_each_symbol = read_symbol(
            symbol_name,
            config_manager.bar_window,
            root=DATA_ROOT,
            fake=args.fake_data,
            start_date=datetime(2020, 1, 1),
        )
        df_for_each_symbol.index = df_for_each_symbol.index.tz_localize("UTC")
        if df_for_each_symbol.size == 0:
            raise ValueError()
        for c in df_for_each_symbol.columns:
            # 归一化
            coin_name = config_manager.convert_symbol_to_coin(symbol_name)
            if c in ["Close", "High", "Low", "Open"]:
                GLOBAL_DIVIDE_BASE[coin_name][c] = float(
                    df_for_each_symbol["Close"].iloc[0]
                )
            elif c in ["Count"]:
                GLOBAL_DIVIDE_BASE[coin_name][c] = float(df_for_each_symbol[c].iloc[0])
        for c in df_for_each_symbol.columns:
            if c in ["Close", "High", "Low", "Open", "Count"]:
                df_for_each_symbol[c] /= GLOBAL_DIVIDE_BASE[coin_name][c]
        new_dfs[symbol_name] = df_for_each_symbol

    print("done")
    GLOBAL_DIVIDE_BASE = dict(GLOBAL_DIVIDE_BASE)
    GLOBAL_INFO["pre_process"] = GLOBAL_DIVIDE_BASE

    df_for_all_symbols = pd.concat(new_dfs, axis=0, names=["Asset_ID"])
    dv = {}
    for feature in df_for_all_symbols.columns:
        dv[feature] = df_for_all_symbols.pivot(
            index="timestamp", columns="Asset_ID", values=feature
        )
        dv[feature] = dv[feature].fillna(method="ffill")
    print("get features ...", end="")
    dv_features = get_features_vec(
        dv,
        ALL_CONFIG["FEATURE_CONFIG"],
    )
    print("done")
    FEATURE_NAMES = sorted(list(dv_features.keys()))
    print("FEATURE_NAMES = ", str(FEATURE_NAMES))
    print("#feature = ", str(len(FEATURE_NAMES)))

    dv["target"] = dv["Close"].pct_change(N_BAR).shift(-(N_BAR + 1))
    dv["lag_target"] = dv["Close"].pct_change(N_BAR)

    dv_features["target"] = dv["target"]

    feature_by_asset = {}
    # adx = ADX_VEC(dv, N_MINS_ONE_DAY // config_manager.bar_window)
    # adx = adx.shift(N_MINS_ONE_DAY // config_manager.bar_window)

    n_partitions = 0
    fear_index = get_all_market_fear_index(dv["Close"].copy())
    dv_features["partition_0"] = pd.DataFrame(
        fear_index.values, index=dv["Close"].index, columns=dv["Close"].columns
    )
    n_partitions += 1

    total = 0
    n_features = len(FEATURE_NAMES)
    for symbol_name in tqdm(config_manager.symbols):
        features = {}
        for feat in dv_features.keys():
            features[feat] = dv_features[feat][symbol_name]
        features = pd.concat(features, axis=1)
        features = features.fillna(method="ffill").dropna()
        assert features.isnull().sum().sum() == 0
        assert not np.any(np.isnan(features))
        features = features.reset_index()
        features["Asset_ID"] = ASSET_NAME_TO_ID[symbol_name]
        feature_by_asset[symbol_name] = features
        total += features.shape[0]

    ALL_FEATURES = np.zeros(shape=(total, n_features))
    ALL_TIMESTAMPS = np.zeros(shape=(total,)).astype(int)
    SYMBOLS_IDS = np.zeros(shape=(total,)).astype(int)
    ALL_TARGET = np.zeros(shape=(total,)).astype(float)
    PARTITIONS = [np.zeros(shape=(total,)).astype(int) for _ in range(n_partitions)]

    offset = 0

    # 准备所有的数据成array的形式
    for symbol_name in tqdm(config_manager.symbols):
        feature_df = feature_by_asset[symbol_name]
        timestamp = feature_df["timestamp"].astype(int)
        target = feature_df["target"].astype(float)
        asset_id = feature_df["Asset_ID"].astype(int)
        size = feature_df.shape[0]
        for i in range(n_partitions):
            PARTITIONS[i][offset : offset + size] = (
                feature_df[f"partition_{i}"].astype(int).values
            )

        ALL_TIMESTAMPS[offset : offset + size] = timestamp.values
        SYMBOLS_IDS[offset : offset + size] = asset_id.values
        ALL_FEATURES[offset : offset + size, :] = feature_df[FEATURE_NAMES].values
        ALL_TARGET[offset : offset + size] = target.values

        assert features.isnull().sum().sum() == 0
        offset += size

    pred_by_symbol = defaultdict(list)
    pred_name_by_symbol = defaultdict(list)
    # 生成train，test 的 mask

    if not TEST_DATE_TO:
        TEST_DATE_TO = pd.to_datetime(ALL_TIMESTAMPS.max(), unit="s")

    test_stop = (
        int(pd.Timestamp(TEST_DATE_TO).to_pydatetime().timestamp()) + N_SECS_ONE_DAY
    )
    TEST_INDEX = (ALL_TIMESTAMPS >= TEST_START) & (ALL_TIMESTAMPS < test_stop)

    test_set = lgb.Dataset(
        ALL_FEATURES[TEST_INDEX, :],
        label=ALL_TARGET[TEST_INDEX],
        free_raw_data=False,
        feature_name=FEATURE_NAMES,
    )
    print(f"all symbol test period = {TEST_DATE_FROM} to {TEST_DATE_TO}")
    print("all symbol test size = ", len(TEST_INDEX))
    if TRAIN_START != "":
        train_start_date = pd.Timestamp(TRAIN_START).to_pydatetime()
        TRAIN_START = int(pd.Timestamp(TRAIN_START).to_pydatetime().timestamp())
    else:
        TRAIN_START = np.nanmin(ALL_TIMESTAMPS)
        train_start_date = pd.to_datetime(TRAIN_START, unit="s")
    total_partition = 0

    # clip 极值 并且 remove 一些波动的数据
    train_index = (ALL_TIMESTAMPS < TRAIN_END) & (ALL_TIMESTAMPS >= TRAIN_START)
    if config_manager.target_transform == "clip_20":
        for symbol_name in config_manager.symbols:
            asset_id = ASSET_NAME_TO_ID[symbol_name]
            asset_id_mask = SYMBOLS_IDS == asset_id
            train_features, train_target = (
                ALL_FEATURES[train_index, :],
                ALL_TARGET[train_index].copy(),
            )
            long_t_90 = np.nanquantile(
                ALL_TARGET[train_index & asset_id_mask & (ALL_TARGET > 0)], q=0.9
            )
            long_t_20 = np.nanquantile(
                ALL_TARGET[train_index & asset_id_mask & (ALL_TARGET > 0)], q=0.1
            )
            short_t_90 = np.nanquantile(
                ALL_TARGET[train_index & asset_id_mask & (ALL_TARGET < 0)], q=0.1
            )
            short_t_20 = np.nanquantile(
                ALL_TARGET[train_index & asset_id_mask & (ALL_TARGET < 0)], q=0.9
            )

            ALL_TARGET[train_index & asset_id_mask] = np.clip(
                ALL_TARGET[train_index & asset_id_mask],
                a_min=short_t_90,
                a_max=long_t_90,
            )
            ALL_TARGET[train_index & asset_id_mask] = np.where(
                (ALL_TARGET[train_index & asset_id_mask] >= short_t_20)
                & (ALL_TARGET[train_index & asset_id_mask] <= long_t_20),
                np.nan,
                ALL_TARGET[train_index & asset_id_mask],
            )
            print(
                f"[{symbol_name}] 分位数 {short_t_90} {short_t_20} {long_t_20} {long_t_90}"
            )
    if config_manager.model_type == "classification":
        # ALL_TARGET[train_index & abs(ALL_TARGET) < 4e-4] = np.nan
        # 1) first type
        ALL_TARGET = (ALL_TARGET > 0).astype(float)

    for idx, partition_x in enumerate(PARTITIONS):
        print("new partition {}".format(idx))
        for partition in np.unique(partition_x):
            total_partition += 1
            GLOBAL_INFO["post_process"][total_partition] = {
                config_manager.convert_symbol_to_coin(s): {}
                for s in config_manager.symbols
            }
            GLOBAL_INFO["post_process"][total_partition]["market"] = {}
            train_end_date = pd.to_datetime(TRAIN_END, unit="s")
            filter_mask = partition_x == partition
            train_index = (
                (ALL_TIMESTAMPS < TRAIN_END)
                & (ALL_TIMESTAMPS >= TRAIN_START)
                & filter_mask
            )
            print(f"[{partition}] train size=", len(train_index))
            print(f"[{partition}] train period={train_start_date} to {train_end_date}")
            weight_all = np.ones_like(ALL_TARGET)
            train_features, train_target = (
                ALL_FEATURES[train_index, :],
                ALL_TARGET[train_index].copy(),
            )

            train_weight = weight_all[train_index]
            if config_manager.weighting_strategy == "time-decay":
                train_weights = ALL_TIMESTAMPS[train_index]
                dif_in_weeks = (TRAIN_END - train_weights) / (N_SECS_ONE_DAY * 30)
                train_weight = 1 + np.log(1 + dif_in_weeks) / np.log(dif_in_weeks.max())

            is_nan_mask = np.isnan(train_target)
            train_target_copy = train_target[~is_nan_mask].copy()
            print(
                ("market", partition),
                "_min_",
                train_target_copy.min(),
                "_max_",
                train_target_copy.max(),
            )
            # train_target_copy = (train_target_copy - np.nanmin(train_target_copy)) / (
            #     np.nanmax(train_target_copy) - np.nanmin(train_target_copy)
            # )
            train_set = lgb.Dataset(
                train_features[~is_nan_mask],
                label=train_target_copy,
                free_raw_data=True,
                weight=train_weight[~is_nan_mask],
                feature_name=FEATURE_NAMES,
            )
            booster = lgb.train(
                train_set=train_set,
                params=config_manager.model_params,
                valid_sets=[train_set, test_set],
            )

            GLOBAL_INFO["post_process"][total_partition]["market"][
                "model_path"
            ] = os.path.join(DUMP_ROOT, "market-part_{}.lgb".format(total_partition))
            save_lgb_booster(
                booster,
                GLOBAL_INFO["post_process"][total_partition]["market"]["model_path"],
            )

            FEAT_SPLIT_DF = dict(
                zip(FEATURE_NAMES, booster.feature_importance("split"))
            )
            FEAT_GAIN_DF = dict(zip(FEATURE_NAMES, booster.feature_importance("gain")))
            # feature importance
            pd.Series(FEAT_GAIN_DF).sort_values().to_csv(
                os.path.join(DUMP_ROOT, "FEAT_GAIN_DF.csv")
            )
            pred = booster.predict(ALL_FEATURES[TEST_INDEX])
            test_timestamp = ALL_TIMESTAMPS[TEST_INDEX]

            # 分symbol evaluate一下模型效果
            for symbol_name in config_manager.trade_symbols:
                asset_id = ASSET_NAME_TO_ID[symbol_name]
                symbol_name = ASSET_ID_TO_NAME[asset_id]
                symbol_pred = pred[SYMBOLS_IDS[TEST_INDEX] == asset_id]
                mask = (SYMBOLS_IDS[train_index] == asset_id) & (~is_nan_mask)
                if mask.sum() == 0:
                    mask = np.ones_like(train_target).astype(bool)
                min_v = np.nanmin(train_target[mask])
                max_v = np.nanmax(train_target[mask])
                GLOBAL_INFO["post_process"][total_partition]["market"].setdefault(
                    "train_min", {}
                )
                GLOBAL_INFO["post_process"][total_partition]["market"].setdefault(
                    "train_max", {}
                )
                GLOBAL_INFO["post_process"][total_partition]["market"][
                    "train_min"
                ] = float(min_v)
                GLOBAL_INFO["post_process"][total_partition]["market"][
                    "train_max"
                ] = float(max_v)
                # asset_pred = (asset_pred - min_v) / (max_v - min_v)
                symbol_test_timestamp = ALL_TIMESTAMPS[
                    TEST_INDEX & (SYMBOLS_IDS == asset_id)
                ]
                symbol_test_label = ALL_TARGET[TEST_INDEX & (SYMBOLS_IDS == asset_id)]
                score = validate_one_symble(symbol_pred, symbol_test_label)
                EVAL_RESULTS[("market", partition)][symbol_name] = score

                pred_by_symbol[symbol_name].append(
                    pd.Series(symbol_pred, index=symbol_test_timestamp)
                )
                pred_name_by_symbol[symbol_name].append(("market", partition))

            INDIVIDUAL_MODEL = True
            if INDIVIDUAL_MODEL:
                for symbol_name in config_manager.trade_symbols:
                    asset_id = ASSET_NAME_TO_ID[symbol_name]
                    mask = (SYMBOLS_IDS[train_index] == asset_id) & (~is_nan_mask)
                    if mask.sum() < 100:
                        print(
                            f"no data for pure {symbol_name} under partition [{partition}]"
                        )
                        continue
                    train_target_copy = train_target[mask].copy()
                    print(
                        ("individual", partition),
                        "_min_",
                        train_target_copy.min(),
                        "_max_",
                        train_target_copy.max(),
                    )
                    # train_target_copy = (
                    #     train_target_copy - np.nanmin(train_target_copy)
                    # ) / (np.nanmax(train_target_copy) - np.nanmin(train_target_copy))

                    train_set = lgb.Dataset(
                        train_features[mask],
                        label=train_target_copy,
                        free_raw_data=True,
                        weight=train_weight[mask],
                        feature_name=FEATURE_NAMES,
                    )

                    booster = lgb.train(
                        train_set=train_set,
                        params=config_manager.model_params,
                    )
                    coin_name = config_manager.convert_symbol_to_coin(symbol_name)
                    GLOBAL_INFO["post_process"][total_partition][coin_name][
                        "model_path"
                    ] = os.path.join(
                        DUMP_ROOT, "{}-part_{}.lgb".format(coin_name, total_partition)
                    )
                    save_lgb_booster(
                        booster,
                        GLOBAL_INFO["post_process"][total_partition][coin_name][
                            "model_path"
                        ],
                    )

                    symbol_pred = booster.predict(
                        ALL_FEATURES[(SYMBOLS_IDS == asset_id) & TEST_INDEX]
                    )
                    symbol_test_timestamp = ALL_TIMESTAMPS[
                        TEST_INDEX & (SYMBOLS_IDS == asset_id)
                    ]
                    pred_s = pd.Series(symbol_pred, index=symbol_test_timestamp)
                    min_v = np.nanmin(train_target[mask])
                    max_v = np.nanmax(train_target[mask])
                    GLOBAL_INFO["post_process"][total_partition][coin_name].setdefault(
                        "train_min", {}
                    )
                    GLOBAL_INFO["post_process"][total_partition][coin_name].setdefault(
                        "train_max", {}
                    )
                    GLOBAL_INFO["post_process"][total_partition][coin_name][
                        "train_min"
                    ] = float(min_v)
                    GLOBAL_INFO["post_process"][total_partition][coin_name][
                        "train_max"
                    ] = float(max_v)
                    # pred_s = (pred_s - min_v) / (min_v - max_v)
                    symbol_test_label = ALL_TARGET[
                        TEST_INDEX & (SYMBOLS_IDS == asset_id)
                    ]
                    score = validate_one_symble(symbol_pred, symbol_test_label)
                    EVAL_RESULTS[("individual", partition)][symbol_name] = score
                    pred_by_symbol[symbol_name].append(
                        pd.Series(symbol_pred, index=symbol_test_timestamp)
                    )
                    pred_name_by_symbol[symbol_name].append(("individual", partition))
    pred_by_symbol_reduce_sum = {k: sum(v) for k, v in pred_by_symbol.items()}
    for symbol_name in config_manager.trade_symbols:
        symbol_pred = pred_by_symbol_reduce_sum[symbol_name]
        asset_id = ASSET_NAME_TO_ID[symbol_name]
        symbol_test_timestamp = ALL_TIMESTAMPS[TEST_INDEX & (SYMBOLS_IDS == asset_id)]
        symbol_test_label = ALL_TARGET[TEST_INDEX & (SYMBOLS_IDS == asset_id)]
        score = validate_one_symble(symbol_pred, symbol_test_label)
        EVAL_RESULTS["ensemble"][symbol_name] = score

    EVAL_RESULTS = pd.DataFrame(EVAL_RESULTS)
    print("EVAL_RESULTS: \n", EVAL_RESULTS)
    config_manager.all_config["GLOBAL_INFO"] = GLOBAL_INFO
    config_manager.dump(os.path.join(DUMP_ROOT, "baseline.yaml"))

    for symbol_name in pred_by_symbol:
        for j in range(len(pred_by_symbol[symbol_name])):
            pred_name = pred_name_by_symbol[symbol_name][j]
            p0 = np.min(pred_by_symbol[symbol_name][j])
            p1 = np.percentile(pred_by_symbol[symbol_name][j], 0.25)
            p2 = np.percentile(pred_by_symbol[symbol_name][j], 0.5)
            p3 = np.percentile(pred_by_symbol[symbol_name][j], 0.75)
            p4 = np.max(pred_by_symbol[symbol_name][j])
            print(
                "{} {} stats: ({} {} {} {} {})".format(
                    symbol_name, pred_name, p0, p1, p2, p3, p4
                )
            )

    signals = pd.concat(pred_by_symbol_reduce_sum, axis=1)
    target = dv["target"].reindex(index=signals.index)

    results_by_symbol = {}
    equity_curves = {}
    if len(args.TRADE_STRATEGYS) == 0:
        _trade_stra = (
            ALL_CONFIG["TRADER_MODEL"]["BACKTEST_TRADE_STRATEGY"]
            if args.BACKTEST_TRADE_STRATEGY == ""
            else args.BACKTEST_TRADE_STRATEGY
        )
        strategies = [_trade_stra]
    else:
        strategies = args.TRADE_STRATEGYS

    for stra in strategies:
        print("test strategies ", stra)
        # 并行跑回测
        if args.multi_process:
            executor = ProcessPoolExecutor()
            for symbol_name in config_manager.trade_symbols:
                results_by_symbol[symbol_name] = executor.submit(
                    run_model,
                    symbol_name,
                    args.plot,
                    stra,
                    dv,
                    DUMP_ROOT,
                    signals=signals,
                    config_manager=config_manager,
                )

            executor.shutdown()
            results_by_symbol = {k: v.result() for k, v in results_by_symbol.items()}
        else:
            for symbol_name in config_manager.trade_symbols:
                results_by_symbol[symbol_name] = run_model(
                    symbol_name,
                    args.plot,
                    stra,
                    dv,
                    DUMP_ROOT,
                    signals=signals,
                    config_manager=config_manager,
                )
        calc_portfolio_results(
            results_by_symbol, config_manager.trade_symbols, DUMP_ROOT, stra
        )
