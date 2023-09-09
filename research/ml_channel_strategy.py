import json
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, "./")
sys.path.insert(0, "./research")
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from backtesting._stats import compute_stats
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.preprocessing import quantile_transform
from tqdm.notebook import tqdm
from utility import *

from feature_transformer.signals_vec import *
from feature_transformer.utility import get_features_vec

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
lgb.register_logger(logger)
import warnings

from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, DateRangeSlider, HoverTool, PreText
from bokeh.palettes import Category20_20 as colors
from bokeh.plotting import figure

from vnpy.trader.config_manager import ConfigManager

warnings.filterwarnings("ignore")
import shutil
from pathlib import Path

import yaml

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


config_manager = ConfigManager(path=f"research/configs", config_name=args.CONFIG_NAME)
ALL_CONFIG = config_manager.all_config
ALL_CONFIG_PATH = str(Path("research/configs").joinpath(args.CONFIG_NAME))


class Global:
    MANAGER: ConfigManager = config_manager
    SYMBOLS = ALL_CONFIG["SYMBOLS"]
    BACKTEST_TRADE_STRATEGY = (
        ALL_CONFIG["TRADER_MODEL"]["BACKTEST_TRADE_STRATEGY"]
        if args.BACKTEST_TRADE_STRATEGY == ""
        else args.BACKTEST_TRADE_STRATEGY
    )
    TARGET_TRANSFORM = (
        ALL_CONFIG["ML_MODEL"]["TARGET_TRANSFORM"]
        if args.TARGET_TRANSFORM == ""
        else args.TARGET_TRANSFORM
    )
    BAR_WINDOW = ALL_CONFIG["BAR_WINDOW"]
    PRED_MINUTES = ALL_CONFIG["ML_MODEL"]["PRED_MINUTES"]
    TRAIN_DATE_FROM = (
        ALL_CONFIG["ML_MODEL"]["TRAIN_DATE_FROM"]
        if args.TRAIN_DATE_FROM == ""
        else args.TRAIN_DATE_FROM
    )
    TRAIN_TEST_GAP_IN_DAYS = ALL_CONFIG["ML_MODEL"]["TRAIN_TEST_GAP_IN_DAYS"]
    TEST_DATE_FROM = args.TEST_DATE_FROM
    TEST_DATE_TO = args.TEST_DATE_TO

    N_MINS_ONE_DAY = 60 * 24
    N_SECS_ONE_DAY = N_MINS_ONE_DAY * 60
    N_BAR = PRED_MINUTES // BAR_WINDOW
    N_BARS_ONE_DAY = N_MINS_ONE_DAY // BAR_WINDOW
    MODEL_PARAMS = ALL_CONFIG["ML_MODEL"]["MODEL_PARAMS"]
    CONFIG = ALL_CONFIG["FEATURE_CONFIG"]["RAW_FEATURES"]
    GLOBAL_DIVIDE_BASE = None
    AVERAGED_FEATURE = ALL_CONFIG["FEATURE_CONFIG"]["AVERAGED_FEATURES"]
    TS_AVERAGED_FEATURE = ALL_CONFIG["FEATURE_CONFIG"]["TS_AVERAGED_FEATURES"]
    TS_AVERAGED_WINDOW = ALL_CONFIG["FEATURE_CONFIG"]["TS_AVERAGED_WINDOW"]
    FEATURE_NAMES = None
    ASSET_NAME_TO_ID = {v: i for i, v in enumerate(SYMBOLS)}
    ASSET_ID_TO_NAME = {v: k for k, v in ASSET_NAME_TO_ID.items()}
    WEIGHTS = pd.Series(1 / len(SYMBOLS), index=SYMBOLS)
    TRADE_WEIGHTS = None
    MODEL_TYPE = ALL_CONFIG["ML_MODEL"]["MODEL_TYPE"]


_global = Global()

if __name__ == "__main__":
    print("DUMP_ROOT = ", DUMP_ROOT)
    if P.exists():
        shutil.rmtree(DUMP_ROOT)
    P.mkdir(parents=True, exist_ok=True)
    if ALL_CONFIG["TRADER_MODEL"]["WEIGHTS"] is not None:
        _global.TRADE_WEIGHTS = np.array(ALL_CONFIG["TRADER_MODEL"]["WEIGHTS"])
        _global.TRADE_WEIGHTS = _global.TRADE_WEIGHTS / _global.TRADE_WEIGHTS.sum()

    assert _global.BACKTEST_TRADE_STRATEGY in globals()
    if len(args.TRADE_STRATEGYS) != 0:
        for stra in args.TRADE_STRATEGYS:
            assert stra in globals()
    for s in _global.SYMBOLS:
        assert s in _global.SYMBOLS, str(s) + "_" + str(_global.SYMBOLS)
    new_dfs = {}
    GLOBAL_DIVIDE_BASE = defaultdict(dict)

    print("load data ...", end="")
    for asset_name in _global.SYMBOLS:
        sub_df = read_symbol(
            asset_name, _global.BAR_WINDOW, root=DATA_ROOT, fake=args.fake_data
        )
        if sub_df.size == 0:
            raise ValueError()
        for c in sub_df.columns:
            # 归一化
            if c in ["Close", "High", "Low", "Open"]:
                GLOBAL_DIVIDE_BASE[asset_name][c] = float(sub_df["Close"].iloc[0])
            elif c in ["Count"]:
                GLOBAL_DIVIDE_BASE[asset_name][c] = float(sub_df[c].iloc[0])
        for c in sub_df.columns:
            if c in ["Close", "High", "Low", "Open", "Count"]:
                sub_df[c] /= GLOBAL_DIVIDE_BASE[asset_name][c]
        new_dfs[asset_name] = sub_df

    print("done")
    GLOBAL_DIVIDE_BASE = dict(GLOBAL_DIVIDE_BASE)
    _global.GLOBAL_DIVIDE_BASE = GLOBAL_DIVIDE_BASE
    with open(os.path.join(DUMP_ROOT, "GLOBAL_DIVIDE_BASE.json"), "w") as f:
        f.write(json.dumps(GLOBAL_DIVIDE_BASE, indent=1))
    result_df = pd.concat(new_dfs, axis=0, names=["Asset_ID"])
    dv = {}
    for feature in result_df.columns:
        dv[feature] = result_df.pivot(
            index="timestamp", columns="Asset_ID", values=feature
        )
        dv[feature] = dv[feature].fillna(method="ffill")
    print("get features ...", end="")
    dv_features = get_features_vec(
        dv,
        ALL_CONFIG["FEATURE_CONFIG"],
    )
    print("done")
    feature_names = sorted(list(dv_features.keys()))

    print("FEATURE_NAMES = ", str(feature_names))
    print("#feature = ", str(len(feature_names)))
    future_high = dv["High"].rolling(_global.N_BAR).max().shift(-_global.N_BAR)
    future_low = dv["Low"].rolling(_global.N_BAR).min().shift(-_global.N_BAR)
    tp = _global.MANAGER.take_profit
    sl = _global.MANAGER.stop_loss
    dv["target_long"] = (((future_high - dv["Close"]) / dv["Close"]) > tp).astype(float)
    dv["target_short"] = (((dv["Close"] - future_low) / dv["Close"]) > tp).astype(float)
    feature_by_asset = {}
    for asset_name in tqdm(_global.SYMBOLS):
        features = {}
        features["target_long"] = dv["target_long"][asset_name]
        features["target_short"] = dv["target_short"][asset_name]
        for feat in dv_features.keys():
            features[feat] = dv_features[feat][asset_name]
        features = pd.concat(features, axis=1)
        features = features.fillna(method="ffill").dropna()
        assert features.isnull().sum().sum() == 0
        assert not np.any(np.isnan(features))
        features = features.reset_index()
        features["Asset_ID"] = _global.ASSET_NAME_TO_ID[asset_name]
        feature_by_asset[asset_name] = features

        n_features = len(feature_names)

    pred_by_symbol_L = {}
    pred_by_symbol_S = {}
    _global.FEATURE_NAMES = feature_names
    test_start_date = _global.TEST_DATE_FROM
    test_end_date = _global.TEST_DATE_TO
    for asset_name in tqdm(_global.SYMBOLS):
        feature_df = feature_by_asset[asset_name]
        timestamp = feature_df["timestamp"].astype(int)
        target_long = feature_df["target_long"].astype(float)
        target_short = feature_df["target_short"].astype(float)
        asset_id = feature_df["Asset_ID"].astype(int)
        feature_df = feature_df[feature_names].astype(float)
        size = feature_df.shape[0]

        if not test_end_date:
            test_end_date = pd.to_datetime(timestamp.max(), unit="s")
        train_start = _global.TRAIN_DATE_FROM
        start = int(pd.Timestamp(test_start_date).to_pydatetime().timestamp())
        train_end = start - _global.TRAIN_TEST_GAP_IN_DAYS * _global.N_SECS_ONE_DAY
        train_end_date = pd.to_datetime(train_end, unit="s")
        if train_start != "":
            train_start_date = pd.Timestamp(train_start).to_pydatetime()
            train_start = int(pd.Timestamp(train_start).to_pydatetime().timestamp())
        else:
            train_start = timestamp.min()
            train_start_date = pd.to_datetime(train_start, unit="s")

        stop = (
            int(pd.Timestamp(test_end_date).to_pydatetime().timestamp())
            + _global.N_SECS_ONE_DAY
        )
        test_index = np.where((timestamp >= start) & (timestamp < stop))[0]
        train_index = np.where(((timestamp < train_end) & (timestamp >= train_start)))[
            0
        ]

        print(f"[{asset_name}] test period={test_start_date} to {test_end_date}")
        print(f"[{asset_name}] train period={train_start_date} to {train_end_date}")
        print(f"[{asset_name}] train size={len(train_index)}")
        print(f"[{asset_name}] test size={len(test_index)}")

        train_set_long = lgb.Dataset(
            feature_df.iloc[train_index],
            label=target_long.iloc[train_index],
            free_raw_data=True,
            feature_name=feature_names,
        )
        train_set_short = lgb.Dataset(
            feature_df.iloc[train_index],
            label=target_short.iloc[train_index],
            free_raw_data=True,
            feature_name=feature_names,
        )
        test_set_long = lgb.Dataset(
            feature_df.iloc[test_index],
            label=target_long.iloc[test_index],
            free_raw_data=True,
            feature_name=feature_names,
        )
        test_set_short = lgb.Dataset(
            feature_df.iloc[test_index],
            label=target_short.iloc[test_index],
            free_raw_data=True,
            feature_name=feature_names,
        )

        booster_long = lgb.train(
            train_set=train_set_long,
            params=_global.MODEL_PARAMS,
            valid_sets=[train_set_long, test_set_long],
            feval=pearson_eval,
        )
        booster_short = lgb.train(
            train_set=train_set_short,
            params=_global.MODEL_PARAMS,
            valid_sets=[train_set_short, test_set_short],
            feval=pearson_eval,
        )

        FEAT_GAIN_DF_long = dict(
            zip(feature_names, booster_long.feature_importance("gain"))
        )
        FEAT_GAIN_DF_short = dict(
            zip(feature_names, booster_short.feature_importance("gain"))
        )

        pd.Series(FEAT_GAIN_DF_long).sort_values().to_csv(
            os.path.join(DUMP_ROOT, "FEAT_GAIN_DF_{}.csv".format(asset_name))
        )
        asset_id = _global.ASSET_NAME_TO_ID[asset_name]
        asset_name = _global.ASSET_ID_TO_NAME[asset_id]

        asset_pred_long = booster_long.predict(feature_df.iloc[test_index])
        asset_pred_short = booster_short.predict(feature_df.iloc[test_index])
        asset_t = timestamp[test_index]

        asset_label = target_long.iloc[test_index]
        score = validate_one_symble(asset_pred_long, asset_label)
        print("(L) {}: {}".format(asset_name, score))
        score = validate_one_symble(asset_pred_short, asset_label)
        print("(S) {}: {}".format(asset_name, score))
        pred_by_symbol_L[asset_name] = pd.Series(asset_pred_long, index=asset_t)
        pred_by_symbol_S[asset_name] = pd.Series(asset_pred_short, index=asset_t)

    # 生成信号和阈值
    signals_L = pd.concat(pred_by_symbol_L, axis=1)
    signals_S = pd.concat(pred_by_symbol_S, axis=1)

    # save model
    model_path_L = os.path.join(
        DUMP_ROOT, "lgb-{}-{}-long".format(test_start_date, test_end_date)
    )
    model_path_S = os.path.join(
        DUMP_ROOT, "lgb-{}-{}-short".format(test_start_date, test_end_date)
    )
    print("LGB model saved to {} and {}".format(model_path_L, model_path_S))
    save_lgb_booster(booster_long, model_path_L)
    save_lgb_booster(booster_short, model_path_S)

    results_by_symbol = {}
    equity_curves = {}
    if len(args.TRADE_STRATEGYS) == 0:
        strategies = [_global.BACKTEST_TRADE_STRATEGY]
    else:
        strategies = args.TRADE_STRATEGYS

    for stra in strategies:
        print("test strategies ", stra)
        # 并行跑回测
        executor = ProcessPoolExecutor()
        for symbol in _global.SYMBOLS:
            results_by_symbol[symbol] = executor.submit(
                run_model,
                symbol,
                args.plot,
                stra,
                _global,
                dv,
                DUMP_ROOT,
                signals_L=signals_L,
                signals_S=signals_S,
            )

        executor.shutdown()
        close_prices = {}
        market_trades = {}
        for symbol in _global.SYMBOLS:
            res = results_by_symbol[symbol].result()
            results_by_symbol[symbol] = res[1]
            close_prices[symbol] = pd.Series(res[0].Close.values, index=res[0].index)

        # 拿结果
        for symbol in _global.SYMBOLS:
            results_by_symbol[symbol]["# Trades"] = int(
                results_by_symbol[symbol]["# Trades"]
            )
            results_by_symbol[symbol]["# Trades/day"] = results_by_symbol[symbol][
                "# Trades"
            ] / (results_by_symbol[symbol]["Duration"].total_seconds() // 3600 // 24)
            results_dict = results_by_symbol[symbol]
            _equity_curve = results_by_symbol[symbol].pop("_equity_curve")
            equity_curves[symbol] = _equity_curve["Equity"]
            market_trades[symbol] = results_by_symbol[symbol].pop("_trades")

        with open(os.path.join(DUMP_ROOT, "market_trades"), "wb") as f:
            pickle.dump(market_trades, f)

        results_by_symbol_df = pd.DataFrame(results_by_symbol)
        # 分symbol画equity 曲线

        equity_curve = pd.concat(equity_curves, axis=1)
        close_prices = pd.concat(close_prices, axis=1)

        cols = _global.SYMBOLS
        equity_curve = equity_curve.fillna(method="ffill")

        equity_curve = equity_curve / equity_curve.iloc[0]
        best_symbols = equity_curve.iloc[-1].sort_values(ascending=False).index
        # for i in range(2, len(_global.SYMBOLS) + 1):
        # portfolio results
        i = len(_global.SYMBOLS)
        symbols = best_symbols[:i]
        selected_market_trades = pd.concat([market_trades[k] for k in symbols], axis=0)
        all_reweight_points = pd.date_range(
            str(equity_curve.index[0] + pd.Timedelta("1W")),
            end=str(equity_curve.index[-1]),
            freq="W",
        )
        new_weights = equity_curve[symbols].copy()
        portfolio_return = np.log(equity_curve[symbols].copy()).diff(1)
        new_weights.loc[:, :] = 1
        for i in range(len(all_reweight_points) - 1):
            change_time = all_reweight_points[i]
            change_time_until = all_reweight_points[i + 1]
            review_since = change_time - pd.Timedelta("1W")
            performance = portfolio_return.loc[review_since:change_time].copy()
            ratio = performance.sum(axis=0)
            ratio = np.nan_to_num(ratio, posinf=1, neginf=0)
            ratio = np.maximum(ratio, 0)
            ratio = ratio / ratio.sum()
            ratio = np.round(ratio, 2)
            ratio[np.argmax(ratio)] -= ratio.sum() - 1
            new_weights.loc[
                change_time + pd.Timedelta("1s") : change_time_until
            ] = ratio.reshape(1, -1)
        new_weights = new_weights.divide(new_weights.sum(axis=1), axis=0)
        new_weights.to_csv(os.path.join(DUMP_ROOT, "weights.csv"))
        weights = new_weights
        equity_curve[f"Portfolio"] = (
            portfolio_return[symbols].multiply(weights).sum(axis="columns")
        ).cumsum()
        close_prices[f"Portfolio"] = close_prices[symbols].mean(axis=1)
        portafolio_stats = compute_stats(
            trades=selected_market_trades,
            equity=equity_curve[f"Portfolio"].values,
            ohlc_data=close_prices[f"Portfolio"].to_frame("Close"),
            strategy_instance=None,
        )

        del portafolio_stats["_trades"]
        del portafolio_stats["_equity_curve"]
        results_by_symbol[f"Portfolio"] = portafolio_stats

        # hedge
        market_return = close_prices[f"Portfolio"] / close_prices[f"Portfolio"].iloc[0]
        equity_curve["Hedge"] = (equity_curve[f"Portfolio"] - market_return) / 2
        hedge_stats = compute_stats(
            trades=selected_market_trades,
            equity=equity_curve["Hedge"].values,
            ohlc_data=close_prices[f"Portfolio"].to_frame("Close"),
            strategy_instance=None,
        )
        del hedge_stats["_trades"]
        del hedge_stats["_equity_curve"]
        results_by_symbol["Hedge"] = hedge_stats
        all_values = []

        # data = df
        no_float_columns = [
            "Start",
            "End",
            "Duration",
            "Max. Drawdown Duration",
            "Avg. Drawdown Duration",
            "Max. Trade Duration",
            "Avg. Trade Duration",
            "_strategy",
        ]

        result_df = pd.DataFrame(results_by_symbol).T
        float_type = {}
        for col in result_df.columns:
            if col not in no_float_columns:
                float_type[col] = "float"

        result_df = result_df.astype(float_type)

        result_df.index.name = "Symbols"
        result_df.columns.name = "Metrics"
        result_df.to_csv(os.path.join(DUMP_ROOT, "result_df.csv"))
        equity_curve.to_csv(os.path.join(DUMP_ROOT, "equity_curve.csv"))
        core_metrics = [
            "Return [%]",
            "Return (Ann.) [%]",
            "Expectancy [%]",
            "Profit Factor",
            "Win Rate [%]",
            "# Trades/day",
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
        indexs = [("Profit Metrics", col) for col in core_metrics] + [
            ("Risk Metrics", col) for col in risk_metrics
        ]
        metric_df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(indexs), columns=_global.SYMBOLS
        )
        for symbol in result_df.index:
            for metric in core_metrics:
                metric_df.loc[("Profit Metrics", metric), symbol] = result_df.loc[
                    symbol, metric
                ]
            for metric in risk_metrics:
                metric_df.loc[("Risk Metrics", metric), symbol] = result_df.loc[
                    symbol, metric
                ]
        stats = PreText(text=str(metric_df), width=1600)
        fig = figure(
            width=1600,
            height=800,
            title="Cumulative Return",
            x_axis_label="Datetime",
            y_axis_label="Return",
            x_axis_type="datetime",
            tools="xpan,xwheel_zoom,box_zoom,undo,redo,reset,save",
        )
        equity_curve.index.name = "datetime"
        source = ColumnDataSource(equity_curve)
        for i, col in enumerate(equity_curve.columns):
            plot1 = fig.line(
                "datetime",
                col,
                source=source,
                legend_label=col,
                line_width=1,
                color=colors[i],
                name=col,
            )
            fig.add_tools(
                HoverTool(
                    renderers=[plot1],
                    tooltips=[
                        ("Return", "@{}".format(col)),
                        ("Date", "@datetime{%c}"),
                        ("Name", "$name"),
                    ],
                    mode="vline",
                    formatters={"@datetime": "datetime"},
                )
            )

        fig.legend.location = "top_left"
        fig.legend.click_policy = "hide"
        start = equity_curve.index[0]
        end = equity_curve.index[-1]
        slider = DateRangeSlider(
            title="DateRangeSlider", start=start, end=end, value=(start, end)
        )
        # slider.on_change('value', callback)
        slider.js_link("value", fig.x_range, "start", attr_selector=0)
        slider.js_link("value", fig.x_range, "end", attr_selector=1)
        NBSP = "\N{NBSP}" * 4
        layout = column(stats, fig, slider)
        output_file(os.path.join(DUMP_ROOT, "Cumulative_Return.html"))
        save(layout)
