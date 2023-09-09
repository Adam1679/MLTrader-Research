import json
import os
import pickle
import re
from asyncore import write
from datetime import datetime
from fileinput import close

import numpy as np
import pandas as pd
import requests
import talib
from backtesting import Backtest
from backtesting._stats import compute_stats
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, DateRangeSlider, HoverTool, PreText
from bokeh.palettes import Category20_20 as colors
from bokeh.plotting import figure
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from research.strategies import *


def _get_portfolio_return(equity, price, trades):
    portafolio_stats = compute_stats(
        trades=trades,
        equity=equity.values,
        ohlc_data=price.to_frame("Close"),
        strategy_instance=None,
    )

    del portafolio_stats["_trades"]
    del portafolio_stats["_equity_curve"]
    return portafolio_stats


def calc_portfolio_results(results_by_symbol, trade_symbols, dump_dir, stra_name):
    equity_curves = {}
    trades_by_symbol = {}
    close_prices = {}
    for symbol in trade_symbols:
        res = results_by_symbol[symbol]
        results_by_symbol[symbol] = res[1]
        close_prices[symbol] = pd.Series(res[0].Close.values, index=res[0].index)

    # 拿结果

    for symbol in trade_symbols:
        results_by_symbol[symbol]["# Trades"] = int(
            results_by_symbol[symbol]["# Trades"]
        )

        _equity_curve = results_by_symbol[symbol].pop("_equity_curve")
        equity_curves[symbol] = _equity_curve["Equity"]
        trades_by_symbol[symbol] = results_by_symbol[symbol].pop("_trades")

    with open(os.path.join(dump_dir, f"{stra_name}_trades_by_symbol"), "wb") as f:
        pickle.dump(trades_by_symbol, f)

    results_by_symbol_df = pd.DataFrame(results_by_symbol)
    # 分symbol画equity 曲线

    equity_curve = pd.concat(equity_curves, axis=1)
    close_prices = pd.concat(close_prices, axis=1)

    equity_curve = equity_curve.fillna(method="ffill")
    equity_curve = equity_curve / equity_curve.iloc[0]
    selected_trades_by_symbol = pd.concat(
        [trades_by_symbol[k] for k in trade_symbols], axis=0, ignore_index=True
    )

    all_reweight_points = pd.date_range(
        str(equity_curve.index[0] + pd.Timedelta("1W")),
        end=str(equity_curve.index[-1] + pd.Timedelta("1W")),
        freq="W",
    )
    new_weights = equity_curve[trade_symbols].copy()
    portfolio_return = np.log(equity_curve[trade_symbols].copy()).diff(1)
    new_weights.loc[:, :] = 1
    for i in range(len(all_reweight_points) - 1):
        change_time = all_reweight_points[i]
        change_time_until = all_reweight_points[i + 1]
        review_since = change_time - pd.Timedelta("1W")
        performance = portfolio_return.loc[review_since:change_time].copy()
        ratio = performance.sum(axis=0)
        ratio = np.nan_to_num(ratio, posinf=1, neginf=0)
        ratio = np.maximum(ratio, 0)
        ratio = ratio / (ratio.sum() + 1e-12)
        ratio = np.round(ratio, 2)
        ratio[np.argmax(ratio)] -= ratio.sum() - 1
        new_weights.loc[
            change_time + pd.Timedelta("1s") : change_time_until
        ] = ratio.reshape(1, -1)

    new_weights = new_weights.divide(new_weights.sum(axis=1), axis=0)
    new_weights.to_csv(os.path.join(dump_dir, f"{stra_name}_weights.csv"))
    weights = new_weights
    equity_curve["Portfolio"] = (
        equity_curve[trade_symbols].multiply(weights).sum(axis="columns")
    )
    close_prices["Portfolio"] = close_prices[trade_symbols].mean(axis=1)
    results_by_symbol["Portfolio"] = _get_portfolio_return(
        equity_curve["Portfolio"], close_prices["Portfolio"], selected_trades_by_symbol
    )

    # hedge
    # market_return = close_prices[f"Portfolio"] / close_prices[f"Portfolio"].iloc[0]
    # equity_curve["Hedge"] = (equity_curve[f"Portfolio"] - market_return) / 2
    # results_by_symbol["Hedge"] = _get_portfolio_return(
    #     equity_curve["Hedge"], close_prices[f"Portfolio"], selected_trades_by_symbol
    # )

    # equal
    equity_curve["Portfolio-Avg"] = equity_curve[trade_symbols].mean(axis="columns")
    results_by_symbol["Portfolio-Avg"] = _get_portfolio_return(
        equity_curve["Portfolio-Avg"],
        close_prices["Portfolio"],
        selected_trades_by_symbol,
    )

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
    result_df.to_csv(os.path.join(dump_dir, f"{stra_name}_result_df.csv"))

    equity_curve.to_csv(os.path.join(dump_dir, f"{stra_name}_equity_curve.csv"))
    core_metrics = [
        "Return [%]",
        "Return (Ann.) [%]",
        "Expectancy [%]",
        "Profit Factor",
        "Win Rate [%]",
        "# Trades/Day",
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
        index=pd.MultiIndex.from_tuples(indexs), columns=trade_symbols
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
    test_start = equity_curve.index[0]
    end = equity_curve.index[-1]
    slider = DateRangeSlider(
        title="DateRangeSlider", start=test_start, end=end, value=(test_start, end)
    )
    # slider.on_change('value', callback)
    slider.js_link("value", fig.x_range, "start", attr_selector=0)
    slider.js_link("value", fig.x_range, "end", attr_selector=1)
    layout = column(stats, fig, slider)
    output_file(os.path.join(dump_dir, f"{stra_name}_Cumulative_Return.html"))
    save(layout)
    return result_df


def run_model(
    args,
    symbol,
    plot,
    strategy_name,
    dv,
    DUMP_ROOT,
    signals=None,
    signals_L=None,
    signals_S=None,
    config_manager=None,
):
    if signals is not None:
        signal = signals[symbol].copy()
        index = signal.index
        signal.index = pd.to_datetime(signal.index, unit="s")
        params = {
            "signal": signals[symbol],
            "config_manager": config_manager,
        }
    elif signals_L is not None and signals_S is not None:
        signal = signals_L[symbol].copy()
        index = signal.index
        signals_L[symbol].index = pd.to_datetime(signals_L[symbol].index, unit="s")
        signals_S[symbol].index = pd.to_datetime(signals_S[symbol].index, unit="s")
        params = {
            "long_signal": signals_L[symbol],
            "short_signal": signals_S[symbol],
            "config_manager": config_manager,
            "args": args
        }
    else:
        params = {
            "args": args
        }
        index = dv["Close"][symbol].dropna().index

    data = {
        "Close": dv["Close"][symbol].reindex(index=index).fillna(method="ffill"),
        "Open": dv["Open"][symbol].reindex(index=index).fillna(method="ffill"),
        "High": dv["High"][symbol].reindex(index=index).fillna(method="ffill"),
        "Low": dv["Low"][symbol].reindex(index=index).fillna(method="ffill"),
        "Volume": dv["Volume"][symbol].reindex(index=index).fillna(method="ffill"),
    }

    data = pd.concat(data, axis=1)
    data.index = pd.to_datetime(data.index, unit="s")
    stra_cls = eval(strategy_name)
    bt = Backtest(
        data,
        stra_cls,
        commission=float(args.fee),
        margin=1,
        cash=float(args.cash),
        hedging=False,
        trade_on_close=True,
    )

    results = bt.run(**params)
    if plot:
        bt.plot(
            filename=os.path.join(
                DUMP_ROOT, "{}_{}_图.html".format(strategy_name, symbol)
            ),
            results=results,
            resample=False,
            plot_pl=True,
            plot_volume=True,
            plot_return=True,
            plot_equity=False,
            plot_drawdown=False,
            superimpose=False,
            open_browser=False,
            smooth_equity=True,
        )
        results.to_csv(
            os.path.join(DUMP_ROOT, "{}_{}_result.json".format(strategy_name, symbol))
        )
        g_path = os.path.join(DUMP_ROOT, "{}_{}_图.html".format(strategy_name, symbol))
        print(f"graph saved to {os.path.abspath(g_path)}")

    equity_curve_and_signals = results._equity_curve
    for indicator in results._strategy._indicators:
        name = indicator.name
        value = pd.Series(
            data=indicator, index=indicator._opts["index"], name=indicator.name
        )
        equity_curve_and_signals[name] = value

    equity_curve_and_signals.to_csv(
        os.path.join(
            DUMP_ROOT, "{}_{}_equity_signals.csv".format(strategy_name, symbol)
        )
    )
    results._trades.to_csv(
        os.path.join(DUMP_ROOT, "{}_{}_trades.csv".format(strategy_name, symbol))
    )
    print(f"everything saved to {os.path.abspath(DUMP_ROOT)}")
    return bt._data, results.to_dict()


def read_ls_by_symbol(symbol, bar_window, root, fake=True):
    if fake:
        symbol = symbol.replace("USDT", "BUSD")
    path = os.path.join(root, f"{symbol}-5m-ls.csv")
    df = pd.read_csv(
        path,
        header=None,
        names=[
            "symbol",
            "datetime",
            "type",
            "exchange",
            "longAccount",
            "shortAccount",
            "longShortRatio",
        ],
        parse_dates=[
            1,
        ],
    )
    df = df.set_index(["type", "datetime"])
    GlobalPosition = df.loc["LongShortType.GlobalPosition", "longShortRatio"]
    TopAccount = df.loc["LongShortType.TopAccount", "longShortRatio"]
    TopPosition = df.loc["LongShortType.TopPosition", "longShortRatio"]
    GlobalPosition = GlobalPosition.resample(
        "{}min".format(bar_window), kind="timestamp", label="right", closed="left"
    ).last()
    GlobalPosition.name = "GlobalPosition"
    TopAccount = TopAccount.resample(
        "{}min".format(bar_window), kind="timestamp", label="right", closed="left"
    ).last()
    TopAccount.name = "TopAccount"
    TopPosition = TopPosition.resample(
        "{}min".format(bar_window), kind="timestamp", label="right", closed="left"
    ).last()
    TopPosition.name = "TopPosition"
    feature = pd.concat([GlobalPosition, TopAccount, TopPosition], axis=1)
    print(
        f"load longShort data for {symbol} from {feature.index[0]} to {feature.index[-1]}"
    )
    return feature


def read_symbol(
    symbol, bar_window, root, fake=False, start_date=None, end_date=None, future=False
):
    if fake:
        path = os.path.join(root, "{}-1m-fake.csv".format(symbol))
    elif not future:
        path = os.path.join(root, "{}-1m.csv".format(symbol))
    else:
        path = os.path.join(root, "{}-1m-perpetual.csv".format(symbol))

    df = pd.read_csv(
        path,
        header=None,
        names=[
            "open_timestamp",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "close_timestamp",
            "quote_asset_volume",
            "number_of_trade",
            "tbbav",
            "tbqav",
            "ignore",
        ],
        dtype={
            "open_timestamp": int,
            "open_price": float,
            "high_price": float,
            "low_price": float,
            "close_price": float,
            "volume": float,
            "quote_asset_volume": float,
            "tbbav": float,
            "tbqav": float,
            "close_timestamp": int,
            "number_of_trade": int,
        },
    )
    df = df.rename(
        columns={
            "close_price": "Close",
            "open_price": "Open",
            "high_price": "High",
            "low_price": "Low",
            "quote_asset_volume": "Volume",
            "number_of_trade": "Count",
        }
    )

    df.drop_duplicates(subset=["open_timestamp"], inplace=True)
    df["open_timestamp"] = df["open_timestamp"] // 1_000
    df["close_timestamp"] = df["close_timestamp"] // 1_000
    df = df.set_index("open_timestamp", drop=True)
    df = df.sort_index(ascending=True)

    start, end = df.index[0], df.index[-1]
    df: pd.DataFrame = df.reindex(pd.Int64Index(np.arange(start, end + 60, 60)))
    df["datetime"] = pd.to_datetime(df.index, unit="s")
    df = df.set_index("datetime", drop=True)
    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        df = df.loc[start_date:]
    if end_date is not None:
        end_date = pd.Timestamp(end_date)
        df = df.loc[:end_date]
    agg_func = {
        "Close": "last",
        "High": "max",
        "Low": "min",
        "Open": "first",
        "Volume": "sum",
        "Count": "sum",
    }

    df = (
        df[["Open", "Close", "High", "Low", "Volume", "Count"]]
        .resample(
            "{}min".format(bar_window), kind="timestamp", label="left", closed="left"
        )
        .agg(agg_func)  # 假设数据对timestamp时间都是bar open time，而不是close time， 所以都left参数
    )
    df["timestamp"] = df.index.astype(int) // 1_000_000_000
    df["Asset_ID"] = symbol
    assert df.index.duplicated().sum() == 0, symbol
    print(
        "load {} data succeed. {} to {}; bar window={}".format(
            symbol, df.index[0], df.index[-1], bar_window
        )
    )
    return df


def weighted_correlation(a, b, weights):
    w = np.ravel(weights)
    a = np.ravel(a)
    b = np.ravel(b)

    sum_w = np.sum(w)
    mean_a = np.sum(a * w) / sum_w
    mean_b = np.sum(b * w) / sum_w
    var_a = np.sum(w * np.square(a - mean_a)) / sum_w
    var_b = np.sum(w * np.square(b - mean_b)) / sum_w

    cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
    corr = cov / (np.sqrt(var_a * var_b) + 1e-12)
    return corr


def get_ic(total_dv, feature_name, return_name):
    x = total_dv[feature_name]
    y = total_dv[return_name]
    x_mean = np.nanmean(x, axis=1, keepdims=True)
    y_mean = np.nanmean(y, axis=1, keepdims=True)
    nom = np.nansum((x - x_mean) * (y - y_mean), axis=1)
    denom = np.sqrt(
        np.nansum((x - x_mean) ** 2, axis=1) * np.nansum((y - y_mean) ** 2, axis=1)
    )
    return pd.Series(nom / denom, index=x.index)


def get_quantile_return(total_dv, feature_name, return_name, window, q=0.8):
    x = total_dv[feature_name]
    y = total_dv[return_name]
    upper = x.rolling(window).quantile(q, axis=0)
    lower = x.rolling(window).quantile(1 - q, axis=0)
    ret1 = np.where(x > upper, y, 0) + np.where(x < lower, y, 0)
    ret1[ret1 == 0] = np.nan
    ret1 = pd.Series(np.nanmean(ret1, axis=1), index=x.index)
    return ret1


def get_ts_ic(total_dv, feature_name, return_name, rolling_window):
    x = total_dv[feature_name]
    y = total_dv[return_name]  # (T, K)
    return x.rolling(rolling_window).corr(y, pairwise=False)


def validate_one_symble(pred, label):
    if len(np.unique(label)) == 2:
        auc = roc_auc_score(label, pred)
        return auc
    else:
        dummy_weights = np.ones_like(pred)
        corr = weighted_correlation(label, pred, dummy_weights)
        return corr


def neutralize_series(series: pd.Series, by: pd.Series, proportion=1.0):
    """
    neutralize pandas series (originally from the Numerai Tournament)
    """
    scores = np.nan_to_num(series.values).reshape(-1, 1)
    exposures = np.nan_to_num(by.values).reshape(-1, 1)
    exposures = np.hstack(
        (
            exposures,
            np.array([np.mean(np.nan_to_num(series.values))] * len(exposures)).reshape(
                -1, 1
            ),
        )
    )
    correction = proportion * (exposures.dot(np.linalg.lstsq(exposures, scores)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def save_lgb_booster(booster, file_name):
    with open(file_name, "w") as f:
        f.write(booster.model_to_string())


def get_timestamp_partition(close_price: pd.DataFrame):
    t = pd.to_datetime(close_price.index, unit="s")
    hours = np.array(t.hour)
    parition = np.zeros_like(hours)
    parition[(hours >= 12) & (hours < 20)] = 0
    parition[(hours >= 20) | (hours < 4)] = 1
    parition[(hours >= 4) & (hours < 12)] = 2
    value = pd.DataFrame(index=close_price.index, columns=close_price.columns)
    for col in close_price.columns:
        value[col] = parition
    value.where(~close_price.isnull(), inplace=True)
    return value


def get_all_market_fear_index(close_price):
    res = requests.get("https://api.alternative.me/fng/?limit=0").json()["data"]
    for data in res:
        data["timestamp"] = int(data["timestamp"])
        data["datetime"] = datetime.utcfromtimestamp(int(data["timestamp"]))

    df = pd.DataFrame(res)
    df = df.set_index("timestamp")
    map_value = {
        "Fear": 0,
        "Extreme Fear": 1,
        "Greed": 2,
        "Extreme Greed": 3,
        "Neutral": 4,
    }
    df["value_classification"] = df["value_classification"].map(map_value)

    df = df.reindex(index=close_price.index)
    df.fillna(method="ffill", inplace=True)
    value = pd.DataFrame(index=close_price.index, columns=close_price.columns)
    for col in value.columns:
        value[col] = df["value_classification"]
    value.fillna(inplace=True, value=4)
    value.where(~close_price.isnull(), inplace=True)
    return value


if __name__ == "__main__":
    # df = read_ls_by_symbol("XRPBUSD", 15, "/Users/bytedance/binance_data/csv_data")
    df = get_all_market_fear_index()
    print(df.head())
