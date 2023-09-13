import functools
import gzip
import itertools
import os
import re
import traceback
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import _pickle as cPickle
import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dask import compute, delayed
from scipy.stats import kurtosis, skew

from research.orderbook_strategies.utils.product_info import *

BASIK_TICK_FILED = [
    "time",
    "bid",
    "ask",
    "bid.qty",
    "ask.qty",
    "intra.time",
    "wpr",
    "wpr.ret",
    "good",
    "price",
    "qty",
    "quote_qty",
    "active.buy.qty",
    "active.sell.qty",
    "active.buy.quote_qty",
    "active.sell.quote_qty",
    "trades",
]


warnings.simplefilter("ignore")
## path of our program
# SIGNAL_PATH = "/mnt/hgfs/intern"
## path of data
EXP_ROOT = Path("/Volumes/AdamDrive/experiment_root/orderbook_research")
AGG_TRADES_ROOT = Path("/Volumes/AdamDrive/binance_data/data/futures/um/daily/aggTrades")
METRICS_ROOT = Path("/Volumes/AdamDrive/binance_data/data/futures/um/daily/metrics")
def get_cache_path(name):
    res = EXP_ROOT / name
    if not res.exists():
        res.mkdir()
    return res


DATA_PATH = get_cache_path("data")
SIGNAL_PATH = get_cache_path("signal")
SIGNAL_RESULTS_PATH = get_cache_path("signal_results")
CACHE_PATH = get_cache_path("cache")
CORE_NUM = 8


def find_date(name):
    match = re.search(r"\d{4}-\d{2}-\d{2}", name)
    if match is not None:
        date_str = match.group()
        return date_str
    else:
        return None


def get_file_list(product: str, signal_name=None) -> List[Path]:
    assert isinstance(product, str)
    if signal_name is None:
        file_list = list(
            map(lambda x: DATA_PATH / product / x, os.listdir(DATA_PATH / product))
        )  ## files of each day
    else:

        def get_signal_dir(product, signal_name):
            return SIGNAL_PATH / product / signal_name

        file_list = list(
            map(
                lambda x: get_signal_dir(product, signal_name) / x,
                os.listdir(get_signal_dir(product, signal_name)),
            )
        )  ## files of each day
    file_list = [file for file in file_list if find_date(str(file))]
    return file_list


def get_dates_list(product, signal_name=None):
    file_paths: List[Path] = get_file_list(product, signal_name)
    dates = list(
        sorted([find_date(str(file)) for file in file_paths if find_date(str(file))])
    )
    return dates

def get_signal_list_with_period(product):
    signal_path = SIGNAL_PATH / product
    signal_names = []
    for p in signal_path.iterdir():
        if p.is_dir():
            signal_names.append(p.name)
    return signal_names

def get_signal_list_without_period(product):
    signal_path = SIGNAL_PATH / product
    signal_names = []
    for p in signal_path.iterdir():
        if p.is_dir():
            name = ".".join(p.name.split(".")[:-1])
            if len(name) > 0:
                signal_names.append(name)
    signal_names = list(set(signal_names))
    return signal_names

def get_data(product, date_str, columns=None, good=False):
    assert isinstance(product, str)
    assert isinstance(date_str, str)
    assert isinstance(columns, list) or columns is None
    for filename in os.listdir(DATA_PATH / product):
        if date_str in filename:
            file = filename
            break
    else:
        raise ValueError(f"no file found for {product} {date_str}")
    ori_columns = columns
    if columns is not None:
        if good:
            columns = list(set(columns) | {"good"})
        columns = ",".join(columns)

    data = load(DATA_PATH / product / file, columns=columns)
    if good:
        data = data[data["good"]]
    if ori_columns:
        data = data[ori_columns]
    return data

def get_trades_data(product, date_str):
    try:
        path = AGG_TRADES_ROOT / product
        trade_file = None
        for file_path in path.iterdir():
            if date_str in file_path.name:
                trade_file = file_path
                break
        if trade_file is None:
            return None
        df = pd.read_csv(trade_file)
        if "transact_time" not in df.columns:
            df.columns = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
        df["transact_time"] = pd.to_datetime(df["transact_time"], unit='ms')
        df = df.drop_duplicates(subset=["transact_time"], keep="first")
        df = df.set_index("transact_time")
        
        return df
    except:
        print("open csv file failed {}".format(trade_file))
        traceback.print_exc()
        return None

def get_metrics_data(product, date_str):
    path = METRICS_ROOT / product
    trade_file = None
    for file_path in path.iterdir():
        if date_str in file_path.name:
            trade_file = file_path
            break
    if trade_file is None:
        return None
    # create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio
    df = pd.read_csv(trade_file, parse_dates=["create_time"]).set_index("create_time")
    return df

    

    

## parallel generate the distribution of a signal
def par_generate_alldates_signal(
    signal_name, date_list, product, period, overwrite=False
):
    """
    get singal of all dates in date_list. dumsample it by period
    """
    date_list = list(sorted(date_list))
    assert len(date_is_continuous(date_list)) == 0
    from_date, to_date = date_list[0], date_list[-1]
    file_path = (
        get_cache_path("all_dates_signal")
        / product
        / signal_name
        / f"{from_date}_to_{to_date}.pkl"
    )
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    if overwrite or not file_path.exists():
        all_signal = get_good_signal(product, signal_name, all_dates=date_list)
        if all_signal is not None:
            chosen = (np.arange(len(all_signal)) + 1) % period == 0
            all_signal = all_signal[chosen]
            save(all_signal, file_path)
        else:
            print("error: all_signal is None for {} {}".format(product, signal_name))


def auto_get_alldates_signal(signal_name, product):
    all_signal_path = get_cache_path("all_dates_signal") / product / signal_name
    file = None
    try:
        max_range = None
        # find the largest all_signal
        for filename in all_signal_path.glob("*.pkl"):
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", str(filename))
            if len(dates) == 2:
                date_from, date_to = dates
                time_delta = (
                    datetime.strptime(date_to, "%Y-%m-%d")
                    - datetime.strptime(date_from, "%Y-%m-%d")
                ).total_seconds()
                if max_range is None:
                    max_range = time_delta
                if time_delta >= max_range:
                    file = filename
    except StopIteration:
        pass
    assert file is not None, "found no all_signal data for {} {}".format(
        signal_name, product
    )
    all_signal = load(file)  ## get the distribution of the signal
    assert all_signal is not None, "load all_signal failed"
    return all_signal


def get_good_data(
    product, date_from=None, date_to=None, date_str=None, columns=None, use_cache=True
):
    assert date_str is not None or (date_from is not None and date_to is not None)
    if date_str is not None:
        return get_data(product, date_str, good=True)
    else:
        all_dates = get_dates_list(product)
        selected_dates = [date for date in all_dates if date_from <= date <= date_to]
        selected_data = []
        datas = {}
        hash_key = f"get_good_data_{product}_{date_from}_{date_to}"
        file_name = get_cache_path("cache") / f"{hash_key}.pkl"
        if file_name.exists() and use_cache:
            return load(file_name)
        with ThreadPoolExecutor(max_workers=8) as executor:
            for date_str in selected_dates:
                datas[date_str] = executor.submit(
                    get_data, product, date_str, good=True, columns=columns
                )
            executor.shutdown()
        for date_str in selected_dates:
            selected_data.append(datas[date_str].result())
        res = pd.concat(selected_data, axis=0)
        save(res, file_name)
        return res


def get_good_signal(
    product, signal_name, date_from=None, date_to=None, date_str=None, all_dates=None, use_cache=True
):
    assert date_str is not None or (date_from is not None and date_to is not None) or all_dates is not None
    if date_str is not None:
        return get_signal(product, signal_name, date_str, good=True)
    elif all_dates is None:
        all_dates = get_dates_list(product)
        selected_dates = [date for date in all_dates if date_from <= date <= date_to]
    else:
        assert all_dates is not None
        selected_dates = all_dates
    hash_key = f"get_good_signal_{product}_{signal_name}_{date_from}_{date_to}"
    file_name = get_cache_path("cache") / f"{hash_key}.pkl"
    if file_name.exists() and use_cache:
        return load(file_name)

    datas = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for date_str in selected_dates:
            datas[date_str] = executor.submit(
                get_signal, product, signal_name, date_str, good=True
            )
        executor.shutdown()
    selected_data = []
    for date_str in selected_dates:
        res = datas[date_str].result()
        if res is not None:
            selected_data.append(res)
    if len(selected_data) < len(selected_dates) / 2:
        print("warning: too many None in selected_data. Only collect {}/{} dates".format(len(selected_data), len(selected_dates)))
    if len(selected_data) == 0:
        return None
    res = np.concatenate(selected_data, axis=0)
    save(res, file_name)
    return res


def get_signal(product, signal_name, date_str, good=False) -> Optional[np.ndarray]:
    assert isinstance(product, str) and product in product_info.keys()
    assert isinstance(signal_name, str)
    assert isinstance(date_str, str)
    files = find_files_without_suffix(SIGNAL_PATH / product / signal_name, date_str)
    if len(files) == 0:
        print("no file found for {} {} {}".format(product, signal_name, date_str))
        return None
    if len(files) > 1:
        print("warning: found too many files for {} {} {}: {}".format(product, signal_name, date_str, str(files)))
    S = load(SIGNAL_PATH / product / signal_name / files[0])
    assert S is not None, f"{SIGNAL_PATH / product / signal_name / files[0]} not found."
    if good:
        data = get_data(product, date_str, good=False, columns=["good"])
        assert data is not None, "no data {} {}".format(product, date_str)
        S = S[data["good"]]
    return S


def create_signal_root_path(product_list):
    for product in product_list:
        os.makedirs(SIGNAL_PATH / product, exist_ok=True)


@lru_cache(20)
def load(path: Path, columns: Optional[str] = None):

    if columns is not None:
        columns = columns.split(",")
        if len(columns) == 0:
            columns = None

        # remove suffix
    path = Path(path)
    basename = path.name
    if not basename.split(".")[-1] in ["pkl", "parquet", "pd_pkl"]:
        find_files = find_files_without_suffix(path.parent, basename)
        if len(find_files) == 0:
            print("no file found for {}".format(path))
            return None
        if len(find_files) > 1:
            print("warning: found too many files for {}: {}".format(path, str(find_files)))
        path = find_files[0]
    try:
        if str(path).endswith(".pd_pkl"):
            data = pd.read_pickle(path)
            if columns is not None:
                data = data[columns]
            return data
        if str(path).endswith(".parquet"):
            data = pd.read_parquet(path, columns=columns)
            return data
        else:
            with gzip.open(path, "rb", compresslevel=1) as file_object:
                raw_data = file_object.read()
            return cPickle.loads(raw_data)

    except Exception as e:
        print("open file failed: {}, e={}".format(path, e))
        return None


def save(data, path: Path, storage_format="pkl"):
    assert storage_format in ["pkl", "parquet"]
    path = Path(path)
    try:
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"create {path.parent}")
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            if storage_format == "pkl":
                path = path.with_suffix(".pd_pkl")
                pd.to_pickle(data, str(path) + ".unconfirmed.pkl")
                os.rename(str(path) + ".unconfirmed.pkl", str(path))
            elif storage_format == "parquet":
                path = path.with_suffix(".parquet")
                data.to_parquet(str(path) + ".unconfirmed.parquet")
                os.rename(str(path) + ".unconfirmed.parquet", str(path))
            return
        else:
            path = path.with_suffix(".pkl")
            serialized = cPickle.dumps(data)
            unconfirmed_path = str(path) + ".unconfirmed.pkl"
            with gzip.open(unconfirmed_path, "wb", compresslevel=1) as file_object:
                file_object.write(serialized)
            os.rename(unconfirmed_path, str(path))
    except Exception as e:
        print("save file failed: {}".format(path))
        raise e
    finally:
        if os.path.exists(str(path) + ".unconfirmed.pkl"):
            os.remove(str(path) + ".unconfirmed.pkl")


def parLapply(iterable, func, *args, **kwargs):
    with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute([delayed(f_par)(item) for item in iterable])[0]
        return result


def parLapplyV2(iterable1, iterable2, func, *args, **kwargs):
    with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute(
            [
                delayed(f_par)(item1, item2)
                for item1, item2 in itertools.product(iterable1, iterable2)
            ]
        )[0]
        return result


def parLapplyV3(iterable1, iterable2, iterable3, func, *args, **kwargs):
    with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute(
            [
                delayed(f_par)(item1, item2, item3)
                for item1, item2, item3 in itertools.product(
                    iterable1, iterable2, iterable3
                )
            ]
        )[0]
        return result


def parLapplyV4(iterable1, iterable2, iterable3, iterable4, func, *args, **kwargs):
    with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute(
            [
                delayed(f_par)(item1, item2, item3, item4)
                for item1, item2, item3, item4 in itertools.product(
                    iterable1, iterable2, iterable3, iterable4
                )
            ]
        )[0]
        return result


def zero_divide(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.divide(x, y)
    if hasattr(y, "__len__"):
        res[y == 0] = 0
    elif y == 0:
        if hasattr(x, "__len__"):
            res = np.zeros(len(x))
        else:
            res = 0

    return res


## forward selection of signals
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    chosen_signals = []
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} - 1".format(
                "data['" + response + "']",
                " + ".join(selected + ["data['" + candidate + "']"]),
            )
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append("data['" + best_candidate + "']")
            current_score = best_new_score
            chosen_signals.append(best_candidate)
    formula = "{} ~ {} - 1".format("data['" + response + "']", " + ".join(selected))
    model = smf.ols(formula, data).fit()
    return OrderedDict([("model", model), ("chosen.signals", chosen_signals)])


def moving_average(a, n=3):
    ret_sum = np.cumsum(a, dtype=float)
    ret = a
    ret[n:] = (ret_sum[n:] - ret_sum[:-n]) / n
    return ret


## calculate exponential moving avergae
## may different from python's ewm
## warning: adjust is NOT the adjust of ewm
## the adjust of ewm is ALWAYS False
## look-back period is halflife in ewma
## halflife is the period of alpha decay to half
## this is only one method to calculate ewma, there are others, suchas (n-1)/(n+1) and 2/(n+1)
## if adjust=false, the first (period-1) values may be too large
## so we use adjust=true to adjust it, it's divided by aa
## and the first (period-1) values are normal in value
def ewma(x, halflife, init=0, adjust=False):
    # calculate the ewma of x
    init_s = pd.Series(data=init)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = init_s.append(x)
    if adjust:
        xx = range(len(x))
        lamb = 1 - 0.5 ** (1 / halflife)
        aa = 1 - np.power(1 - lamb, xx) * (1 - lamb)
        bb = s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]
        return bb / aa
    else:
        return s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]


def find_files_without_suffix(directory, date_string):
    # Get the list of all files in the directory
    directory = Path(directory)
    if not directory.exists():
        return []
    assert directory.is_dir
    matching_files = []
    for path in directory.iterdir():
        if path.is_file and date_string in str(path):
            matching_files.append(path)
    assert len(matching_files) <= 1, "find too many matched files. {}".format(
        matching_files
    )
    return matching_files


def ewma_lambda(x, lambda_, init=0, adjust=False):
    init_s = pd.Series(data=init)
    s = init_s.append(x)
    if adjust:
        xx = range(len(x))
        aa = 1 - np.power(1 - lambda_, xx) * (1 - lambda_)
        bb = s.ewm(alpha=lambda_, adjust=False).mean().iloc[1:]
        return bb / aa
    else:
        return s.ewm(alpha=lambda_, adjust=False).mean()[1:]


## moving sum of x
## we don't use rollSum because rollSum would make the first n data to be zero
def cum(x: pd.Series, n):
    sum_x = x.cumsum()
    sum_x_shift = sum_x.shift(n)
    sum_x_shift[:n] = 0
    return sum_x - sum_x_shift


def sharpe(x):
    # calculate the annualized sharpe ratio of x.
    # X is a series of daily log return
    return zero_divide(np.mean(x) * np.sqrt(250), np.std(x, ddof=1))


def drawdown(x):
    # x is a series of daily log return
    y = np.cumsum(x)
    return np.max(y) - np.max(y[-1:])


def max_drawdown(x):
    # x is a series of daily log return
    y = np.cumsum(x)
    return np.max(np.maximum.accumulate(y) - y)


def get_sample_signal(
    good_night_files, sample, product, signal_list, period, daily_num
):
    n_samples = sum(daily_num[sample])
    n_signal = len(signal_list)
    all_signal = np.ndarray(shape=(int(n_samples), n_signal))
    cur = 0
    for file in good_night_files[sample]:
        data = load(SIGNAL_PATH + "/night pkl tick/" + product + "/" + file)
        chosen = (np.arange(sum(data["good"])) + 1) % period == 0
        n_chosen = sum(chosen)
        for i in range(n_signal):
            signal_name = signal_list[i]
            S = load(
                SIGNAL_PATH + "/tmp_pkl/" + product + "/" + signal_name + "/" + file
            )
            S = S[data["good"]]
            signal = S[(np.arange(len(S)) + 1) % period == 0]
            signal[np.isnan(signal)] = 0  ## the ret.cor has some bad records
            signal[np.isinf(signal)] = 0  ## the ret.cor has some bad records
            all_signal[cur : (cur + n_chosen), i] = signal
        cur = cur + n_chosen
    all_signal = pd.DataFrame(all_signal, columns=signal_list)
    return all_signal


def vanish_thre(x, thre):
    x[np.abs(x) > thre] = 0
    return x

def par_get_signal_mat(date_str, product, signal_list):
    datas = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for signal in signal_list:
            datas[signal] = executor.submit(get_signal, product, signal, date_str)
        executor.shutdown()
    signal_mat = np.stack([datas[signal].result() for signal in signal_list], axis=1)
    signal_mat[np.isnan(signal_mat)] = 0
    return signal_mat
    
def par_get_daily_pred(date_str, product, coef, bias, strat, overwrite=False):
    if len(find_files_without_suffix(SIGNAL_PATH / product / strat, date_str)) > 0 and not overwrite:
        return
    signal_list = list(coef.keys())
    coef_a = np.array([coef[signal] for signal in signal_list]).reshape(-1)
    signal_mat = par_get_signal_mat(date_str, product, signal_list)
    S = np.dot(signal_mat, coef_a).reshape(-1) + bias
    assert S.shape[0] == signal_mat.shape[0]
    save(S, SIGNAL_PATH / product / strat / date_str)
    

def get_sample_signal(
    good_night_files, sample, product, signal_list, period, daily_num, SIGNAL_PATH
):
    n_samples = sum(daily_num[sample])
    n_days = sum(sample)
    n_signal = len(signal_list)
    all_signal = np.ndarray(shape=(int(n_samples), n_signal))
    cur = 0
    for file in good_night_files[sample]:
        good = load(SIGNAL_PATH + "/good pkl/" + product + "/" + file)
        chosen = (np.arange(sum(good)) + 1) % period == 0
        n_chosen = sum(chosen)
        for i in range(n_signal):
            signal_name = signal_list[i]
            S = load(
                SIGNAL_PATH + "/tmp_pkl/" + product + "/" + signal_name + "/" + file
            )
            S = S[good]
            signal = S[(np.arange(len(S)) + 1) % period == 0]
            signal[np.isnan(signal)] = 0  ## the ret.cor has some bad records
            signal[np.isinf(signal)] = 0  ## the ret.cor has some bad records
            all_signal[cur : (cur + n_chosen), i] = signal
        cur = cur + n_chosen
    all_signal = pd.DataFrame(all_signal, columns=signal_list)
    return all_signal


def get_range_pos(wpr, min_period, max_period, period):
    return (
        ewma(
            zero_divide(wpr - min_period, max_period - min_period), period, adjust=True
        )
        - 0.5
    )


def get_signal_train_stat(
    signal_name,
    thre_mat,
    product,
    all_dates,
    CORE_NUM,
    split_str="2018",
    reverse=1,
    tranct=1.1e-4,
    max_spread=0.61,
    tranct_ratio=True,
    min_pnl=2,
    min_num=20,
    atr_filter=0,
):
    train_sample = all_dates < split_str
    with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
        f_par = functools.partial(
            get_signal_pnl,
            product=product,
            signal_name=signal_name,
            thre_mat=thre_mat,
            reverse=reverse,
            tranct=tranct,
            max_spread=max_spread,
            tranct_ratio=tranct_ratio,
            atr_filter=atr_filter,
        )
        train_result = compute(
            [delayed(f_par)(file) for file in all_dates[train_sample]]
        )[0]
    train_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
    return train_stat


def rsi(ret, period):
    abs_move = np.abs(ret)
    up_move = np.maximum(ret, 0)
    up_total = ewma(up_move, period, adjust=True)
    move_total = ewma(abs_move, period, adjust=True)
    rsi = zero_divide(up_total, move_total) - 0.5
    return rsi


from collections import OrderedDict


def get_list_signal_stat(
    signal_name,
    thre_mat,
    product_list,
    all_dates,
    split_str="2018",
    reverse=1,
    min_pnl=2,
    min_num=20,
    atr_filter=20,
):
    CORE_NUM = int(os.environ["NUMBER_OF_PROCESSORS"])
    train_sample = np.array(all_dates) < split_str
    test_sample = np.array(all_dates) > split_str
    date_str = [n[0:8] for n in all_dates]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    train_trade_stat = dict([])
    print("training")
    for product in product_list:
        spread = product_info[product]["spread"]
        tranct = product_info[product]["tranct"]
        tranct_ratio = product_info[product]["tranct.ratio"]
        with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
            f_par = functools.partial(
                get_signal_pnl,
                product=product,
                signal_name=signal_name,
                thre_mat=thre_mat,
                reverse=reverse,
                tranct=tranct,
                max_spread=spread * 1.1,
                tranct_ratio=tranct_ratio,
                atr_filter=atr_filter,
            )
            train_result = compute(
                [delayed(f_par)(file) for file in np.array(all_dates)[train_sample]]
            )[0]
        trade_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
        train_trade_stat[product] = trade_stat
    print("testing")
    test_trade_stat = dict([])
    for product in product_list:
        spread = product_info[product]["spread"]
        tranct = product_info[product]["tranct"]
        tranct_ratio = product_info[product]["tranct.ratio"]
        with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
            f_par = functools.partial(
                get_signal_pnl,
                product=product,
                signal_name=signal_name,
                thre_mat=thre_mat,
                reverse=reverse,
                tranct=tranct,
                max_spread=spread * 1.1,
                tranct_ratio=tranct_ratio,
                atr_filter=atr_filter,
            )
            result = compute(
                [delayed(f_par)(file) for file in np.array(all_dates)[test_sample]]
            )[0]
        trade_stat = get_hft_summary(result, thre_mat, sum(test_sample))
        test_trade_stat[product] = trade_stat
    result = dict([])
    result["train_trade_stat"] = train_trade_stat
    result["test_trade_stat"] = test_trade_stat
    if reverse == -1:
        save(result, SIGNAL_PATH + "/" + signal_name + ".result.pkl")
    else:
        save(result, SIGNAL_PATH + "/" + signal_name + ".pos.result.pkl")


def get_list_signal_result(
    signal_name,
    product_list,
    all_dates,
    split_str="2018",
    reverse=1,
    tranct=1.1e-4,
    tranct_ratio=True,
    min_pnl=2,
    min_num=20,
    atr_filter=20,
):
    if reverse == -1:
        result = load(SIGNAL_PATH + "/" + signal_name + ".result.pkl")
    else:
        result = load(SIGNAL_PATH + "/" + signal_name + ".pos.result.pkl")
    train_trade_stat = result["train_trade_stat"]
    test_trade_stat = result["test_trade_stat"]
    train_sample = np.array(all_dates) < split_str
    test_sample = np.array(all_dates) > split_str
    date_str = [n[0:8] for n in all_dates]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    i = 0
    test_all_pnl = np.zeros([sum(test_sample), len(product_list)])
    train_all_pnl = np.zeros([sum(train_sample), len(product_list)])
    for product in product_list:
        spread = product_info[product]["spread"]
        trade_stat = train_trade_stat[product]
        good_strat = (trade_stat["final.result"]["avg.pnl"] > min_pnl * spread) & (
            trade_stat["final.result"]["num"] > min_num
        )
        if sum(good_strat) > 0:
            train_pnl = trade_stat["daily.pnl"].loc[:, good_strat].sum(axis=1) / sum(
                good_strat
            )
            train_std = np.std(train_pnl)
            train_pnl = train_pnl / train_std
            trade_stat = test_trade_stat[product]
            test_pnl = (
                trade_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)
                / sum(good_strat)
                / train_std
            )
            print(
                product,
                "train sharpe ",
                sharpe(train_pnl),
                "test sharpe ",
                sharpe(test_pnl),
            )
            test_all_pnl[:, i] = test_pnl
            train_all_pnl[:, i] = train_pnl
            i = i + 1
    if i > 0:
        train_portfolio = np.array(np.mean(train_all_pnl[:, :i], axis=1))
        test_portfolio = np.array(np.mean(test_all_pnl[:, :i], axis=1))
        all_portfolio = np.append(train_portfolio, test_portfolio)
        plt.figure(1, figsize=(16, 10))
        plt.title("")
        plt.xlabel("date")
        plt.ylabel("pnl")
        plt.title("portfolio")
        plt.plot(format_dates, all_portfolio.cumsum())
        plt.plot(format_dates[test_sample], all_portfolio.cumsum()[test_sample])
        signal_stat = dict([])
        signal_stat["train.stat"] = train_trade_stat
        signal_stat["test.stat"] = test_trade_stat
        print(
            "train sharpe: ",
            sharpe(train_portfolio),
            "test sharpe: ",
            sharpe(test_portfolio),
        )


def par_get_arb_all_signal(
    signal_name, file_list, product_x, product_y, period, SIGNAL_PATH="d:/intern"
):
    n_files = len(file_list)
    all_signal = np.array([])
    for file in file_list:
        S_x = load(
            SIGNAL_PATH + "/tmp_pkl/" + product_x + "/" + signal_name + "/" + file
        )
        S_y = load(
            SIGNAL_PATH + "/tmp_pkl/" + product_y + "/" + signal_name + "/" + file
        )
        [time_x, time_y] = load(
            SIGNAL_PATH + "/comb time/" + product_x + "_" + product_y + "/" + file
        )
        signal = S_x[time_x] - S_y[time_y]
        chosen = (np.arange(len(signal)) + 1) % period == 0
        all_signal = np.concatenate((all_signal, signal[chosen]), axis=0)
    save(
        all_signal,
        SIGNAL_PATH
        + "/all signal/"
        + product_x
        + "_"
        + product_y
        + "."
        + signal_name
        + ".pkl",
    )


def add_min_max(file, period_list):
    data = load(file)
    data = data.reset_index(drop=True)
    for period in period_list:
        if not ("min." + str(period) in data.columns):
            data["min." + str(period)] = data["wpr"].rolling(period).min()
            data.loc[: period - 1, ("min." + str(period))] = data["wpr"][0]
        if not ("max." + str(period) in data.columns):
            data["max." + str(period)] = data["wpr"].rolling(period).max()
            data.loc[: period - 1, ("max." + str(period))] = data["wpr"][0]
    save(data, file, storage_format="parquet")


def get_field_data(product_list, field_name):
    # get field for all dates and all products
    all_product_field = dict([])
    for product in product_list:
        dates = get_dates_list(product)
        all_field = get_good_data(
            product, date_from=dates[0], date_to=dates[-1], columns=[field_name]
        )
        all_product_field[product] = all_field
    return all_product_field


def get_signal_data(product_list, signal_name, use_cache=False):
    # get field for all dates and all products
    all_product_field = dict([])
    for product in product_list:
        dates = get_dates_list(product, signal_name=signal_name)
        all_field = get_good_signal(
            product,
            signal_name,
            date_from=dates[0],
            date_to=dates[-1],
            use_cache=use_cache,
        )
        all_product_field[product] = all_field
    return all_product_field


def get_cache(hash_name):
    hash_path = get_cache_path("cache")
    if (hash_path / hash_name).exists():
        return load(hash_path / hash_name)
    else:
        return None


def save_cache(hash_name, data):
    hash_path = get_cache_path("cache")
    cache_size_in_gb = get_file_sizes_in_gb(hash_path)
    save(data, hash_path / hash_name)


def get_file_sizes_in_gb(directory_path):
    file_sizes = 0
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            file_sizes += size
    return file_sizes / (1024**3)


def get_all_dates_signal_data(product_list, signal_name):
    # get signal for all dates and all products
    all_signal_list = []
    all_product_signal = dict([])
    for product in product_list:
        parent_path = SIGNAL_PATH / product / signal_name
        dates = list(sorted(os.listdir(parent_path)))
        date_strs = [find_date(n) for n in dates if find_date(n)]
        from_date = min(date_strs)
        to_date = max(date_strs)
        hash_name = f"get_signal_data.{product}.{signal_name}.{from_date}_to_{to_date}"
        all_signal = get_cache(hash_name)
        if all_signal is None:
            for file in dates:
                if find_date(file):
                    data = load(DATA_PATH / product / file)
                    S = load(SIGNAL_PATH / product / signal_name / file)
                    S = S[data["good"]]
                    all_signal_list.append(S)

            all_signal = np.concatenate(all_signal_list, axis=0)
            save_cache(hash_name, all_signal)
        else:
            print(f"load {hash_name} from cache")
        all_product_signal[product] = all_signal
    return all_product_signal


def get_signal_pnl_close(
    file, product, signal_name, thre_mat, reverse=1, rebate=0, SIGNAL_PATH="d:/intern"
):
    ## load data
    data = load(SIGNAL_PATH + "/night pkl tick/" + product + "/" + file)
    ## load signal
    S = load(SIGNAL_PATH + "/tmp_pkl/" + product + "/" + signal_name + "/" + file)
    ## we don't know the signal is positive correlated or negative correlated
    pred = S * reverse
    pred = pred[data["good"]]
    data = data[data["good"]]
    ## load product info
    tranct = product_info[product]["tranct"] * (1 - rebate)
    min_spread = product_info[product]["spread"] + 0.001
    close = product_info[product]["close"] * (1 - rebate)
    tranct_ratio = product_info[product]["tranct.ratio"]
    result = pd.DataFrame(
        data=OrderedDict(
            [
                ("open", thre_mat["open"].values),
                ("close", thre_mat["close"].values),
                ("num", 0),
                ("avg.pnl", 0),
                ("pnl", 0),
            ]
        ),
        index=thre_mat.index,
    )
    bid_ask_spread = data["ask"] - data["bid"]
    next_spread = bid_ask_spread.shift(-1)
    next_spread.iloc[-1] = bid_ask_spread.iloc[-1]
    not_trade = (
        (data["time"] == "10:15:00")
        | (data["time"] == "11:30:00")
        | (data["time"] == "15:00:00")
        | (bid_ask_spread > min_spread)
        | (next_spread > min_spread)
    )

    for thre in thre_mat.iterrows():
        buy = pred > thre[1]["open"]
        sell = pred < -thre[1]["open"]
        signal = pd.Series(data=0, index=data.index)
        position = signal.copy()
        signal[buy] = 1
        signal[sell] = -1
        signal[not_trade] = 0
        scratch = -thre[1]["close"]
        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[
            (signal == 1) & (data["next.ask"] > 0) & (data["next.bid"] > 0)
        ] = 1
        position_pos[(pred < -scratch) & (data["next.bid"] > 0)] = 0
        position_pos.ffill(inplace=True)
        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[
            (signal == -1) & (data["next.ask"] > 0) & (data["next.bid"] > 0)
        ] = -1
        position_neg[(pred > scratch) & (data["next.ask"] > 0)] = 0
        position_neg.ffill(inplace=True)
        position = position_pos + position_neg
        position.iloc[0] = 0
        position.iloc[-2:] = 0
        change_pos = position - position.shift(1)
        change_pos.iloc[0] = 0
        # change_base = pd.Series(data=0, index=data.index)

        pre_pos = position.shift(1)
        pre_pos.iloc[0] = 0
        open_buy = (pre_pos <= 0) & (position > 0)
        open_sell = (pre_pos >= 0) & (position < 0)
        close_buy = (pre_pos < 0) & (position >= 0)
        close_sell = (pre_pos > 0) & (position <= 0)
        open_buy_pnl = pd.Series(data=0, index=data.index)
        open_sell_pnl = pd.Series(data=0, index=data.index)
        close_buy_pnl = pd.Series(data=0, index=data.index)
        close_sell_pnl = pd.Series(data=0, index=data.index)

        if tranct_ratio:
            open_buy_pnl[open_buy] = -data["next.ask"][open_buy] * (1 + tranct)
            open_sell_pnl[open_sell] = data["next.bid"][open_sell] * (1 - tranct)
            close_buy_pnl[close_buy] = -data["next.ask"][close_buy] * (1 + close)
            close_sell_pnl[close_sell] = data["next.bid"][close_sell] * (1 - close)
        else:
            open_buy_pnl[open_buy] = -data["next.ask"][open_buy] - tranct
            open_sell_pnl[open_sell] = data["next.bid"][open_sell] - tranct
            close_buy_pnl[close_buy] = -data["next.ask"][close_buy] - close
            close_sell_pnl[close_sell] = data["next.bid"][close_sell] - close
        final_pnl = sum(open_buy_pnl + open_sell_pnl + close_buy_pnl + close_sell_pnl)
        num = sum((position != 0) & (change_pos != 0))

        if num == 0:
            avg_pnl = 0
            final_pnl = 0
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            return result
        else:
            avg_pnl = np.divide(final_pnl, num)
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)

    return result


def fast_roll_var(x, period):
    x_ma = cum(x, period) / period
    x2 = x * x
    x2_ma = cum(x2, period) / period
    var_x = x2_ma - x_ma * x_ma
    var_x[var_x < 0] = 0
    return var_x


def fast_roll_cor_ewma(x, y, period):
    x_ma = ewma(x, period)
    x2 = x * x
    x2_ma = ewma(x2, period)
    var_x = x2_ma - x_ma * x_ma
    var_x[var_x < 0] = 0
    y_ma = ewma(y, period)
    y2 = y * y
    y2_ma = ewma(y2, period)
    var_y = y2_ma - y_ma * y_ma
    var_y[var_y < 0] = 0
    upper = ewma(x * y, period) - x_ma * y_ma
    result = zero_divide(upper, np.sqrt(var_x * var_y))
    return result


def fast_roll_cor(x, y, period):
    x_ma = cum(x, period) / period
    x2 = x * x
    x2_ma = cum(x2, period) / period
    var_x = x2_ma - x_ma * x_ma
    var_x[var_x < 0] = 0
    y_ma = cum(y, period) / period
    y2 = y * y
    y2_ma = cum(y2, period) / period
    var_y = y2_ma - y_ma * y_ma
    var_y[var_y < 0] = 0
    # upper = (x-x_ma)*(y-y_ma)
    # result = zero_divide(cum(upper, period), np.sqrt(var_x*var_y))/period
    upper = cum(x * y, period) - period * x_ma * y_ma
    result = zero_divide(upper, np.sqrt(var_x * var_y)) / period
    return result


def fast_ret_cor(ret, period):
    pre_ret = ret.shift(1)
    pre_ret[0] = 0
    rolling_cor = fast_roll_cor(pre_ret, ret, period)
    rolling_cor.iloc[: period - 1] = 0
    return rolling_cor


def fast_ret_cor_ewma2(ret, period):
    pre_ret = ret.shift(1)
    pre_ret[0] = 0
    result = fast_roll_cor_ewma(pre_ret, ret, period) * ewma(ret, period) * period
    result.iloc[: period - 1] = 0
    return result


def fast_ret_cor_ewma(ret, period):
    pre_ret = ret.shift(1)
    pre_ret[0] = 0
    result = fast_roll_cor_ewma(pre_ret, ret, period) * ewma(ret, period) * period
    result = np.asarray(result)
    return result


def vol_cor(ret, qty, period):
    result = fast_roll_cor_ewma(qty, ret, period) * ewma(np.abs(ret), period) * period
    return result


def check_strat_prob(train_pnl, test_pnl, num=10000):
    random.seed([100])
    aa = np.random.standard_normal(num).reshape(-1, num)
    aa.sum(axis=1)


def fcum(x, n, fill=0):
    """future cumsum. Asumming future is filled with `fill` for `n` periods

    Args:
        x (_type_): _description_
        n (_type_): _description_
        fill (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return pd.Series(
        data=cum(pd.concat((x, pd.Series(np.repeat(fill, n))), ignore_index=True), n)
        .shift(-n)[:-n]
        .values,
        index=x.index,
    )


# def get_daily_pred(file_name, product, signal_list, coef, strat, SIGNAL_PATH):
#     signal_mat = load(SIGNAL_PATH+"/signal mat pkl/"+product+"/"+file_name)
#     if len(coef)>1:
#         S = np.dot(signal_mat.T, coef)
#     else:
#         S = signal_mat * coef
#     save(S, SIGNAL_PATH+"/tmp_pkl/"+product+"/"+strat+"/"+file_name)

from scipy.optimize import minimize


def TotalTRC(x, Cov):
    x = np.append(x, 1 - np.sum(x))
    TRC = np.prod((np.dot(Cov, x), x), axis=0)
    if np.sum(x < 0) > 0:
        return 10**12
    else:
        return np.sum((TRC[:, None] - TRC) ** 2)


def risk_parity(Sub, only_diag=False, maxiter=9999):
    m = Sub.shape[1]
    Cov = np.cov(Sub, rowvar=False)
    if only_diag:
        Cov = np.diag(np.diag(Cov))
    res = minimize(
        functools.partial(TotalTRC, Cov=Cov),
        np.repeat(1 / m, m - 1),
        method="Nelder-Mead",
        options={"xtol": 1e-6, "maxiter": maxiter, "disp": True},
    )
    w = np.append(res["x"], 1 - np.sum(res["x"]))
    # res = nelder_mead(functools.partial(TotalTRC, Cov=Cov), np.repeat(1/m, m-1), step=1e-3, no_improve_thr=1e-05)
    # w = np.append(res[0], 1-np.sum(res[0]))
    return w


def get_signal_stat_close(
    signal_name,
    thre_mat,
    product,
    good_night_files,
    split_str="2018",
    reverse=1,
    min_pnl=2,
    min_num=10,
    rebate=0,
):
    train_sample = good_night_files < split_str
    test_sample = good_night_files > split_str
    with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
        f_par = functools.partial(
            ll_close,
            product=product,
            signal_name=signal_name,
            thre_mat=thre_mat,
            reverse=reverse,
            rebate=rebate,
        )
        train_result = compute(
            [delayed(f_par)(file) for file in good_night_files[train_sample]]
        )[0]
    train_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
    good_strat = (train_stat["final.result"]["avg.pnl"] >= min_pnl) & (
        train_stat["final.result"]["num"] >= min_num
    )
    if sum(good_strat) == 0:
        print("no good strategy!")
        return 0
    print("good strategies: \n", good_strat[good_strat], "\n")
    good_pnl = train_stat["daily.pnl"].loc[:, good_strat].sum(axis=1) / sum(good_strat)
    print("train sharpe: ", sharpe(good_pnl), "\n")
    date_str = [n[0:8] for n in good_night_files]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    plt.figure(1, figsize=(16, 10))
    plt.title("train")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[train_sample], good_pnl.cumsum())
    with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
        f_par = functools.partial(
            get_signal_pnl_close,
            product=product,
            signal_name=signal_name,
            thre_mat=thre_mat,
            reverse=reverse,
            rebate=rebate,
        )
        test_result = compute(
            [delayed(f_par)(file) for file in good_night_files[test_sample]]
        )[0]
    test_stat = get_hft_summary(test_result, thre_mat, sum(test_sample))
    test_pnl = test_stat["daily.pnl"].loc[:, good_strat].sum(axis=1) / sum(good_strat)
    print("test sharpe: ", sharpe(test_pnl), "\n")
    plt.figure(2, figsize=(16, 10))
    plt.title("test")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[test_sample], test_pnl.cumsum())
    return OrderedDict(
        [
            ("train.stat", train_stat),
            ("test.stat", test_stat),
            ("good.strat", good_strat),
        ]
    )


def get_signal_stat_roll(
    signal_name,
    thre_mat,
    product,
    good_night_files,
    train_sample,
    test_sample,
    reverse=1,
    min_pnl=2,
    min_num=10,
    CORE_NUM=4,
):
    with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
        f_par = functools.partial(
            get_signal_pnl_close,
            product=product,
            signal_name=signal_name,
            thre_mat=thre_mat,
            reverse=reverse,
        )
        train_result = compute(
            [delayed(f_par)(file) for file in good_night_files[train_sample]]
        )[0]
    train_stat = get_hft_summary(train_result, thre_mat, len(train_sample))
    good_strat = (train_stat["final.result"]["avg.pnl"] >= min_pnl) & (
        train_stat["final.result"]["num"] >= min_num
    )
    if sum(good_strat) == 0:
        print("no good strategy!")
        return 0
    print("good strategies: \n", good_strat[good_strat], "\n")
    good_pnl = train_stat["daily.pnl"].loc[:, good_strat].sum(axis=1) / sum(good_strat)
    print("train sharpe: ", sharpe(good_pnl), "\n")
    date_str = [n[0:8] for n in good_night_files]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    plt.figure(1, figsize=(16, 10))
    plt.title("train")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[train_sample], good_pnl.cumsum())
    with dask.config.set(scheduler="processes", num_workers=CORE_NUM):
        f_par = functools.partial(
            get_signal_pnl_close,
            product=product,
            signal_name=signal_name,
            thre_mat=thre_mat,
            reverse=reverse,
        )
        test_result = compute(
            [delayed(f_par)(file) for file in good_night_files[test_sample]]
        )[0]
    test_stat = get_hft_summary(test_result, thre_mat, len(test_sample))
    test_pnl = test_stat["daily.pnl"].loc[:, good_strat].sum(axis=1) / sum(good_strat)
    print("test sharpe: ", sharpe(test_pnl), "\n")
    plt.figure(2, figsize=(16, 10))
    plt.title("test")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[test_sample], test_pnl.cumsum())
    return OrderedDict(
        [
            ("train.stat", train_stat),
            ("test.stat", test_stat),
            ("good.strat", good_strat),
        ]
    )


def get_daily_gbm(
    file_name,
    product,
    signal_list,
    model,
    strat,
    SIGNAL_PATH,
    SAVE_PATH,
    train_std,
    thre=float("Inf"),
):
    signal_mat = (
        load(SAVE_PATH + "/signal mat pkl/" + product + "/" + file_name).T / train_std
    )
    S = model.predict(signal_mat)
    S[np.abs(S) > thre] = 0
    save(S, SAVE_PATH + "/tmp_pkl/" + product + "/" + strat + "/" + file_name)


from sklearn import linear_model
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def get_multiple_lasso_roll_model(
    train_start,
    train_end,
    y_signal,
    n_signal,
    daily_ticks,
    all_mat,
    forward_len,
    strat,
    single_product,
    combine_product,
    period=4096,
    SAVE_PATH="e:/intern",
):
    product_list = np.append(single_product, combine_product)
    cum_daily_ticks = dict([])
    train_tick_start = dict([])
    train_tick_end = dict([])
    test_tick_start = dict([])
    test_tick_end = dict([])
    for product in product_list:
        cum_daily_ticks[product] = daily_ticks[product].cumsum()
        if train_start == 0:
            train_tick_start[product] = 0
        else:
            train_tick_start[product] = int(
                cum_daily_ticks[product][train_start - 1] + 1
            )
        train_tick_end[product] = int(cum_daily_ticks[product][train_end] - 1)
        test_tick_start[product] = int(train_tick_end[product] + 2)
        test_tick_end[product] = int(cum_daily_ticks[product][train_end + 1])

    coef_list = dict([])
    for product in single_product:
        x_train = (
            all_mat[product]
            .iloc[train_tick_start[product] : train_tick_end[product], :n_signal]
            .values
        )
        y_train = all_mat[product][y_signal][
            train_tick_start[product] : train_tick_end[product]
        ]
        n_train = x_train.shape[0]
        scaler = StandardScaler(copy=True, with_mean=False, with_std=True)
        scaler.fit(x_train)
        x_std = np.sqrt(scaler.var_)
        x_train_normal = scaler.transform(x_train)
        model = LassoCV(n_alphas=100, fit_intercept=False, cv=5, max_iter=10000).fit(
            x_train_normal, y_train
        )
        coef = model.coef_ / x_std
        coef_list[product] = coef
    train_std_mat = dict([])
    train_std_mat
    n_samples = x_train.shape[0]
    train_mat = np.zeros((0, n_signal))
    y_train = np.array([])
    for product in combine_product:
        x_train = (
            all_mat[product]
            .iloc[train_tick_start[product] : train_tick_end[product], :n_signal]
            .values
        )
        cur_y_train = all_mat[product][y_signal][
            train_tick_start[product] : train_tick_end[product]
        ]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        train_std_mat[product] = np.sqrt(scaler.var_)
        train_mat = np.append(train_mat, x_train, axis=0)
        y_train = np.append(y_train, cur_y_train)
    model = LassoCV(n_alphas=100, fit_intercept=False, cv=5, max_iter=10000).fit(
        train_mat, y_train
    )
    fit_coef = model.coef_
    for product in combine_product:
        coef_list[product] = fit_coef / train_std_mat[product]
    os.makedirs(SAVE_PATH + "/model", exist_ok=True)
    save(coef_list, SAVE_PATH + "/model/" + strat + ".pkl")


# def get_glmnet_ensemble_roll_model(
#     train_start, train_end, forward_len, alpha=1, start_year=2018, period=2048
# ):
#     cum_daily_ticks = daily_ticks.cumsum()
#     if train_start == 0:
#         train_tick_start = 0
#     else:
#         train_tick_start = cum_daily_ticks[train_start - 1] + 1
#     train_tick_end = cum_daily_ticks[train_end] - 1
#     test_tick_start = train_tick_end + 2
#     test_tick_end = cum_daily_ticks[train_end + 1]
#     n_signal = len(signal_list)
#     nfold = 10
#     model_coef = np.zeros((n_signal, n_mod))
#     for i_mod in range(n_mod):
#         x_train = train_array[i_mod, :, :n_signal]
#         y_train = train_array[i_mod, :, n_signal]
#         n_train = x_train.shape[0]
#         model = ElasticNetCV(
#             l1_ratio=alpha, n_alphas=100, fit_intercept=False, cv=10, max_iter=1000
#         ).fit(x_train, y_train)
#         model_coef[:, i_mod] = model.coef_
#     coef = np.mean(model_coef, axis=1)
#     if alpha == 1:
#         strat = "lasso.ensemble.roll." + str(start_year) + "." + str(period)
#     elif alpha == 0:
#         strat = "ridge.ensemble.roll." + str(start_year) + "." + str(period)
#     else:
#         strat = "elastic.ensemble.roll." + str(start_year) + "." + str(period)
#     os.makedirs(SIGNAL_PATH + "/roll model", exist_ok=True)
#     os.makedirs(SIGNAL_PATH + "/roll model/" + product, exist_ok=True)
#     save(
#         model_coef,
#         SIGNAL_PATH
#         + "/roll model/"
#         + product
#         + "/"
#         + strat
#         + "."
#         + str(train_start)
#         + "."
#         + str(train_end)
#         + ".pkl",
#     )


def preprecess_parsed_tick(directory, overwrite=False):
    # 处理解析后的tick数据，添加一些特征，添加前面和后面的数据。
    def _add_front_and_end(product, filename):
        try:
            match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
            if match:
                date_str = match.group()
                date = datetime.strptime(date_str, "%Y-%m-%d")
            else:
                raise ValueError("Cannot parse name {}".format(date_str))

            yst_str = (date - timedelta(days=1)).strftime("%Y-%m-%d")
            tmr_str = (date + timedelta(days=1)).strftime("%Y-%m-%d")
            new_p = os.path.join(DATA_PATH, product, date_str + ".pd_pkl")
            find_files = find_files_without_suffix(DATA_PATH / product, date_str)
            if len(find_files):
                filename = find_files[0].name
                if not overwrite:
                    print("found {}. Skip this file".format(filename))
                    return

            if os.path.exists(new_p) and not overwrite:
                return
            yst_path = directory / filename.replace(date_str, yst_str)
            today_path = directory / filename
            tmr_path = directory / filename.replace(date_str, tmr_str)
            data = load(today_path)
            assert data is not None, "no data for {} {}".format(product, date_str)
            data = data.copy()
            data["good"] = True
            yst_data = load(yst_path)
            if yst_data is None or len(yst_data) == 0:
                print(
                    "[{}] skip {} as {} doen't have data".format(
                        product, date_str, yst_str
                    )
                )
                return
            yst_data = yst_data.copy()
            tmr_data = load(tmr_path)
            if tmr_data is None or len(tmr_data) == 0:
                print(
                    "[{}] skip {} as {} doen't have data".format(
                        product, date_str, tmr_str
                    )
                )
                return
            tmr_data = tmr_data.copy()
            assert len(yst_data) > 10000, yst_data.shape
            assert len(tmr_data) > 10000, tmr_data.shape
            yst_data["good"] = False
            tmr_data["good"] = False
            data = pd.concat(
                [yst_data.iloc[-10000:], data, tmr_data.iloc[:10000]], axis=0
            )
            Level = 20
            
            data["intra.time"] = np.array(map(lambda x: x[11:19], data["time"]))
            data["ask.qty"] = (
                data[list(map(lambda x: f"ask_{x}_v", range(Level)))].sum(axis=1)
                / Level
            )
            data["bid.qty"] = (
                data[list(map(lambda x: f"bid_{x}_v", range(Level)))].sum(axis=1)
                / Level
            )
            data["bid"] = data["bid_0_p"]
            data["ask"] = data["ask_0_p"]
            data["wpr"] = (
                data["bid.qty"] * data["ask"] + data["ask.qty"] * data["bid"]
            ) / (data["bid.qty"] + data["ask.qty"])
            data["next.bid"] = data["bid"].shift(-1)
            data["next.bid"].iloc[-1] = data["bid"].iloc[-1]
            data["next.ask"] = data["ask"].shift(-1)
            data["next.ask"].iloc[-1] = data["ask"].iloc[-1]
            pre_wrp = data["wpr"].shift(1)
            pre_wrp[0] = data["wpr"][0]
            data["wpr.ret"] = data["wpr"] - pre_wrp
            data["ret"] = np.log(data["wpr"]) - np.log(pre_wrp)
            for window in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
                data[f"max.{window}"] = data["wpr"].rolling(window).max()
                data[f"min.{window}"] = data["wpr"].rolling(window).min()

            data["atr.4096"] = (data["max.4096"] - data["min.4096"]) / data["wpr"]
            data["atr.1024"] = (data["max.1024"] - data["min.1024"]) / data["wpr"]

            data["std.1024"] = np.sqrt(fast_roll_var(data["wpr"], 1024))
            data["std.4096"] = np.sqrt(fast_roll_var(data["wpr"], 4096))
            float32_cols = ["bid", "ask", "bid.qty", "ask.qty", "wpr", "next.bid", "next.ask", "ret", "wpr.ret", "std.1024", "std.4096", "atr.1024", "atr.4096"]
            for window in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
                float32_cols.append(f"max.{window}")
                float32_cols.append(f"min.{window}")
            for col in float32_cols:
                data[col] = data[col].astype("float32")
            save(data, new_p, storage_format="parquet")
        except Exception as e:
            print(f"errors in {product} {filename}. e={str(e)}")
            traceback.print_exc()

    product = os.path.basename(directory)
    dates = []
    for filename in sorted(os.listdir(directory)):
        if find_date(filename) is not None:
            dates.append(filename)
    parLapply(dates, _add_front_and_end, product)


def vanish_thre(x, thre):
    x[np.abs(x) > thre] = 0
    return x


def fcum(x, n, fill=0):
    return pd.Series(
        data=cum(pd.concat((x, pd.Series(np.repeat(fill, n))), ignore_index=True), n)
        .shift(-n)[:-n]
        .values,
        index=x.index,
    )


def date_is_continuous(dates, date_format="%Y-%m-%d"):
    if len(dates) < 2:
        return []

    previous_date = datetime.strptime(dates[0], date_format)
    missing_date = []
    for date_str in dates[1:]:
        current_date = datetime.strptime(date_str, date_format)

        # Check if the difference between current date and previous date is exactly one day
        if (current_date - previous_date) != timedelta(days=1):
            while previous_date < (current_date + timedelta(days=1)):
                previous_date += timedelta(days=1)
                missing_date.append(previous_date.strftime(date_format))

        previous_date = current_date

    return missing_date


def ffill(arr, axis=0):
    shape = arr.shape
    if np.ndim(arr) == 1:
        arr = arr.reshape(-1, 1)
    df = pd.DataFrame(arr)
    df.fillna(method="ffill", axis=axis, inplace=True)
    out = df.values.reshape(shape)
    return out


def shift(arr, n, axis=0):
    arr = np.roll(arr, n)
    arr[:n] = np.nan
    return arr


def htan(x, loc=0, scale=1):
    x = (x - loc) * scale
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
