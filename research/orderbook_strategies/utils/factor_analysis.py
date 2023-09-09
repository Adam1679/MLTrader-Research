import inspect
import itertools
import traceback
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import *
from typing import TypedDict
import numpy as np
import pandas as pd
import scipy.stats as stats
from research.orderbook_strategies.utils.helper import *
from research.orderbook_strategies.utils.product_info import product_info

warnings.simplefilter("ignore")
# write a fisher yates algorithm

DailyResult = TypedDict(
    "DailyResult",
    {
        "date": date,
        "num": int,
        "avg_pnl": float,
        "final_pnl": float,
        "avg_ret": float,
        "ret": float,
        "ic.256": float,
        "ic.512": float,
        "ic.1024": float,
        "ic.2048": float,
        # "ic.p90.256": float,
        # "ic.p90.512":float,
        # "ic.p90.1024", float,
        # "ic.p90.2048": float,
        # "ec.256": float,
        # "ec.512":float,
        # "ec.1024", float,
        # "ec.2048": float,
        "open": float,
        "close": float,
        "reverse": float,
        "tranct": float,
        "max_spread": float,
        "atr_filter": float,
        "win_rate": float,
        "profit_factor": float,
    },
)


def print_input_on_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Input: {args}, {kwargs}")
            raise e

    return wrapper


class FactorData(pd.DataFrame):
    @property
    def _constructor(self):
        return FactorData

    @property
    def _constructor_sliced(self):
        return pd.Series

    @property
    def fdate(self):
        return self._fdate

    @fdate.setter
    def fdate(self, value):
        self._fdate = value

    @property
    def fproduct(self):
        return self._fproduct

    @fproduct.setter
    def fproduct(self, value):
        self._fproduct = value

    @property
    def fHEAD_PATH(self):
        return self._fHEAD_PATH

    @fHEAD_PATH.setter
    def fHEAD_PATH(self, value):
        self._fHEAD_PATH = value

    def __getitem__(self, key):
        try:
            s = super().__getitem__(key)
        except KeyError:
            p = self._fHEAD_PATH / self._fproduct / key / self._fdate
            s = load(p)
            if s is not None:
                self[key] = s
            else:
                raise KeyError("key not found {} {} {}".format(p))
        return s


class factor_template(object):
    factor_name = ""

    params = OrderedDict([("period", np.power(2, range(10, 13)))])

    def formula(self):
        pass

    def form_info(self):
        return inspect.getsource(self.formula)

    def info(self):
        info = ""
        info = info + "factor_name:\n"
        info = info + self.factor_name + "\n"
        info = info + "\n"
        info = info + "formula:\n"
        info = info + self.form_info() + "\n"
        info = info + "\n"
        info = info + "params:\n"
        for key in self.params.keys():
            info = info + "$" + key + ":" + str(self.params.get(key)) + "\n"
        return info

    def __repr__(self):
        return self.info()

    def __str__(self):
        return self.info()


def merge_metric_trades_and_construct_indicators(date_str, product, signal_list=[], float32=True, float16=False, overwrite=True):
    try:
        trades_data = get_trades_data(product, date_str)
        assert trades_data is not None, f"trades data is None for {product} {date_str}"
        metric_data = get_metrics_data(product, date_str)
        if metric_data is None:
            print("metric data is None for ", product, date_str)
            return
        trades_data["good"] = True
        date = datetime.strptime(date_str, "%Y-%m-%d")
        yst_str = (date - timedelta(days=1)).strftime("%Y-%m-%d")
        tmr_str = (date + timedelta(days=1)).strftime("%Y-%m-%d")
        yst_data = get_trades_data(product, yst_str)
        if yst_data is None:
            print("yst_data is None for ", product, yst_str)
            return
        yst_data["good"] = False
        yst_metrics = get_metrics_data(product, yst_str)
        if yst_metrics is None:
            print("yst metrics is None for ", product, yst_str)
            return None
        
        tmr_data = get_trades_data(product, tmr_str)
        if tmr_data is None:
            print("tmr_data is None for ", product, tmr_str)
            return
        tmr_data["good"] = False
        tmr_metrics = get_metrics_data(product, tmr_str)
        if tmr_metrics is None:
            print("tmr_metrics is None for ", product, tmr_str)
            return None
        
        metric_data = pd.concat([yst_metrics, metric_data, tmr_metrics], axis=0)
        trades_data = pd.concat([yst_data, trades_data, tmr_data], axis=0)
        names = []
        for name in ['sum_open_interest', 'sum_open_interest_value', 'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']:
            metric_data[f"{name}_log_diff"] = np.log(metric_data[name]).diff()
            metric_data[f"{name}_log_diff"].iloc[0] = 0
            names.append(name)
            names.append(f"{name}_log_diff")
            
        metric_data["more_position.12"] = (metric_data["sum_open_interest_value_log_diff"] > 0).rolling(12).mean()
        names.append("more_position.12")
        assert trades_data.index.isnull().sum() == 0, f"trades data has null index for {product} {date_str}"
        assert metric_data.index.isnull().sum() == 0, f"metric_data has null index for {product} {date_str}"
        merged_data = pd.merge_asof(trades_data, metric_data[names], left_index=True, right_index=True, direction='backward')
        
        signal_names = ['sum_open_interest_value_log_diff', 'count_long_short_ratio', 'count_long_short_ratio_log_diff', 'sum_taker_long_short_vol_ratio', 'sum_taker_long_short_vol_ratio_log_diff']
        
        for signal in signal_list:
            keys = list(signal.params.keys())
            for cartesian in itertools.product(*signal.params.values()):
                signal_name = signal.factor_name
                for i in range(len(cartesian)):
                    signal_name = signal_name.replace(keys[i], str(cartesian[i]))
                signal_path = SIGNAL_PATH / product / signal_name / f"{date_str}.pkl"
                if overwrite or (signal_name not in merged_data and not signal_path.exists()):
                    merged_data[signal_name] = signal.formula(merged_data, *cartesian)
                    signal_names.append(signal_name)

        data = get_data(product, date_str, columns=["time"])
        time = pd.to_datetime(data['time']).dt.tz_localize(None)
        merged_data = merged_data[signal_names]
        merged_data = merged_data.reindex(time, method='ffill')
        for signal_name in signal_names:
            S = merged_data[signal_name].values
            assert len(S) == len(data)
            S = np.asarray(S)
            if float32:
                S = S.astype(np.float32)
            if float16:
                S = S.astype(np.float16)
            signal_path = SIGNAL_PATH / product / signal_name / f"{date_str}.pkl"
            if overwrite or not signal_path.exists():
                save(S, SIGNAL_PATH / product / signal_name / f"{date_str}.pkl")
    except Exception as e:
        print("error in merge_metric_trades_and_construct_indicators", product, date_str)
        traceback.print_exc()
        raise e
        
def build_composite_signal(
    file_name: Path,
    signal_list: List[Any],
    product: str,
    overwrite=False,
    float32=True,
    float16=True,
):
    file_name = Path(file_name)
    assert file_name.exists(), "file not found: " + str(file_name)
    # 根据params生成所有的signal，并且保存到对应的文件夹中
    file_name_str = str(file_name)
    signals_needed = set()
    for signal in signal_list:
        keys = list(signal.params.keys())
        date_str = find_date(file_name_str)
        for cartesian in itertools.product(*signal.params.values()):
            signal_name = signal.factor_name
            for i in range(len(cartesian)):
                signal_name = signal_name.replace(keys[i], str(cartesian[i]))
            path = SIGNAL_PATH / product / signal_name / "{}.pkl".format(date_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or overwrite:
                signals_needed.add((product, signal_name, date_str))
    if len(signals_needed) == 0:
        return
    raw_data = load(file_name)
    assert raw_data is not None
    data = FactorData(raw_data)
    data.fdate = date_str
    data.fproduct = product
    data.fHEAD_PATH = SIGNAL_PATH
    for signal in signal_list:
        keys = list(signal.params.keys())
        date_str = find_date(file_name_str)
        assert date_str is not None, "invalid date file name: " + file_name_str
        for cartesian in itertools.product(*signal.params.values()):
            signal_name = signal.factor_name
            for i in range(len(cartesian)):
                signal_name = signal_name.replace(keys[i], str(cartesian[i]))
            if (product, signal_name, date_str) in signals_needed:
                path = SIGNAL_PATH / product / signal_name / "{}.pkl".format(date_str)
                path.parent.mkdir(parents=True, exist_ok=True)
                S = signal.formula(data, *cartesian)
                assert len(S) == data.shape[0], (
                    "signal length mismatch: " + signal_name + " " + date_str
                )
                S = np.asarray(S)
                if float32:
                    S = S.astype(np.float32)
                if float16:
                    S = S.astype(np.float16)
                assert np.ndim(S) == 1
                save(S, path)


def construct_composite_signal(
    dire_signal_list: List[int],
    range_signal_list: List[int],
    period_list: List[int],
    filter_period: int,
    product: str,
    date_list=None,
):
    assert isinstance(dire_signal_list, list)
    assert isinstance(range_signal_list, list)
    assert isinstance(period_list, list)
    assert isinstance(product, str)

    def make_factor(dire_signal, range_signal):
        class foctor_xx_period(factor_template):
            factor_name = dire_signal + "." + range_signal + f".{filter_period}.period"
            params = OrderedDict([("period", period_list)])

            def formula(self, data, period):
                try:
                    res = (
                        data[dire_signal + "." + str(period)]
                        * data[range_signal + "." + str(filter_period)]
                    )
                    return res
                except Exception as e:
                    print(
                        "error in construct_composite_signal, {} {} {} {}".format(
                            self.factor_name, period, data.fdate, data.fproduct
                        )
                    )
                    traceback.print_exc()
                    raise e

        return foctor_xx_period()

    factor_list = []
    for range_signal in range_signal_list:
        for dire_signal in dire_signal_list:
            xx = make_factor(dire_signal, range_signal)
            factor_list.append(xx)
    if date_list is not None:
        file_list = [DATA_PATH / product / date for date in date_list]
    else:
        file_list = get_file_list(product)
    parLapply(
        file_list, build_composite_signal, signal_list=factor_list, product=product
    )

def _count_continuous_nonzero(arr):
    arr = np.asarray(arr).reshape(-1)
    mask = (arr != 0).astype(float)
    change_indices = np.where(np.diff(mask))[0]

    # Calculate lengths of continuous sequences of positive numbers
    sequence_lengths = np.diff(change_indices)

    # Calculate the average length of the sequences
    average_length = np.nanmean(sequence_lengths)
    return average_length

def backtest_par(
    pred: pd.Series,
    data: pd.DataFrame,
    thre: List[Tuple],
    signal_name: str,
    product: str,
    reverse=1.0,
    atr: Optional[pd.Series] = None,
    atr_filter: Optional[float] = None,
    max_spread=float("inf"),
    tranct_ratio=False,
    tranct=0.0,
) -> List[DailyResult]:
    K = len(thre)
    N = data.shape[0]
    ori_index = data.index
    pred = reverse * pred
    pred = np.asarray(pred)
    future_ret = np.asarray(data["ret"])
    cur_spread = (data["ask"] - data["bid"]).values  # (N,)
    pred = np.repeat(pred.reshape(-1, 1), K, axis=1)  # (N, K)
    future_ret = np.repeat(future_ret.reshape(-1, 1), K, axis=1)  # (N, K)
    
    open_thre = np.array([thre[i][0] for i in range(K)])  # (K,)
    close_thre = np.array([thre[i][1] for i in range(K)])
    buy = pred > open_thre.reshape(1, -1)  # (N, K)
    sell = pred < -open_thre.reshape(1, -1)
    date_ = pd.to_datetime(data["time"].iloc[0]).date()
    signal = np.zeros((N, K))
    position = np.zeros((N, K))
    signal[buy] = 1
    signal[sell] = -1
    if atr is not None and atr_filter is not None:
        atr = np.asarray(atr)
        signal[atr < atr_filter] = 0
    scratch = -close_thre  # (K,)
    position_pos = np.zeros((N, K))
    position_pos.fill(np.nan)
    position_pos[0] = 0
    # mask = (signal == 1)
    # mask = np.where((data["next.ask"] > 0) & (data["next.bid"] > 0) & (cur_spread < max_spread), mask, False)
    position_pos[
        (signal == 1)
        & (data["next.ask"] > 0).values.reshape(-1, 1)
        & (data["next.bid"] > 0).values.reshape(-1, 1)
        & (cur_spread < max_spread).reshape(-1, 1)
    ] = 1
    position_pos[
        (pred < -scratch.reshape(1, -1))
        & (data["next.bid"] > 0).values.reshape(-1, 1)
        & (cur_spread < max_spread).reshape(-1, 1)
    ] = 0
    # use pandas for forward fill
    position_pos = pd.DataFrame(
        data=position_pos, index=ori_index, columns=range(K)
    ).fillna(method="ffill")
    position_pos.ffill(inplace=True)
    pre_pos = position_pos.shift(1)

    position_pos = position_pos.values
    pre_pos = pre_pos.values

    # 同金额
    notional_position_pos = np.zeros((N, K))
    notional_position_pos[position_pos == 1] = 1
    notional_position_pos[(position_pos == 1) & (pre_pos == 1)] = np.nan
    ask_price = np.repeat(data["next.ask"].values.reshape(-1, 1), K, axis=1)
    notional_position_pos[(notional_position_pos == 1)] = (1 / ask_price)[
        (notional_position_pos == 1)
    ]
    notional_position_pos = pd.DataFrame(
        data=notional_position_pos, index=ori_index, columns=range(K)
    ).fillna(method="ffill")
    notional_position_pos = notional_position_pos.values
    position_neg = np.zeros((N, K))
    position_neg.fill(np.nan)
    position_neg[0] = 0
    position_neg[
        (signal == -1)
        & (data["next.ask"] > 0).values.reshape(-1, 1)
        & (data["next.bid"] > 0).values.reshape(-1, 1)
        & (cur_spread < max_spread).reshape(-1, 1)
    ] = -1
    position_neg[
        (pred > scratch.reshape(1, -1))
        & (data["next.ask"] > 0).values.reshape(-1, 1)
        & (cur_spread < max_spread).reshape(-1, 1)
    ] = 0
    position_neg = pd.DataFrame(
        data=position_neg, index=ori_index, columns=range(K)
    ).fillna(method="ffill")
    pre_neg = position_neg.shift(1)
    position_neg = position_neg.values
    pre_neg = pre_neg.values

    notional_position_neg = np.zeros((N, K))
    notional_position_neg[position_neg == -1] = -1
    notional_position_neg[(position_neg == -1) & (pre_neg == -1)] = np.nan
    bid_price = np.repeat(data["next.bid"].values.reshape(-1, 1), K, axis=1)
    notional_position_neg[(notional_position_neg == -1)] = (-1 / bid_price)[
        (notional_position_neg == -1)
    ]
    notional_position_neg = pd.DataFrame(
        data=notional_position_neg, index=ori_index, columns=range(K)
    ).fillna(method="ffill")
    notional_position_neg.ffill(inplace=True)
    notional_position_neg = notional_position_neg.values
    position = position_pos + position_neg  # 1, -1, 0，多/ 空/ 空仓

    notional_position = notional_position_pos + notional_position_neg  # 1块钱仓位
    position = pd.DataFrame(data=position, index=ori_index, columns=range(K))
    notional_position = pd.DataFrame(
        data=notional_position, index=ori_index, columns=range(K)
    )
    # position[n_bar-1] = 0
    position.iloc[0] = 0
    position.iloc[-2:] = 0
    notional_position.iloc[0] = 0
    notional_position.iloc[-2:] = 0
    change_pos = position - position.shift(1, axis=0)
    notional_change_pos = notional_position - notional_position.shift(1, axis=0)
    change_pos.iloc[0] = 0
    notional_change_pos.iloc[0] = 0

    change_pos = change_pos.values
    notional_change_pos = notional_change_pos.values

    change_base = np.zeros((N, K))
    change_buy = change_pos > 0
    change_sell = change_pos < 0
    if tranct_ratio:
        change_base[change_buy] = ask_price[change_buy] * (1 + tranct)
        change_base[change_sell] = bid_price[change_sell] * (1 - tranct)
    else:
        change_base[change_buy] = ask_price[change_buy] + tranct
        change_base[change_sell] = bid_price[change_sell] - tranct
    final_pnl = -np.sum(change_base * change_pos, axis=0)  # 等仓位回测，永远买一个单位
    ret = -np.sum(change_base * notional_change_pos, axis=0)  # 等金额回测，永远买一块钱单位
    num = np.sum((position != 0) & (change_pos != 0), axis=0)
    
    position_pnl = np.where(position.values != 0, position.values * future_ret, 0)
    wins = np.nansum(position_pnl > 0, axis=0)
    losses = np.nansum(position_pnl < 0, axis=0)
    total_profit = np.nansum(np.where(position_pnl > 0, position_pnl, 0), axis=0)
    total_loss = np.nansum(np.where(position_pnl < 0, position_pnl, 0), axis=0)

    avg_profit = total_profit / wins
    avg_loss = total_loss / losses
    
    win_rate = wins / (wins + losses)
    profit_rate = avg_profit / -avg_loss
        
    results = []

    ret1 = fcum(data["ret"], 256).values
    ret2 = fcum(data["ret"], 512).values
    ret3 = fcum(data["ret"], 1024).values
    ret4 = fcum(data["ret"], 2048).values

    s = pred[:, 0]
    spearman_corr1, _ = stats.spearmanr(s, ret1)
    spearman_corr2, _ = stats.spearmanr(s, ret2)
    spearman_corr3, _ = stats.spearmanr(s, ret3)
    spearman_corr4, _ = stats.spearmanr(s, ret4)

    for i in range(K):
        open_, close_ = thre[i]
        long_hold_len = _count_continuous_nonzero(position_pos[:, i])
        short_hold_len = _count_continuous_nonzero(position_neg[:, i])
        hold_len = (long_hold_len + short_hold_len) / 2
        hold_len = 0 if np.isnan(hold_len) else hold_len
        if num[i] == 0:
            res = {
                "date": date_,
                "num": 0,
                "avg_pnl": 0,
                "final_pnl": 0,
                "avg_ret": 0,
                "ret": 0,
                "ic.256": float(spearman_corr1),
                "ic.512": float(spearman_corr2),
                "ic.1024": float(spearman_corr3),
                "ic.2048": float(spearman_corr4),
                "open": open_,
                "close": close_,
                "reverse": reverse,
                "tranct": tranct,
                "max_spread": max_spread,
                "atr_filter": atr_filter,
                "product": product,
                "signal_name": signal_name,
                "win_rate": 0.5,
                "profit_factor": 1,
                "hold_len": hold_len
            }
        else:
            res = {
                "date": date_,
                "num": num[i],
                "avg_pnl": final_pnl[i] / num[i],
                "final_pnl": final_pnl[i],
                "avg_ret": ret[i] / num[i],
                "ret": ret[i],
                "win_rate": win_rate[i],
                "profit_factor": profit_rate[i],
                "ic.256": float(spearman_corr1),
                "ic.512": float(spearman_corr2),
                "ic.1024": float(spearman_corr3),
                "ic.2048": float(spearman_corr4),
                "open": open_,
                "close": close_,
                "reverse": reverse,
                "tranct": tranct,
                "max_spread": max_spread,
                "atr_filter": atr_filter,
                "product": product,
                "signal_name": signal_name,
                "hold_len": hold_len
            }

        results.append(res)
    return results


def backtest_loop(
    pred: pd.Series,
    data: pd.DataFrame,
    thre: Tuple,
    atr: Optional[pd.Series] = None,
    atr_filter: Optional[float] = None,
    max_spread=float("inf"),
    tranct_ratio=False,
    tranct=0.0,
) -> DailyResult:
    raise NotImplementedError()
    assert len(data) > 0, "empty data"
    assert len(pred) == len(data), "length mismatch"
    assert len(thre) > 0

    cur_spread = (data["ask"] - data["bid"]).values
    date_ = pd.to_datetime(data["time"].iloc[0]).date()
    position_pos = pd.Series(data=np.nan, index=data.index)
    position_neg = pd.Series(data=np.nan, index=data.index)
    position_pos.iloc[0] = 0
    position_neg.iloc[0] = 0
    notional_position_pos = pd.Series(data=0, index=data.index)
    notional_position_neg = pd.Series(data=0, index=data.index)
    for i in range(len(data)):
        if (
            (data["next.ask"].iloc[i] > 0)
            and (data["next.bid"].iloc[i] > 0)
            and (cur_spread[i] < max_spread)
        ):
            if atr is None or (
                atr is not None and atr_filter is not None and atr.iloc[i] > atr_filter
            ):
                if pred[i] > thre[0]:
                    position_pos.iloc[i] = 1

                if pred[i] < -thre[0]:
                    position_neg.iloc[i] = -1

        if (
            (pred[i] < thre[1])
            and (data["next.bid"].iloc[i] > 0)
            and (cur_spread[i] < max_spread)
        ):
            position_pos.iloc[i] = 0
        if (
            (pred[i] > -thre[1])
            and (data["next.ask"].iloc[i] > 0)
            and (cur_spread[i] < max_spread)
        ):
            position_neg.iloc[i] = 0
    position_pos = position_pos.fillna(method="ffill")
    position_neg = position_neg.fillna(method="ffill")
    for i in range(len(data)):
        if position_pos.iloc[i] == 1:
            notional_position_pos.iloc[i] = 1 / data["next.ask"].iloc[i]
            if i >= 1 and position_pos.iloc[i - 1] == 1:
                notional_position_pos.iloc[i] = np.nan
        if position_neg.iloc[i] == -1:
            notional_position_neg.iloc[i] = -1 / data["next.bid"].iloc[i]
            if i >= 1 and position_neg.iloc[i - 1] == -1:
                notional_position_neg.iloc[i] = np.nan
    notional_position_pos = notional_position_pos.fillna(method="ffill")
    notional_position_neg = notional_position_neg.fillna(method="ffill")

    position = position_pos + position_neg  # 1, -1, 0，多/ 空/ 空仓
    notional_position = notional_position_pos + notional_position_neg  # 1块钱仓位
    position.iloc[0] = 0
    position.iloc[-2:] = 0
    notional_position.iloc[0] = 0
    notional_position.iloc[-2:] = 0

    change_pos = position - position.shift(1)
    notional_change_pos = notional_position - notional_position.shift(1)
    change_pos.iloc[0] = 0
    notional_change_pos.iloc[0] = 0
    change_base = pd.Series(data=0, index=data.index)
    change_buy = change_pos > 0
    change_sell = change_pos < 0
    if tranct_ratio:
        change_base[change_buy] = data["next.ask"][change_buy] * (1 + tranct)
        change_base[change_sell] = data["next.bid"][change_sell] * (1 - tranct)
    else:
        change_base[change_buy] = data["next.ask"][change_buy] + tranct
        change_base[change_sell] = data["next.bid"][change_sell] - tranct
    final_pnl = -sum(change_base * change_pos)  # 等仓位回测，永远买一个单位
    ret = -sum(change_base * notional_change_pos)  # 等金额回测，永远买一块钱单位
    num = sum((position != 0) & (change_pos != 0))
    if num == 0:
        # 交易次数为0，说明该参数不可用
        res = {
            "date": date_,
            "num": 0,
            "avg_pnl": 0,
            "final_pnl": 0,
            "avg_ret": 0,
            "ret": 0,
        }
        return res
    else:
        avg_pnl = np.divide(final_pnl, num)
        avg_ret = np.divide(ret, num)
        res = {
            "date": date_,
            "num": num,
            "avg_pnl": avg_pnl,
            "final_pnl": final_pnl,
            "avg_ret": avg_ret,
            "ret": ret,
        }
        return res


def backtest(
    pred: pd.Series,
    data: pd.DataFrame,
    thre: Tuple,
    atr: Optional[pd.Series] = None,
    atr_filter: Optional[float] = None,
    max_spread=float("inf"),
    tranct_ratio=False,
    tranct=0.0,
) -> DailyResult:

    raise NotImplementedError()
    assert len(data) > 0, "empty data"
    assert len(pred) == len(data), "length mismatch"
    assert len(thre) > 0

    cur_spread = data["ask"] - data["bid"]
    buy = pred > thre[0]
    sell = pred < -thre[0]
    date_ = pd.to_datetime(data["time"].iloc[0]).date()
    signal = pd.Series(data=0, index=data.index)
    position = signal.copy()
    signal[buy] = 1
    signal[sell] = -1
    if atr is not None and atr_filter is not None:
        signal[atr < atr_filter] = 0
    scratch = -thre[1]
    position_pos = pd.Series(data=np.nan, index=data.index)
    position_pos.iloc[0] = 0
    position_pos[
        (signal == 1)
        & (data["next.ask"] > 0)
        & (data["next.bid"] > 0)
        & (cur_spread < max_spread)
    ] = 1
    position_pos[
        (pred < -scratch) & (data["next.bid"] > 0) & (cur_spread < max_spread)
    ] = 0
    position_pos.ffill(inplace=True)
    pre_pos = position_pos.shift(1)
    notional_position_pos = pd.Series(data=0, index=data.index)
    notional_position_pos[position_pos == 1] = 1
    notional_position_pos[(position_pos == 1) & (pre_pos == 1)] = np.nan
    notional_position_pos[(notional_position_pos == 1)] = (
        1 / data["next.ask"][(notional_position_pos == 1)]
    )
    notional_position_pos.ffill(inplace=True)
    position_neg = pd.Series(data=np.nan, index=data.index)
    position_neg.iloc[0] = 0
    position_neg[
        (signal == -1)
        & (data["next.ask"] > 0)
        & (data["next.bid"] > 0)
        & (cur_spread < max_spread)
    ] = -1
    position_neg[
        (pred > scratch) & (data["next.ask"] > 0) & (cur_spread < max_spread)
    ] = 0
    position_neg.ffill(inplace=True)
    pre_neg = position_neg.shift(1)
    notional_position_neg = pd.Series(data=0, index=data.index)
    notional_position_neg[position_neg == -1] = -1
    notional_position_neg[(position_neg == -1) & (pre_neg == -1)] = np.nan
    notional_position_neg[(notional_position_neg == -1)] = (
        -1 / data["next.bid"][(notional_position_neg == -1)]
    )
    notional_position_neg.ffill(inplace=True)
    position = position_pos + position_neg  # 1, -1, 0，多/ 空/ 空仓
    notional_position = notional_position_pos + notional_position_neg  # 1块钱仓位
    # position[n_bar-1] = 0
    position.iloc[0] = 0
    position.iloc[-2:] = 0
    notional_position.iloc[0] = 0
    notional_position.iloc[-2:] = 0
    change_pos = position - position.shift(1)
    notional_change_pos = notional_position - notional_position.shift(1)
    change_pos.iloc[0] = 0
    notional_change_pos.iloc[0] = 0
    change_base = pd.Series(data=0, index=data.index)
    change_buy = change_pos > 0
    change_sell = change_pos < 0
    if tranct_ratio:
        change_base[change_buy] = data["next.ask"][change_buy] * (1 + tranct)
        change_base[change_sell] = data["next.bid"][change_sell] * (1 - tranct)
    else:
        change_base[change_buy] = data["next.ask"][change_buy] + tranct
        change_base[change_sell] = data["next.bid"][change_sell] - tranct
    final_pnl = -sum(change_base * change_pos)  # 等仓位回测，永远买一个单位
    ret = -sum(change_base * notional_change_pos)  # 等金额回测，永远买一块钱单位
    num = sum((position != 0) & (change_pos != 0))
    if num == 0:
        # 交易次数为0，说明该参数不可用
        res = {
            "date": date_,
            "num": 0,
            "avg_pnl": 0,
            "final_pnl": 0,
            "avg_ret": 0,
            "ret": 0,
        }
        return res
    else:
        avg_pnl = np.divide(final_pnl, num)
        avg_ret = np.divide(ret, num)
        res = {
            "date": date_,
            "num": num,
            "avg_pnl": avg_pnl,
            "final_pnl": final_pnl,
            "avg_ret": avg_ret,
            "ret": ret,
        }
        return res


@print_input_on_error
def get_signal_pnl_for_threhold_strategy(
    file_or_date,
    product,
    signal_name,
    thre_mat,
    reverse=1,
    tranct=1.1e-4,
    max_spread=0.61,
    tranct_ratio=True,
    atr_filter=0,
) -> List[DailyResult]:
    """_summary_

    Args:
        file (_type_): 日期
        product (_type_): _description_
        signal_name (_type_): _description_
        thre_mat (_type_): _description_
        reverse (int, optional): _description_. Defaults to 1.
        tranct (_type_, optional): _description_. Defaults to 1.1e-4.
        max_spread (float, optional): _description_. Defaults to 0.61.
        tranct_ratio (bool, optional): _description_.Always True
        atr_filter (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    ## load data
    data = get_data(
        product,
        file_or_date,
        columns=[
            "time",
            "good",
            "bid",
            "ask",
            "next.bid",
            "next.ask",
            "atr.4096",
            "ret",
        ],
    )
    assert data is not None, "get_data return None"
    S = get_signal(product, signal_name, file_or_date)
    assert S is not None, "get_signal return None"
    pred = S
    pred = np.asarray(pred[data["good"]])
    atr = data["atr.4096"][data["good"]].reset_index(drop=True)
    data = data[data["good"]].reset_index(drop=True)
    return backtest_par(
        pred,
        data,
        product=product,
        signal_name=signal_name,
        thre=[(thre["open"], thre["close"]) for _, thre in thre_mat.iterrows()],
        reverse=reverse,
        atr=atr,
        atr_filter=atr_filter,
        tranct_ratio=tranct_ratio,
        max_spread=max_spread,
        tranct=tranct,
    )


class HFTSummary(TypedDict):
    final_result: pd.DataFrame
    daily_num: pd.DataFrame
    daily_pnl: pd.DataFrame
    daily_ret: pd.DataFrame


@print_input_on_error
def get_hft_summary(daily_result: List[List[DailyResult]]) -> HFTSummary:
    # result is a list of output from get_signal_pnl
    # 总结不同超参结果
    assert len(daily_result) > 0, "empty results"
    values = []
    for res_per_strategy in daily_result:
        for res_per_thre in res_per_strategy:
            values.append(res_per_thre)
    daily_results = pd.DataFrame(values)
    daily_results = daily_results.sort_values(by=["open", "close", "date"])
    daily_results["open_close"] = (
        daily_results["open"].astype(str) + "_" + daily_results["close"].astype(str)
    )
    daily_num_per_thre = daily_results.pivot_table(
        values="num", columns="open_close", index="date"
    )
    dates = list(sorted(daily_num_per_thre.index))
    daily_pnl_per_thre = daily_results.pivot_table(
        values="final_pnl", columns="open_close", index="date"
    )
    daily_ret_per_thre = daily_results.pivot_table(
        values="ret", columns="open_close", index="date"
    )
    daily_pf_per_thre = daily_results.pivot_table(
        values="profit_factor", columns="open_close", index="date"
    )
    daily_win_rate_thre = daily_results.pivot_table(
        values="win_rate", columns="open_close", index="date"
    )
    daily_sp1_per_thre = daily_results.pivot_table(
        values="ic.256", columns="open_close", index="date"
    )
    
    daily_sp2_per_thre = daily_results.pivot_table(
        values="ic.512", columns="open_close", index="date"
    )
    
    daily_sp3_per_thre = daily_results.pivot_table(
        values="ic.1024", columns="open_close", index="date"
    )
    
    daily_sp4_per_thre = daily_results.pivot_table(
        values="ic.2048", columns="open_close", index="date"
    )
    open_thre = daily_results.pivot_table(
        values="open", columns="open_close", index="date"
    ).iloc[0]
    close_thre = daily_results.pivot_table(
        values="close", columns="open_close", index="date"
    ).iloc[0]
    daily_hold_len_per_thre = daily_results.pivot_table(
        values="hold_len", columns="open_close", index="date"
    )
    
    sp1 = daily_sp1_per_thre.mean(axis=0)
    sp2 = daily_sp2_per_thre.mean(axis=0)
    sp3 = daily_sp3_per_thre.mean(axis=0)
    sp4 = daily_sp4_per_thre.mean(axis=0)
    total_num_per_thre = daily_num_per_thre.sum()
    total_pnl_per_thre = daily_pnl_per_thre.sum()
    total_ret_per_thre = daily_ret_per_thre.sum()
    hold_len_per_thre = daily_hold_len_per_thre.mean()
    
    avg_profit_factor = daily_pf_per_thre.mean()
    
    avg_win_rate = daily_win_rate_thre.mean()
    avg_pnl_per_thre = zero_divide(total_pnl_per_thre, total_num_per_thre)
    avg_ret_per_thre = zero_divide(total_ret_per_thre, total_num_per_thre)
    avg_num_per_thre = total_num_per_thre / daily_num_per_thre.shape[0]
    total_sharp = sharpe(daily_pnl_per_thre)
    total_drawdown = drawdown(daily_pnl_per_thre)
    total_max_drawdown = max_drawdown(daily_pnl_per_thre)
    sharpe_ret = sharpe(daily_ret_per_thre)
    drawdown_ret = drawdown(daily_ret_per_thre)
    max_drawdown_ret = max_drawdown(daily_ret_per_thre)
    assert open_thre.shape[0] == close_thre.shape[0], "{} != {}".format(open_thre.shape[0], close_thre.shape[0])
    assert open_thre.shape[0] == total_num_per_thre.shape[0], "{} != {}".format(open_thre.shape[0], total_num_per_thre.shape[0])
    assert open_thre.shape[0] == total_pnl_per_thre.shape[0], "{} != {}".format(open_thre.shape[0], total_pnl_per_thre.shape[0])
    assert open_thre.shape[0] == total_ret_per_thre.shape[0], "{} != {}".format(open_thre.shape[0], total_ret_per_thre.shape[0])
    assert open_thre.shape[0] == avg_num_per_thre.shape[0], "{} != {}".format(open_thre.shape[0], avg_num_per_thre.shape[0])
    assert open_thre.shape[0] == total_sharp.shape[0], "{} != {}".format(open_thre.shape[0], total_sharp.shape[0])
    assert open_thre.shape[0] == total_drawdown.shape[0], "{} != {}".format(open_thre.shape[0], total_drawdown.shape[0])
    assert open_thre.shape[0] == total_max_drawdown.shape[0], "{} != {}".format(open_thre.shape[0], total_max_drawdown.shape[0])
    assert open_thre.shape[0] == avg_ret_per_thre.shape[0], "{} != {}".format(open_thre.shape[0], avg_ret_per_thre.shape[0])
    assert open_thre.shape[0] == total_ret_per_thre.shape[0], "{} != {}".format(open_thre.shape[0], total_ret_per_thre.shape[0])
    assert open_thre.shape[0] == hold_len_per_thre.shape[0], "{} != {}".format(open_thre.shape[0], hold_len_per_thre.shape[0])
    assert open_thre.shape[0] == sharpe_ret.shape[0], "{} != {}".format(open_thre.shape[0], sharpe_ret.shape[0])
    assert open_thre.shape[0] == drawdown_ret.shape[0], "{} != {}".format(open_thre.shape[0], drawdown_ret.shape[0])
    assert open_thre.shape[0] == max_drawdown_ret.shape[0], "{} != {}".format(open_thre.shape[0], max_drawdown_ret.shape[0])
    assert open_thre.shape[0] == avg_profit_factor.shape[0], "{} != {}".format(open_thre.shape[0], avg_profit_factor.shape[0])
    assert open_thre.shape[0] == avg_pnl_per_thre.shape[0], "{} != {}".format(open_thre.shape[0], avg_pnl_per_thre.shape[0])
    assert open_thre.shape[0] == sp1.shape[0], "{} != {}".format(open_thre.shape[0], sp1.shape[0])
    assert open_thre.shape[0] == sp2.shape[0], "{} != {}".format(open_thre.shape[0], sp2.shape[0])
    assert open_thre.shape[0] == sp3.shape[0], "{} != {}".format(open_thre.shape[0], sp3.shape[0])
    assert open_thre.shape[0] == sp4.shape[0], "{} != {}".format(open_thre.shape[0], sp4.shape[0])
    
    
    final_result = pd.DataFrame(
        data=OrderedDict(
            [
                ("open", open_thre),
                ("close", close_thre),
                ("num", total_num_per_thre),  # 总交易次数
                ("avg.num", avg_num_per_thre),  # 总交易次数/天数
                ("avg.pnl", avg_pnl_per_thre),
                ("total.pnl", total_pnl_per_thre),
                ("sharpe", total_sharp),
                ("drawdown", total_drawdown),
                ("max.drawdown", total_max_drawdown),
                ("avg.ret", avg_ret_per_thre),
                ("total.ret", total_ret_per_thre),
                ("avg.hold", hold_len_per_thre),
                ("sharpe.ret", sharpe_ret),
                ("drawdown.ret", drawdown_ret),
                ("max.drawdown.ret", max_drawdown_ret),
                ("mar", total_pnl_per_thre / total_max_drawdown),
                ("mar.ret", total_ret_per_thre / max_drawdown_ret),
                ("win_rate", avg_win_rate),
                ("profit_factor", avg_profit_factor),
                ("ic.256", sp1),
                ("ic.512", sp2),
                ("ic.1024", sp3),
                ("ic.2048", sp4),
            ]
        ),
    )
    final_result['date_from'] = dates[0]
    final_result['date_to'] = dates[-1]
    final_result['reverse'] = daily_result[0][0]["reverse"]
    final_result['tranct'] = daily_result[0][0]["tranct"]
    final_result['max_spread'] = daily_result[0][0]["max_spread"]
    final_result['atr_filter'] = daily_result[0][0]["atr_filter"]
    final_result['product'] = daily_result[0][0]["product"]
    final_result['signal_name'] = daily_result[0][0]["signal_name"]
    return OrderedDict(
        [
            ("final_result", final_result),
            ("daily.num", daily_num_per_thre),
            ("daily.pnl", daily_pnl_per_thre),
            ("daily.ret", daily_ret_per_thre),
            ("daily.result", daily_result),
        ]
    )


def get_signal_stat(
    signal_name,
    thre_mat,
    product,
    all_dates,
    reverse=1,
    tranct=1.1e-4,
    max_spread=0.61,
    tranct_ratio=True,
    atr_filter=0,
    test_dates=None,
) -> Union[HFTSummary, Tuple[HFTSummary, HFTSummary]]:
    """_summary_

    Args:
        signal_name (_type_): signal_name with period
        thre_mat (_type_): threshold for open/close
        product (_type_): _description_
        all_dates (_type_): _description_
        split_str (str, optional): _description_. date str to split train/test. Defaults to "2018".
        reverse (int, optional): _description_. Defaults to 1.
        tranct (_type_, optional): _description_. Defaults to 1.1e-4.
        max_spread (float, optional): _description_. Defaults to 0.61.
        tranct_ratio (bool, optional): _description_. Defaults to True.
        min_pnl (int, optional): _description_. Defaults to 2.
        min_num (int, optional): _description_. Defaults to 20.
        atr_filter (int, optional): _description_. Defaults to 0.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    assert len(all_dates) > 0, "empty all_dates"
    assert len(thre_mat) > 0, "empty thre_mat"
    train_dates = all_dates
    if test_dates is not None:
        assert len(test_dates) > 0
        all_dates = all_dates + test_dates
    missing_dates = date_is_continuous(all_dates)
    if len(missing_dates):
        print("warning: missing dates for {} {}: {}".format(product, signal_name, missing_dates))
    with dask.config.set(scheduler="threads", num_workers=CORE_NUM):
        f_par = functools.partial(
            get_signal_pnl_for_threhold_strategy,
            product=product,
            signal_name=signal_name,
            thre_mat=thre_mat,
            reverse=reverse,
            tranct=tranct,
            max_spread=max_spread,
            tranct_ratio=tranct_ratio,
            atr_filter=atr_filter,
        )
        daily_result: List[List[DailyResult]] = compute(
            [delayed(f_par)(file) for file in all_dates]
        )[0]
    train_result = []
    test_result = []
    for items in daily_result:
        daily_res = items[0]
        date_str = daily_res["date"].strftime("%Y-%m-%d")
        if date_str in train_dates:
            train_result.append(items)
        if test_dates is not None and date_str in test_dates:
            test_result.append(items)

    train_stat = get_hft_summary(train_result)
    if test_dates:
        test_stat = get_hft_summary(test_result)
        return train_stat, test_stat
    else:
        return train_stat


SignalTrainTestStat = TypedDict(
    "SignalTrainTestStat", {"train.stat": HFTSummary, "test.stat": HFTSummary}
)


def evaluate_signal(
    signal,
    all_dates,
    product,
    period=4096,
    split_str="2018",
    tranct=1.1e-4,
    max_spread=0.61,
    tranct_ratio=True,
    atr_filter=0,
    save_path="signal_result_atr",
    reverse=0,
    overwrite=False
):
    """_summary_

    Args:
        signal (_type_): base signal name, without period
        all_dates (_type_): dates with format "xxxx-xx-xx.pkl" or "xxxx-xx-xx"
        product (_type_): name
        period (int, optional): _description_. Defaults to 4096.
        split_str (str, optional): _description_. Defaults to "2018".
        tranct (_type_, optional): _description_. Defaults to 1.1e-4.
        max_spread (float, optional): _description_. Defaults to 0.61.
        tranct_ratio (bool, optional): _description_. Defaults to True.
        atr_filter (int, optional): _description_. Defaults to 0.
        save_path (str, optional): _description_. Defaults to "signal result".
        reverse (int, optional): _description_. Defaults to 0.
    """
    skip_evaluate = True
    signal_name = signal + "." + str(period)  ## signal name, with period
    if reverse >= 0:
        path = SIGNAL_RESULTS_PATH / save_path / signal_name / product / "trend.pkl"
        if not path.exists():
            skip_evaluate = False
    if reverse <= 0:  ## reversal signal
        path = SIGNAL_RESULTS_PATH / save_path / signal_name / product / "reverse.pkl"
        if not path.exists():
            skip_evaluate = False
    if skip_evaluate and not overwrite:
        return
    
    all_signal = auto_get_alldates_signal(signal_name, product)
    open_list = np.quantile(
        abs(all_signal),
        np.append(np.linspace(0.8, 0.99, 5), np.linspace(0.991, 0.999, 5)),
    )  ## open threshold
    thre_list = []
    for cartesian in itertools.product(
        open_list, np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    ):  ## close threshold
        thre_list.append((cartesian[0], -cartesian[0] * cartesian[1]))
    thre_list = np.array(thre_list)
    thre_mat = pd.DataFrame(
        data=OrderedDict([("open", thre_list[:, 0]), ("close", thre_list[:, 1])])
    )  ## threshold matrix

    all_dates = np.array(all_dates)
    train_samples = list(all_dates[all_dates < split_str])  ## train samples
    test_samples = list(all_dates[all_dates >= split_str])  ## test samples
    meta = {
        "product": product,
        "trading_fee": tranct,
        "max_spread": max_spread,
        "period": period,
        "signal_name": signal_name,
        "atr_filter": atr_filter,
    }
    if reverse >= 0:
        path = SIGNAL_RESULTS_PATH / save_path / f"{product}.{signal_name}.trend.pkl"
        if overwrite or not path.exists() :
            print("evaludating signal trend {} {}".format(product, signal_name), end=" ")
            train_stat, test_stat = get_signal_stat(
                signal_name,
                thre_mat,
                product,
                train_samples,
                reverse=1,
                tranct=tranct,
                max_spread=max_spread,
                tranct_ratio=tranct_ratio,
                atr_filter=atr_filter,
                test_dates=test_samples,
            )
            for key, value in meta.items():
                train_stat["final_result"][key] = value

            train_stat["final_result"]["date_range"] = "{} to {}".format(
                train_samples[0], train_samples[-1]
            )
            train_stat["final_result"]["bt_strategy"] = save_path

            for key, value in meta.items():
                test_stat["final_result"][key] = value
            test_stat["final_result"]["date_range"] = "{} to {}".format(
                test_samples[0], test_samples[-1]
            )
            test_stat["final_result"]["bt_strategy"] = save_path

            trend_signal_stat: SignalTrainTestStat = OrderedDict(
                [("train.stat", train_stat), ("test.stat", test_stat)]
            )
            save(trend_signal_stat, path)
            print("done")
            update_signal_results_db(
                save_path,
                product=product,
                signal_name=signal_name,
                results=[train_stat, ],
                is_train=True,
            )
            update_signal_results_db(
                save_path,
                product=product,
                signal_name=signal_name,
                results=[test_stat],
                is_train=False,
            )

    if reverse <= 0:  ## reversal signal
        path = SIGNAL_RESULTS_PATH / save_path / f"{product}.{signal_name}.reverse.pkl"
        if overwrite or not path.exists():
            print("evaludating signal reverse {} {}".format(product, signal_name), end=" ")
            train_stat, test_stat = get_signal_stat(
                signal_name,
                thre_mat,
                product,
                train_samples,
                reverse=-1,
                tranct=tranct,
                max_spread=max_spread,
                tranct_ratio=tranct_ratio,
                atr_filter=atr_filter,
                test_dates=test_samples,
            )
            for key, value in meta.items():
                train_stat["final_result"][key] = value

            train_stat["final_result"]["date_range"] = "{} to {}".format(
                train_samples[0], train_samples[-1]
            )
            train_stat["final_result"]["bt_strategy"] = save_path

            for key, value in meta.items():
                test_stat["final_result"][key] = value
            test_stat["final_result"]["date_range"] = "{} to {}".format(
                test_samples[0], test_samples[-1]
            )
            test_stat["final_result"]["bt_strategy"] = save_path

            reverse_signal_stat: SignalTrainTestStat = OrderedDict(
                [("train.stat", train_stat), ("test.stat", test_stat)]
            )
            save(reverse_signal_stat, path)
            print("done")
            update_signal_results_db(
                save_path,
                product=product,
                signal_name=signal_name,
                results=[train_stat, ],
                is_train=True,
            )
            update_signal_results_db(
                save_path,
                product=product,
                signal_name=signal_name,
                results=[test_stat],
                is_train=False,
            )


def _parse_data_info(input_string):
    # Define the regular expression pattern to match the product_name, signal_name, and direction
    pattern = r"(.+?)\.(.+?)\.(.+?)\.pkl"

    # Use re.search() to find the first occurrence of the pattern in the input string
    match = re.search(pattern, input_string)

    if match:
        # Extract product_name, signal_name, and direction from the matched groups
        product_name = match.group(1)
        signal_name = match.group(2)
        direction = match.group(3)
        return product_name, signal_name, direction
    else:
        return None, None, None


def update_signal_results_db(
    bt_stra: str,
    product=None,
    signal_name=None,
    results: List[HFTSummary] = None,
    is_train=False,
):
    db_path = SIGNAL_RESULTS_PATH / "signal.csv"
    if db_path.exists():
        table = pd.read_csv(db_path, index_col=0)
    else:
        table = None
    if results is None:
        print("update all signal results")
        for bt_strategy_path in SIGNAL_RESULTS_PATH.iterdir():
            if bt_strategy_path.is_dir():
                bt_stra = bt_strategy_path.name
                for file_path in bt_strategy_path.iterdir():
                    if file_path.is_file():
                        signal_stat: SignalTrainTestStat = load(
                            file_path
                        )  ## statistics of signal over a product
                        if "reverse" in file_path.name:
                            reverse = -1
                        elif "trend" in file_path.name:
                            reverse = 1
                        else:
                            raise ValueError
                        final_result_train, final_result_test = (
                            signal_stat["train.stat"]["final_result"],
                            signal_stat["test.stat"]["final_result"],
                        )

                        final_result_train["is_train"] = True
                        final_result_train["reverse"] = reverse
                        final_result_train["bt_stra"] = bt_stra

                        final_result_test["is_train"] = False
                        final_result_test["bt_stra"] = bt_stra
                        final_result_test["reverse"] = reverse

                        if table is None:
                            table = pd.concat(
                                [final_result_test, final_result_train], axis=0
                            )
                        else:
                            table = pd.concat(
                                [table, final_result_test, final_result_train], axis=0
                            )
    else:
        assert signal_name is not None, "signal_name is None"
        assert product is not None, "product is None"
        assert is_train is not None, "is_train is None"
        for result in results:
            final_result = result["final_result"]
            final_result["is_train"] = is_train
            final_result["bt_stra"] = bt_stra
            if table is None:
                table = final_result
            else:
                table = pd.concat([table, final_result], axis=0)

    if table is not None:
        table.to_csv(db_path, index=True)


## get the signal performance
## including trend and reverse signals
def get_signal_performance_result(
    all_period_signal: List[str],
    save_path: str,
    product_list: List[str],
    min_avg_ret=0.002,
    min_avg_trade_num=1,
    min_stra_threshold=2,
    directions=["trend", "reverse"],
):
    """
    针对每个product
    1)首先filter good strategy. good strategy: avg_pnl >= min_avg_pnl and num >= avg.min_trade_num
    2)然后filter good product. good product: number of good strategy > min_stra_threshold
    3) 每个good strategy的daily pnl取平均, 用于计算good product的trainSharpe和testSharpe
    4) 每个good product的trainSharpe取平均, 用于计算signal的train/test Sharpe

    all_period_signal: List of signal names with period
    signal_result_dir: directory prefix
    min_avg_pnl: minimal average pnl
    return:
        trend_signal_result, ("signal", "num", "trainSharpe", "testSharpe"), trainSharpe是在product上面mean
        reverse_signal_result
    """
    trend_signal_result = pd.DataFrame(
        data=OrderedDict(
            [
                ("signal", all_period_signal),
                ("reverse", 1),
                ("num_good_product", 0),
                ("trainSharpe", 0),
                ("testSharpe", 0),
                ("trainAvgDailyTradeNum", 0),
                ("testAvgDailyTradeNum", 0),
            ]
        )
    )
    reverse_signal_result = pd.DataFrame(
        data=OrderedDict(
            [
                ("signal", all_period_signal),
                ("reverse", -1),
                ("num_good_product", 0),
                ("trainSharpe", 0),
                ("testSharpe", 0),
                ("trainAvgDailyTradeNum", 0),
                ("testAvgDailyTradeNum", 0),
                ("avgTrainRet", 0),
                ("avgTestRet", 0),
            ]
        )
    )
    n_signal = len(all_period_signal)  ## number of all signals
    train_sharpe = np.zeros(len(product_list))
    test_sharpe = np.zeros(len(product_list))
    train_ret = np.zeros(len(product_list))
    test_ret = np.zeros(len(product_list))
    for k in range(n_signal):
        signal_name = all_period_signal[k]
        for direction in directions:
            num_good_products = 0
            good_train_num = 0
            good_test_num = 0
            num_good_stratey = 0
            for product in product_list:
                p = (
                    SIGNAL_RESULTS_PATH
                    / save_path
                    / f"{product}.{signal_name}.{direction}.pkl"
                )
                signal_stat: SignalTrainTestStat = load(
                    p
                )  ## statistics of signal over a product
                if signal_stat is None:
                    print(f"{p} not found ")
                    continue
                
                if tuple(signal_stat.keys())[0] != "train.stat":
                    raise ValueError("wrong key")
                train_stat = signal_stat["train.stat"]
                test_stat = signal_stat["test.stat"]
                good_strat = (
                    train_stat["final_result"]["avg.ret"] > min_avg_ret
                ) & (
                    train_stat["final_result"]["avg.num"] > min_avg_trade_num
                )  ## filter criterion
                good_strat = good_strat.fillna(0)
                num_good_stratey += sum(good_strat)
                if sum(good_strat) > min_stra_threshold:
                    train_stat = signal_stat["train.stat"]
                    test_stat = signal_stat["test.stat"]
                    train_pnl = train_stat["daily.ret"].loc[:, good_strat].sum(
                        axis=1
                    ) / sum(
                        good_strat
                    )  ## get the daily return
                    # train_std = np.std(train_pnl)
                    # train_pnl = train_pnl/train_std
                    test_pnl = test_stat["daily.ret"].loc[:, good_strat].sum(
                        axis=1
                    ) / sum(good_strat)
                    train_sharpe[num_good_products] = sharpe(train_pnl)
                    test_sharpe[num_good_products] = sharpe(test_pnl)
                    
                    train_ret[num_good_products] = np.nanmean(train_pnl)
                    test_ret[num_good_products] = np.nanmean(test_pnl)

                    train_num = train_stat["daily.num"].loc[:, good_strat].mean().mean()

                    # train_std = np.std(train_pnl)
                    # train_pnl = train_pnl/train_std
                    test_num = test_stat["daily.num"].loc[:, good_strat].mean().mean()

                    good_test_num += test_num
                    good_train_num += train_num
                    # print(product, "train sharpe ", sharpe(train_pnl), "test sharpe ", sharpe(test_pnl))
                    num_good_products = num_good_products + 1
                if num_good_products > 0:  ## if there are any good products
                    if direction == "trend":
                        trend_signal_result.loc[
                            k,
                            (
                                "signal",
                                "num_good_product",
                                "trainSharpe",
                                "testSharpe",
                                "trainAvgDailyTradeNum",
                                "testAvgDailyTradeNum",
                                "sum_good_strat",
                                'avgTrainRet',
                                "avgTestRet"
                            ),
                        ] = (
                            signal_name,
                            num_good_products,
                            np.mean(train_sharpe[:num_good_products]),
                            np.mean(test_sharpe[:num_good_products]),
                            good_train_num,
                            good_test_num,
                            num_good_stratey,
                            np.mean(train_ret[:num_good_products]),
                            np.mean(test_ret[:num_good_products]),
                        )
                    else:
                        reverse_signal_result.loc[
                            k,
                            (
                                "signal",
                                "num_good_product",
                                "trainSharpe",
                                "testSharpe",
                                "trainAvgDailyTradeNum",
                                "testAvgDailyTradeNum",
                                'avgTrainRet',
                                "avgTestRet"
                            ),
                        ] = (
                            signal_name,
                            num_good_products,
                            np.mean(train_sharpe[:num_good_products]),
                            np.mean(test_sharpe[:num_good_products]),
                            good_train_num,
                            good_test_num,
                            np.mean(train_ret[:num_good_products]),
                            np.mean(test_ret[:num_good_products]),
                        )
    return OrderedDict(
        [
            ("trend.signal.stat", trend_signal_result),
            ("reverse.signal.stat", reverse_signal_result),
        ]
    )


def get_signal_stat_rolling(
    signal_name,
    product,
    all_dates,
    reverse=1,
    tranct=1.1e-4,
    max_spread=0.61,
    tranct_ratio=True,
    atr_filter=0,
    period=4096,
    q=0.95,
):
    ## load data
    all_dates = list(sorted(all_dates))

    def _get_q(file, q):
        D = get_data(product, file)
        S = get_signal(product, signal_name, file)
        S = S * reverse
        S = S[D["good"]]
        open_t = np.quantile(np.abs(S), q)
        return open_t

    with dask.config.set(scheduler="threads", num_workers=CORE_NUM):
        open_t_by_dates = compute([delayed(_get_q)(file, q=q) for file in all_dates])[0]
    results = []
    global_threlist_by_date = {}
    with dask.config.set(scheduler="threads", num_workers=CORE_NUM):
        for idx, file in enumerate(all_dates):
            if idx == 0:
                continue
            open_list = [open_t_by_dates[idx - 1]]
            thre_list = []
            for cartesian in itertools.product(
                open_list, np.array([0.2, 0.4, 0.6, 0.8, 1.0])
            ):
                thre_list.append((cartesian[0], -cartesian[0] * cartesian[1]))
            thre_list = np.array(thre_list)
            global_threlist_by_date[file] = thre_list
            thre_mat = pd.DataFrame(
                data=OrderedDict(
                    [("open", thre_list[:, 0]), ("close", thre_list[:, 1])]
                )
            )  ## threshold matrix
            res = delayed(get_signal_pnl_for_threhold_strategy)(
                file,
                product,
                signal_name,
                thre_mat,
                reverse=reverse,
                tranct=tranct,
                max_spread=max_spread,
                tranct_ratio=tranct_ratio,
                atr_filter=atr_filter,
            )
            results.append(res)
        results = compute(results)[0]
    thre_mat = pd.DataFrame(
        data=OrderedDict([("open", [q] * 5), ("close", [0.2, 0.4, 0.6, 0.8, 1.0])])
    )  ## threshold matrix
    final_results = get_hft_summary(results)
    return final_results, global_threlist_by_date


def quick_roc_test(signal_name, product, N_threshold=13):
    date_list = get_dates_list(product)
    date_list = list(sorted(date_list))
    assert date_is_continuous(date_list)
    from_date, to_date = date_list[0], date_list[-1]
    S = get_good_signal(product, signal_name, date_from=from_date, date_to=to_date)
    ret = get_good_data(product, date_from=from_date, date_to=to_date, columns=["ret"])
    ret = ret["ret"]
    ret = np.asarray(ret).reshape(-1)
    assert len(ret) == len(S), "{} != {}".format(len(ret), len(S))

    # all_signal = auto_get_alldates_signal(signal_name, product)
    period = int(signal_name.split(".")[-1])
    sampled_signal = S[::period]
    thre = np.quantile(
        sampled_signal, np.linspace(0.001, 0.999, N_threshold)
    )  ## open threshold
    K = len(thre)
    future_ret = ret[1:]
    future_ret = np.append(future_ret, [0])
    future_ret_stack = np.concatenate([future_ret[:, np.newaxis]] * K, axis=1)  # (N, K)
    results = {}
    results["threshold"] = thre
    sig = S.reshape(-1, 1)
    for reverse in [-1, 1]:
        if reverse == 1:
            hold = sig > thre.reshape(1, -1)  # (N, K)
        else:
            hold = sig < thre.reshape(1, -1)
        pnl = np.where(hold, future_ret_stack, 0)
        wins = (pnl > 0).sum(axis=0)
        losses = (pnl < 0).sum(axis=0)
        total_profit = np.where(pnl > 0, pnl, 0).sum(axis=0)
        total_loss = np.where(pnl < 0, pnl, 0).sum(axis=0)

        avg_profit = total_profit / wins
        avg_loss = total_loss / losses

        win_rate = wins / (wins + losses)
        profit_rate = avg_profit / -avg_loss
        if reverse == 1:
            results["long_hold_time"] = np.mean(hold, axis=0)
            results["long_pf"] = profit_rate
            results["long_wr"] = win_rate
        else:
            results["short_hold_time"] = np.mean(hold, axis=0)
            results["short_pf"] = profit_rate
            results["short_wr"] = win_rate
    return results
