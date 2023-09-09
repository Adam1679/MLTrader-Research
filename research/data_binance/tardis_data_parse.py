"""_summary_
这个脚本
1）把所有的daily zip文件解压到daily文件夹下
2）把所有的monthly zip文件解压并且合并到daily文件夹下
3）把daily的trades和orderbook合并到parsed tick里面，会做downsample到500ms
4）把parsed tick做相关的预处理
"""
import sys

sys.path.insert(0, ".")
import calendar
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from research.data_binance.down_s3_and_process_by_symbol import combine_orderbook_trades


def is_numeric(num):
    try:
        float(num)  # Try to convert the string to a float
        return True  # If it succeeds, the string is numeric
    except ValueError:  # If a ValueError occurs, it is not numeric
        return False


def get_dates_in_month(year, month):
    _, num_days = calendar.monthrange(year, month)
    start_date = datetime(year, month, 1)
    end_date = start_date + timedelta(days=num_days - 1)

    all_dates = []
    current_date = start_date

    while current_date <= end_date:
        all_dates.append(current_date.date())
        current_date += timedelta(days=1)

    return all_dates


def dump_df(filename, df: pd.DataFrame):
    df.to_pickle(filename + ".unfinished.pkl")
    os.rename(filename + ".unfinished.pkl", filename)


def process(binance_data_root, symbol, date, downsample_ms=1000):
    try:
        new_data_dir = (
            Path(binance_data_root) / "data" / "futures" / "um" / "tick_parsed"
        )
        globs = (new_data_dir / symbol).glob(f"{date}.*")
        try:
            next(globs)
            print(f"skip {symbol} {date} because it is processed")
            return
        except StopIteration:
            pass

        ticks, trades = parse_date_file_get_orderbook_and_trades(
            binance_data_root, symbol, date, downsample_ms=downsample_ms
        )
        if len(ticks) and len(trades):
            fused_orderbook = combine_orderbook_trades(ticks, trades)
            df = pd.DataFrame(fused_orderbook)
            float32_cols = [
                "price",
                "qty",
                "quote_qty",
                "active.buy.qty",
                "active.sell.qty",
                "active.buy.quote_qty",
                "active.sell.quote_qty",
            ]
            for ii in range(20):
                float32_cols.append(f"ask_{ii}_p")
                float32_cols.append(f"ask_{ii}_v")
                float32_cols.append(f"bid_{ii}_p")
                float32_cols.append(f"bid_{ii}_v")
            for col in float32_cols:
                df[col] = df[col].astype("float32")
            expected = 2 * 60 * 60 * 24
            print(f"{df.shape[0]}/{expected} ticks for {symbol} on {date}")
            path = os.path.join(new_data_dir, symbol, f"{date}.pd_pkl")
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            dump_df(path, df)
            print(
                "done merging trades and ticks for {} on {}. {} ticks and {} trades".format(
                    symbol, date, len(ticks), len(trades)
                )
            )
        else:
            print(
                "no data to merge for {} on {}.ticks= {} trades={}".format(
                    symbol, date, len(ticks), len(trades)
                )
            )

    except Exception as e:
        print(f"error in {symbol} {date}, e={e}")
        traceback.print_exc()
        raise e


def parse_date_file_get_orderbook_and_trades(
    binance_data_root, symbol, date, downsample_ms=500
):

    trades_file_path = (
        Path(binance_data_root)
        / "data"
        / "futures"
        / "um"
        / "daily"
        / "trades"
        / symbol.upper()
        / f"{symbol}-trades-{date}.csv"
    )
    trades_zipped_file_path = (
        Path(binance_data_root)
        / "data"
        / "futures"
        / "um"
        / "daily"
        / "trades"
        / symbol.upper()
        / f"{symbol}-trades-{date}.zip"
    )
    orderbook_file_zipped = (
        Path(binance_data_root)
        / "data"
        / "futures"
        / "um"
        / "book_snapshot_25"
        / symbol.upper()
        / f"binance-futures_book_snapshot_25_{date}_{symbol}.csv.gz"
    )
    orderbook_file = (
        Path(binance_data_root)
        / "data"
        / "futures"
        / "um"
        / "book_snapshot_25"
        / symbol.upper()
        / f"binance-futures_book_snapshot_25_{date}_{symbol}.csv"
    )

    if not trades_file_path.exists() and not trades_zipped_file_path.exists():
        print("missing trade file at {}".format(trades_file_path))
        return [], []
    elif not trades_file_path.exists():
        ret = os.system(
            f"unzip -nq {trades_zipped_file_path} -d {trades_zipped_file_path.parent}"
        )
        if ret < 0:
            raise ValueError(f"unzip {trades_zipped_file_path} failed")

    all_order_bookdata = []
    try:
        if not orderbook_file.exists():
            if not orderbook_file_zipped.exists():
                print("missing orderbook file at {}".format(orderbook_file_zipped))
                return [], []
            else:
                ret = os.system(f"gzip -dk {orderbook_file_zipped}")
                if ret < 0:
                    raise ValueError(f"gzip {orderbook_file_zipped} failed")

        # print(
        #     "extracting trades and depth data for {} on {}, downsample by {} ms".format(
        #         symbol, date, downsample_ms
        #     )
        # )
        last_t = None
        interval_in_micro_seconds = downsample_ms * 1000
        with open(orderbook_file, "r") as file:
            first_line = True
            columns = None
            columns_to_idx = {}
            line = file.readline()
            while line:
                if first_line:
                    first_line = False
                    columns = line.strip().split(",")
                    columns_to_idx = {columns[i]: i for i in range(len(columns))}
                    line = file.readline()
                    continue
                meta_data = line.strip().split(",")
                T = int(meta_data[columns_to_idx["timestamp"]])  # micronsecond
                if last_t is not None and T - last_t <= interval_in_micro_seconds:
                    line = file.readline()
                    continue

                bids, asks = [], []
                valid = True

                for level in range(20):
                    bid_p_name = f"bids[{level}].price"
                    bid_v_name = f"bids[{level}].amount"
                    ask_p_name = f"asks[{level}].price"
                    ask_v_name = f"asks[{level}].amount"
                    bid_p = meta_data[columns_to_idx[bid_p_name]]
                    bid_q = meta_data[columns_to_idx[bid_v_name]]
                    ask_p = meta_data[columns_to_idx[ask_p_name]]
                    ask_q = meta_data[columns_to_idx[ask_v_name]]
                    # validation
                    if (
                        not is_numeric(bid_p)
                        or not is_numeric(bid_q)
                        or not is_numeric(ask_p)
                        or not is_numeric(ask_q)
                    ):
                        valid = False
                        break
                    if (
                        float(bid_p) <= 0
                        or float(bid_q) <= 0
                        or float(ask_p) <= 0
                        or float(ask_q) <= 0
                    ):
                        valid = False
                        break
                    bids.append(
                        (
                            bid_p,
                            bid_q,
                        )
                    )
                    asks.append(
                        (
                            ask_p,
                            ask_q,
                        )
                    )

                d = {"data": {"s": symbol, "T": T / 1000, "b": bids, "a": asks}}
                if valid:
                    all_order_bookdata.append(d)
                else:
                    print(f"[{symbol}] [{date}] wrong value {line}")
                line = file.readline()
                last_t = T

        trades = []
        with open(trades_file_path, "r") as f:
            line = f.readline()
            first_line = True
            while line:
                if first_line:
                    line = f.readline()
                    first_line = False
                    continue
                line = line.strip().split(",")
                if len(line) == 6:
                    id, price, qty, quote_qty, time, is_buyer_maker = line
                    meta = {
                        "data": {
                            "s": symbol,
                            "p": float(price),
                            "q": float(qty),
                            "T": int(time),
                            "is_buyer_maker": True
                            if "true" in is_buyer_maker.lower()
                            else False,
                        }
                    }
                    valid = True
                    if (
                        not is_numeric(price)
                        or not is_numeric(qty)
                        or not is_numeric(time)
                    ):
                        valid = False
                    if float(price) <= 0 or float(qty) <= 0 or float(time) <= 0:
                        valid = False
                    if valid:
                        trades.append(meta)
                    else:
                        print(f"[{symbol}] [{date}] wrong value {line}")
                        if float(qty) == 0:
                            print(
                                f"[{symbol}] [{date}] zero trades {line}. Skip the file"
                            )
                            trades = []
                            break

                line = f.readline()
    except Exception as e:
        traceback.print_exc()
    finally:
        if orderbook_file.exists():
            os.remove(str(orderbook_file))
        if trades_file_path.exists():
            os.remove(str(trades_file_path))

    return all_order_bookdata, trades


if __name__ == "__main__":
    import argparse
    import re
    import sys
    from datetime import datetime

    from research.data_binance.download_binance_trades import S_DEFAULTS

    binance_data_root = Path("/Volumes/AdamDrive/binance_data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=S_DEFAULTS)
    parser.add_argument("--start_date", default="")
    parser.add_argument("--end_date", default="")
    args = parser.parse_args()
    symbols = args.symbols
    # get the dates for the depth data

    dates = set()
    for symbol in symbols:
        sub_dates = set()
        for name in os.listdir(
            binance_data_root
            / "data"
            / "futures"
            / "um"
            / "book_snapshot_25"
            / symbol.upper()
        ):
            match = re.search(r"\d{4}-\d{2}-\d{2}", name)
            if match:
                date_str = match.group()
                dates.add(date_str)
                sub_dates.add(datetime.strptime(date_str, "%Y-%m-%d").date())

        sub_dates = sorted(list(sub_dates))
        for i in range(len(sub_dates) - 1):
            diff = sub_dates[i + 1] - sub_dates[i]
            if diff.days > 1:
                print(
                    "[{}] gap between {} and {}".format(
                        symbol, sub_dates[i], sub_dates[i + 1]
                    )
                )

    dates = sorted(list(dates))

    if args.start_date != "":
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start_date = datetime.strptime(dates[0], "%Y-%m-%d").date()

    if args.end_date != "":
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_date = datetime.strptime(dates[-1], "%Y-%m-%d").date()
    dates = [
        date
        for date in dates
        if start_date <= datetime.strptime(date, "%Y-%m-%d").date() <= end_date
    ]

    # print("Stage 2: merge orderbooks and trades data ***************")
    # workers = ProcessPoolExecutor(4)
    # for symbol in symbols:
    #     sub_dates = set()
    #     for name in os.listdir(
    #         binance_data_root
    #         / "data"
    #         / "futures"
    #         / "um"
    #         / "book_snapshot_25"
    #         / symbol.upper()
    #     ):
    #         match = re.search(r"\d{4}-\d{2}-\d{2}", name)
    #         if match:
    #             date_str = match.group()
    #             if (
    #                 start_date
    #                 <= datetime.strptime(date_str, "%Y-%m-%d").date()
    #                 <= end_date
    #             ):
    #                 sub_dates.add(datetime.strptime(date_str, "%Y-%m-%d").date())
    #     for date in sub_dates:
    #         workers.submit(process, binance_data_root, symbol, date)
    # workers.shutdown()

    # preprocess
    sys.path.insert(0, ".")
    from research.orderbook_strategies.utils import helper

    for symbol in symbols:
        print("preprocess {}".format(symbol))
        helper.preprecess_parsed_tick(
            binance_data_root / "data" / "futures" / "um" / "tick_parsed" / symbol
        )
