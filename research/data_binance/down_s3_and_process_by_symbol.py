import gzip
import json
import math
import os
import shutil
import tarfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime

import _pickle as cPickle
import pandas as pd
import pytz
from tqdm import tqdm

bucket_name = "binance-orderbook-data"
prefix = "packet_history_by_symbol"
DATA_DIR = "/Users/bytedance/binance_data/data/futures/um/packet_raw"
DATA_DIR2 = "/Users/bytedance/binance_data/data/futures/um/tick_parsed"
os.makedirs(DATA_DIR2, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
DEBUG = True


def combine_orderbook_trades(orderbooks, trades, level=20):
    fused_orderbook = []
    num_lost = 0

    def check_pack_lost(last, cur):
        if "pu" not in cur["data"]:
            return False
        if last is None:
            return False
        return not cur["data"]["pu"] == last["data"]["u"]

    cur_trade_i = 0
    for i, tick in enumerate(orderbooks):
        symbol = tick["data"]["s"]
        format_tick = {}
        timestamp = datetime.fromtimestamp(int(tick["data"]["T"]) / 1000, pytz.utc)
        date = timestamp.strftime("%Y-%m-%d")
        format_tick["time"] = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        for ii in range(level):
            ask_p, ask_v = tick["data"]["a"][ii]
            bid_p, bid_v = tick["data"]["b"][ii]
            format_tick[f"ask_{ii}_p"] = float(ask_p)
            format_tick[f"ask_{ii}_v"] = float(ask_v)
            format_tick[f"bid_{ii}_p"] = float(bid_p)
            format_tick[f"bid_{ii}_v"] = float(bid_v)

        format_tick["pack_lost"] = check_pack_lost(
            orderbooks[i - 1] if i > 0 else None, tick
        )
        if format_tick["pack_lost"]:
            num_lost += 0

        n_trades = 0
        qty = 0
        quote_qty = 0
        active_buy_volume = 0
        active_sell_volume = 0
        active_buy_volume_in_quote_asset = 0
        active_sell_volume_in_quote_asset = 0
        while cur_trade_i < len(trades) and int(
            trades[cur_trade_i]["data"]["T"]
        ) <= int(tick["data"]["T"]):
            cur_trade = trades[cur_trade_i]
            n_trades += 1
            last_p = float(cur_trade["data"]["p"])
            last_q = float(cur_trade["data"]["q"])
            qty += last_q
            quote_qty += last_q * last_p

            if bool(cur_trade["data"]["is_buyer_maker"]):
                active_sell_volume += last_q
                active_sell_volume_in_quote_asset += last_q * last_p
            else:
                active_buy_volume += last_q
                active_buy_volume_in_quote_asset += last_q * last_p
            cur_trade_i += 1

        format_tick["trades"] = n_trades
        format_tick["price"] = quote_qty / qty if qty > 0 else float("nan")
        format_tick["qty"] = qty
        format_tick["quote_qty"] = quote_qty
        format_tick["active.buy.qty"] = active_buy_volume  # 主动买入量
        format_tick["active.sell.qty"] = active_sell_volume  # 主动买入量

        format_tick["active.buy.quote_qty"] = active_buy_volume_in_quote_asset  # 主动买入量
        format_tick[
            "active.sell.quote_qty"
        ] = active_sell_volume_in_quote_asset  # 主动买入量

        fused_orderbook.append(format_tick)
    if DEBUG and num_lost > 0:
        print(f"[WARNING] num_lost: {num_lost} for {symbol} on {date}")

    return fused_orderbook


def parse_date_file_get_orderbook_and_trades(symbol, date):
    source1 = "depth20@500ms"
    source2 = "aggTrade"
    filename = f"{symbol.lower()}@{source1}-{date}.log"
    filename_2 = filename + ".tar.gz"
    all_order_bookdata = []
    with tarfile.open(os.path.join(DATA_DIR, symbol, filename_2), "r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                file = tar.extractfile(member)
                if file is not None:
                    for line in file:
                        meta_data = json.loads(line.strip())
                        all_order_bookdata.append(meta_data)

    filename = f"{symbol.lower()}@{source2}-{date}.log"
    filename_2 = filename + ".tar.gz"
    all_trades = []
    with tarfile.open(os.path.join(DATA_DIR, symbol, filename_2), "r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                file = tar.extractfile(member)
                if file is not None:
                    for line in file:
                        meta_data = json.loads(line.strip())
                        all_trades.append(meta_data)

    return all_order_bookdata, all_trades


def process(symbol, date):
    try:
        ticks, trades = parse_date_file_get_orderbook_and_trades(symbol, date)
        fused_orderbook = combine_orderbook_trades(ticks, trades)
        df = pd.DataFrame(fused_orderbook)
        expected = 2 * 60 * 60 * 24
        print(f"{df.shape[0]}/{expected} ticks for {symbol} on {date}")
        serialized = cPickle.dumps(df)
        path = os.path.join(DATA_DIR2, symbol, f"{date}.pkl")
        path_raw = os.path.join(DATA_DIR, symbol)
        with gzip.open(path, "wb", compresslevel=1) as file_object:
            file_object.write(serialized)

    except Exception as e:
        print(f"error in {symbol} {date}, e={e}")
        traceback.print_exc()
        raise e


def main(symbol, dates=None):
    symbol = symbol.lower()
    symbol_dir = os.path.join(DATA_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR2, symbol), exist_ok=True)
    if dates is None:
        cmd = f"aws s3 cp s3://{bucket_name}/{prefix}/{symbol} {symbol_dir} --recursive"
        res = os.system(cmd)
    else:
        for date in dates:
            file1 = f"{symbol}@depth20@500ms-{date}.log.tar.gz"
            file2 = f"{symbol}@aggTrade-{date}.log.tar.gz"
            cmd = f"aws s3 cp s3://{bucket_name}/{prefix}/{symbol}/{file1} {symbol_dir}/{file1}"
            cmd2 = f"aws s3 cp s3://{bucket_name}/{prefix}/{symbol}/{file2} {symbol_dir}/{file2}"
            res = os.system(cmd)
            res = os.system(cmd2) and res

    if res < 0:
        exit(-1)
    dates = set()
    for file in os.listdir(symbol_dir):
        if file.endswith(".gz"):
            date_str = file[-21:-11]
            dates.add(date_str)
    dates = sorted(list(dates))
    workers = ThreadPoolExecutor(16)
    for date in dates:
        workers.submit(process, symbol, date)
    workers.shutdown()
    shutil.rmtree(symbol_dir)


if __name__ == "__main__":
    # pull data
    symbols = ["btcusdt", "ethusdt", "bnbusdt", "1000pepeusdt"]
    for symbol in symbols:
        print("process ", symbol)
        # main(symbol, dates=['2023-06-10', '2023-06-13', '2023-06-14', '2023-06-15', '2023-06-16', '2023-06-17'])
        main(symbol)
