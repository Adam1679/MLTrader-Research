import os
from pathlib import Path

from binance_util import query_ls_history

if __name__ == "__main__":
    dir_root = Path("/Users/bytedance/binance_data/data/futures/um/monthly/klines")
    folder = "/Users/bytedance/binance_data/data"
    all_spot_symbols = os.listdir(dir_root)
    all_spot_symbols = [s for s in all_spot_symbols if "USDT" in s or "BUSD" in s]
    for symbol in all_spot_symbols:
        query_ls_history(symbol, folder=folder)
