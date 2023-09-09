import sys

sys.path.insert(0, ".")
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytz
from tardis_dev import datasets, get_exchange_details
from tqdm import tqdm

from research.data_binance.utils.binance_util import S_DEFAULTS, get_all_symbols

if __name__ == "__main__":
    # all_um_symbols = get_all_symbols("perpetual")
    # all_um_symbols = [
    #     s for s in all_um_symbols if s.endswith("USDT") or s.endswith("BUSD")
    # ]
    # all_um_usdt_symbols = [s for s in all_um_symbols if s.endswith("USDT")]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=S_DEFAULTS)
    parser.add_argument("--start_date", default="")
    parser.add_argument("--end_date", default="")
    parser.add_argument(
        "--type", default="book_snapshot_25", choices=["book_snapshot_25", "trades"]
    )

    args = parser.parse_args()
    all_um_usdt_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BCHUSDT",
        "XRPUSDT",
        "EOSUSDT",
        "LTCUSDT",
        "TRXUSDT",
        "ETCUSDT",
        "LINKUSDT",
        "XLMUSDT",
        "ADAUSDT",
        "XMRUSDT",
        "DASHUSDT",
        "ZECUSDT",
        "XTZUSDT",
        "BNBUSDT",
        "ATOMUSDT",
        "ONTUSDT",
        "IOTAUSDT",
        "BATUSDT",
        "VETUSDT",
        "NEOUSDT",
        "QTUMUSDT",
        "IOSTUSDT",
        "THETAUSDT",
        "ALGOUSDT",
        "ZILUSDT",
        "KNCUSDT",
        "ZRXUSDT",
        "COMPUSDT",
        "OMGUSDT",
        "DOGEUSDT",
        "SXPUSDT",
        "KAVAUSDT",
        "BANDUSDT",
        "RLCUSDT",
        "WAVESUSDT",
        "MKRUSDT",
        "SNXUSDT",
        "DOTUSDT",
        "DEFIUSDT",
        "YFIUSDT",
        "BALUSDT",
        "CRVUSDT",
        "TRBUSDT",
        "RUNEUSDT",
        "SUSHIUSDT",
        "EGLDUSDT",
        "SOLUSDT",
        "ICXUSDT",
        "STORJUSDT",
        "BLZUSDT",
        "UNIUSDT",
        "AVAXUSDT",
        "FTMUSDT",
        "ENJUSDT",
        "FLMUSDT",
        "TOMOUSDT",
        "RENUSDT",
        "KSMUSDT",
        "NEARUSDT",
        "AAVEUSDT",
        "FILUSDT",
        "RSRUSDT",
        "LRCUSDT",
        "MATICUSDT",
        "OCEANUSDT",
        "BELUSDT",
        "CTKUSDT",
        "AXSUSDT",
        "ALPHAUSDT",
        "ZENUSDT",
        "SKLUSDT",
        "GRTUSDT",
        "1INCHUSDT",
        "CHZUSDT",
        "SANDUSDT",
        "ANKRUSDT",
        "LITUSDT",
        "UNFIUSDT",
        "REEFUSDT",
        "RVNUSDT",
        "SFPUSDT",
        "XEMUSDT",
        "COTIUSDT",
        "CHRUSDT",
        "MANAUSDT",
        "ALICEUSDT",
        "HBARUSDT",
        "ONEUSDT",
        "LINAUSDT",
        "STMXUSDT",
        "DENTUSDT",
        "CELRUSDT",
        "HOTUSDT",
        "MTLUSDT",
        "OGNUSDT",
        "NKNUSDT",
        "DGBUSDT",
        "1000SHIBUSDT",
        "BAKEUSDT",
        "GTCUSDT",
        "BTCDOMUSDT",
        "IOTXUSDT",
        "AUDIOUSDT",
        "C98USDT",
        "MASKUSDT",
        "ATAUSDT",
        "DYDXUSDT",
        "1000XECUSDT",
        "GALAUSDT",
        "CELOUSDT",
        "ARUSDT",
        "KLAYUSDT",
        "ARPAUSDT",
        "CTSIUSDT",
        "LPTUSDT",
        "ENSUSDT",
        "PEOPLEUSDT",
        "ANTUSDT",
        "ROSEUSDT",
        "DUSKUSDT",
        "FLOWUSDT",
        "IMXUSDT",
        "API3USDT",
        "GMTUSDT",
        "APEUSDT",
        "WOOUSDT",
        "JASMYUSDT",
        "DARUSDT",
        "GALUSDT",
        "OPUSDT",
        "INJUSDT",
        "STGUSDT",
        "FOOTBALLUSDT",
        "SPELLUSDT",
        "1000LUNCUSDT",
        "LUNA2USDT",
        "LDOUSDT",
        "CVXUSDT",
        "ICPUSDT",
        "APTUSDT",
        "QNTUSDT",
        "BLUEBIRDUSDT",
        "FETUSDT",
        "FXSUSDT",
        "HOOKUSDT",
        "MAGICUSDT",
        "TUSDT",
        "RNDRUSDT",
        "HIGHUSDT",
        "MINAUSDT",
        "ASTRUSDT",
        "AGIXUSDT",
        "PHBUSDT",
        "GMXUSDT",
        "CFXUSDT",
        "STXUSDT",
        "BNXUSDT",
        "ACHUSDT",
        "SSVUSDT",
        "CKBUSDT",
        "PERPUSDT",
        "TRUUSDT",
        "LQTYUSDT",
        "USDCUSDT",
        "IDUSDT",
        "ARBUSDT",
        "JOEUSDT",
        "TLMUSDT",
        "AMBUSDT",
        "LEVERUSDT",
        "RDNTUSDT",
        "HFTUSDT",
        "XVSUSDT",
        "BLURUSDT",
        "EDUUSDT",
        "IDEXUSDT",
        "SUIUSDT",
        "1000PEPEUSDT",
        "1000FLOKIUSDT",
        "UMAUSDT",
        "RADUSDT",
        "KEYUSDT",
        "COMBOUSDT",
    ]

    # all_um_usdt_symbols = ["btcusdt", "suiusdt", "stxusdt", '1000pepeusdt', 'cfxusdt', 'highusdt', 'ethusdt', 'linausdt']
    # all_um_usdt_symbols = S_DEFAULTS
    print(
        f"#all_um_usdt_symbols={len(all_um_usdt_symbols)}, all_um_usdt_symbols={all_um_usdt_symbols}"
    )
    KEY = "TD.2tzNHao0ukiP7dY1.hz3RgOQ4YYxIgZr.eMGMfz8ZNkX6SJv.O-yBf1tKtM-EE8S.vaCoKSlqn7ZPYLR.aFja"

    binance_futures = get_exchange_details("binance-futures")
    info_dict = {}
    for symbol_info in binance_futures["availableSymbols"]:
        info_dict[symbol_info["id"]] = symbol_info
    for symbol in tqdm(all_um_usdt_symbols):
        print(symbol)
        try:
            start_date = args.start_date
            to_date = args.end_date
            if to_date == "":
                to_date = (datetime.now(pytz.utc) - timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )
            available_symbols = info_dict[symbol.lower()]["availableSince"][:10]
            start_date = max(start_date, available_symbols)
            if (datetime.strptime(to_date, "%Y-%m-%d") - datetime.strptime(
                start_date, "%Y-%m-%d"
            )).days < 150:
                print(f"skip {symbol}")
                continue
            print("downloading {} from {} to {}".format(symbol, start_date, to_date))
            datasets.download(
                exchange="binance-futures",
                # data_types=["book_snapshot_25"],
                data_types=[args.type],
                # data_types=["incremental_book_L2"],
                # from_date="2023-06-01",
                from_date=start_date,
                to_date=to_date,
                symbols=[symbol],
                api_key=KEY,
                # download_dir=f"/Users/bytedance/binance_data/data/futures/um/tardis_data_raw/{symbol}/",
                download_dir=f"/Volumes/AdamDrive/binance_data/data/futures/um/{args.type}/{symbol.upper()}/",
                # download_dir=f"/Volumes/My Passport/binance_data/data/futures/um/incremental_l2/{symbol}/",
                concurrency=15,
            )
        except Exception as e:
            print(e)
            continue

    # print(binance_futures['availableSymbols'])
