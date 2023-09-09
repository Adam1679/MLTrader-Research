import sys
import os

sys.path.insert(0, "/Users/bytedance/MLTrader-deploy")
from research.orderbook_strategies.utils.helper import *
from research.orderbook_strategies.utils.factor_analysis import *


if __name__ == "__main__":
    product = "BTCUSDT"
    signal_name = "dbook.atr.64"
    file_name = "2023-05-01"
    all_signal = auto_get_alldates_signal(signal_name, product)
    tranct = product_info[product]["tranct"]  ## transaction cost of the product
    tranct_ratio = product_info[product][
        "tranct.ratio"
    ]  ## True: based on notional; False: fixed tranct
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
    S = get_good_signal(product, signal_name, date_str=file_name)
    pred = np.asarray(S)
    data = get_data(
        product,
        file_name,
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
    atr = data["atr.4096"][data["good"]].reset_index(drop=True)
    data = data[data["good"]].reset_index(drop=True)
    results = []
    atr_filter = 0.01
    max_spread = 0.1

    # for i in range(len(results)):
    #     print(f"{i}: {results[i]}")
    #     print(f"{i}: {results[i]}")

    results = backtest_par(
        pred,
        data,
        thre=[(thre["open"], thre["close"]) for _, thre in thre_mat.iterrows()],
        atr=atr,
        atr_filter=atr_filter,
        tranct_ratio=tranct_ratio,
        max_spread=max_spread,
        tranct=tranct,
    )
    i = 0
    for thre in thre_mat.iterrows():
        res_for_a_day = backtest(
            pred,
            data,
            thre=(thre[1]["open"], thre[1]["close"]),
            atr=atr,
            atr_filter=atr_filter,
            tranct_ratio=tranct_ratio,
            max_spread=max_spread,
            tranct=tranct,
        )

        assert res_for_a_day["avg_pnl"] == results[i]["avg_pnl"]
        assert res_for_a_day["num"] == results[i]["num"]
        assert res_for_a_day["avg_ret"] == results[i]["avg_ret"]
        assert res_for_a_day["ret"] == results[i]["ret"]
        assert res_for_a_day["final_pnl"] == results[i]["final_pnl"]
        i += 1
