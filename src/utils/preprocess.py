# import modin.pandas as pd
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from utils.constant import mlp_units_dict, low_by_label_length, high_by_label_length
import pytz
from datetime import datetime

from tqsdk import tafunc
from tqsdk.ta import CCI, DMI



def process_prev_close_spread(df: pd.DataFrame) -> pd.DataFrame:
    print("Processing prev close spread")
    start = time.time()
    price_cols = ["open", "high", "low", "close"]
    # Get underlying_symbols list sort by datatime ascending
    underlying_symbols = df["underlying_symbol"].unique()
    prev_close_spread_price = 0
    sorted_underlying_symbols = sorted(underlying_symbols, key=lambda x: df[df["underlying_symbol"] == x].iloc[0]["datetime"])
    new_df = pd.DataFrame(columns=df.columns)
    for symbol in tqdm(sorted_underlying_symbols):
        tmp_df = df[df["underlying_symbol"] == symbol]
        if prev_close_spread_price == 0:
            # Get first close price
            prev_close_spread_price = tmp_df.iloc[-1]["close"]
        else:
            # Get price spread by diff of close price
            offset = prev_close_spread_price - tmp_df.iloc[0]["close"]
            # print(prev_close_spread_price)
            tmp_df[price_cols] += offset
            prev_close_spread_price = tmp_df.iloc[-1]["close"]
        new_df = pd.concat([new_df, tmp_df])
    print("Processing prev close spread time:", time.time() - start)
    del df
    return new_df

def set_training_label(df: pd.DataFrame, max_label_length, n_classes, interval):
    """
        check if the price changes by percentage within max_label_length step
        n_classes = 5
        label range:
            -inf ~ -high | -high ~ -low | -low ~ low | low ~ high | high ~ inf
        e.g.
            -inf ~ -0.01 | -0.01 ~ -0.005 | -0.005 ~ 0.005 | 0.005 ~ 0.01 | 0.01 ~ inf
    """
    # df = pd.DataFrame(df)
    # low = low_by_label_length[interval][max_label_length]
    # high = high_by_label_length[interval][max_label_length]

    # reset the index from 0 to df.shape[0]
    df = df.reset_index(drop=True).reset_index()
    def check_volatility(v):
        if v < 0:
            return 0
        elif v == 0:
            # hold when price change is 0
            return 1
        elif v > 0:
            return 2
        else:
            return np.nan 

    numerator = df['close'].to_numpy()[max_label_length:]
    denominator = df['close'].to_numpy()[:-max_label_length]
    vol = numerator / denominator - 1
    df = df.iloc[:-max_label_length]
    df['label'] = vol
    df['label'] = df['label'].apply(lambda x: check_volatility(x))
    # print(df["label"].value_counts())
    # print("Class distribution: ", df["label"].value_counts() / df.shape[0])
    return df

    
def process_datatime(df: pd.DataFrame, is_daytime: bool = True):
    print("Processing datetime")
    start = time.time()
    exchange_tz = pytz.timezone('Asia/Shanghai')
    df["datetime"] =  df["datetime"].apply(lambda x: datetime.utcfromtimestamp(x.value / 1e9).astimezone(exchange_tz))
    if is_daytime:
        df["is_daytime"] = df["datetime"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    end = time.time()
    print("Processing datetime time:", end - start)
    return df


def process_by_group(g, max_encode_length, max_label_length, n_classes, interval) -> pd.DataFrame:  
    """
        Deprecated function
    """  
    print("Setting volatility label")    
    g = set_training_label(g, max_label_length, n_classes, interval)
    target_list = []
    train_list = []
    for i in tqdm(range(g.shape[0] - max_encode_length)):
        # yield training, target 
        train, target = g.iloc[i:i+max_encode_length].to_numpy(dtype=np.float32), g.iloc[i+max_encode_length]["vol"]
        train_list.append(train)
        target_list.append(target)
    return train_list, target_list


def timeseries_normalize(data: np.ndarray):
    """
        Deprecated function
    """
    from sklearn.preprocessing import minmax_scale

    print("Normalizing data", data.shape)
    normalize = lambda subset: minmax_scale(subset, axis=0)
    for i in range(data.shape[0]):
        data[i] = normalize(data[i])
    return data

def normalize(src: np.ndarray):
    # normalizes to values from 0 -1
    min_val = np.min(src)
    return (src - min_val) / (np.max(src) - min_val)

def rsi(df: pd.DataFrame, n: int):
    lc = df["close"].shift(1)
    rsi = tafunc.sma(pd.Series(np.where(df["close"] - lc > 0, df["close"] - lc, 0)), n, 1) / \
          tafunc.sma(np.absolute(df["close"] - lc), n, 1) * 100
    return rsi

def wavetrend_moving_average(df: pd.DataFrame, n1, n2):
    # ap = (df["high"] + df["low"] + df["close"]) / 3
    ap = df["close"]
    esa = tafunc.ema(ap, n1)
    d = tafunc.ema(np.absolute(ap - esa), n1)
    ci = (ap - esa) / (0.015 * d)
    tci = tafunc.ema(ci, n2)
    return normalize(tci)

def add_factors(df: pd.DataFrame, factor_names: str):
    """
        Add factors to dataframe
    """
    for factor in factor_names:
        print("Adding factor:", factor)
        if factor.startswith("ma_"):
            # add moving average
            n = int(factor.split("_")[1])
            df[factor] = tafunc.ema(df["close"], n)
        elif factor.startswith("rsi"):
            n = int(factor.split("_")[1])
            df['rsi'] = rsi(df, n)
        elif factor.startswith("wt"):
            n1 = int(factor.split("_")[1])
            n2 = int(factor.split("_")[2])
            df["wt"] = wavetrend_moving_average(df, n1, n2)
        elif factor.startswith("cci"):
            n = int(factor.split("_")[1])
            df["cci"] = CCI(df, n)['cci']
        elif factor.startswith("adx"):
            n = int(factor.split("_")[1])
            m = int(factor.split("_")[2])
            df["adx"] = DMI(df, n)['adx']
        else:
            raise ValueError("Invalid factor name")
    return df

