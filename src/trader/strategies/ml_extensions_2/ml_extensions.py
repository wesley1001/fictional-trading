import math
import typing
import numpy as np
import tqsdk
from tqsdk import tafunc
from tqsdk import ta
import pandas as pd


def rescale(
    src: np.ndarray,
    oldMin: float,
    oldMax: float,
    newMin: float,
    newMax: float,
):
    return newMin + (newMax - newMin) * (src - oldMin) / max(oldMax - oldMin, 10e-10)


def normalize(
    src: np.ndarray,
    #   _min: float, _max: float
):
    # normalizes to values from 0 -1
    min_val = np.min(src)
    return (src - min_val) / (np.max(src) - min_val)

    # return _min + (_max - _min) * (src - np.min(src)) / np.max(src)

def rsi(_close: np.ndarray, n=14):
    lc = _close.shift(1)
    return tafunc.sma(pd.Series(np.where(_close - lc > 0, _close - lc, 0)), n, 1) / tafunc.sma(np.absolute(_close - lc), n, 1) * 100


def n_rsi(_close: np.ndarray, f_paramA, f_paramB):
    rsi = rsi(_close, f_paramA)
    return rescale(tafunc.ema(rsi, f_paramB), 0, 100, 0, 1)

# util functions


def cut_data_to_same_len(data_set: tuple or list, get_list=False):
    # data tuple in and out
    min_len = None
    cutted_data: list = []

    for data in data_set:
        if data is not None:
            _len = len(data)
            if not min_len or _len < min_len:
                min_len = _len
    for data in data_set:
        if data is not None:
            cutted_data.append(data[len(data) - min_len:])
        else:
            cutted_data.append(None)
    if get_list:
        return cutted_data
    return tuple(cutted_data)


def calculate_rma(
    src: np.ndarray, length
) -> np.ndarray:
    # TODO not the same as on here: https://www.tradingview.com/pine-script-reference/v5/#fun_ta%7Bdot%7Drma
    alpha = 1 / length
    # cut first data as its not very accurate
    sma = tafunc.sma(src, length)[50:]
    src, sma = cut_data_to_same_len((src, sma))
    rma: typing.List[float] = [sma[0]]
    for index in range(1, len(src)):
        rma.append((src[index] * alpha) + ((1 - alpha) * rma[-1]))
    return np.array(rma)

########


def n_wt(_hlc3: np.ndarray, f_paramA, f_paramB):
    ema1 = tafunc.ema(_hlc3, f_paramA)
    ema2 = tafunc.ema(abs(_hlc3 - ema1), f_paramA)
    ci = (_hlc3[1:] - ema1[1:]) / (0.015 * ema2[1:])
    wt1 = tafunc.ema(ci, f_paramB)  # tci
    wt2 = tafunc.sma(wt1, 4)
    wt1, wt2 = cut_data_to_same_len((wt1, wt2))
    return normalize(wt1 - wt2)  # , 0, 1)


def n_cci(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    f_paramA,
    f_paramB,
):
    # use closes, closes, closes to get same cci as on tradingview
    return normalize(
        tafunc.ema(tafunc.cci(closes, closes, closes, f_paramA), f_paramB)
    )  # , 0, 1)


def n_adx(
    highSrc: np.ndarray,
    lowSrc: np.ndarray,
    closeSrc: np.ndarray,
    f_paramA: int,
):
    length: int = f_paramA
    data_length: int = len(highSrc)
    trSmooth: typing.List[float] = [0]
    smoothnegMovement: typing.List[float] = [0]
    smoothDirectionalMovementPlus: typing.List[float] = [0]
    dx: typing.List[float] = []

    for index in range(1, data_length):
        tr = max(
            max(
                highSrc[index] - lowSrc[index],
                abs(highSrc[index] - closeSrc[index - 1]),
            ),
            abs(lowSrc[index] - closeSrc[index - 1]),
        )
        directionalMovementPlus = (
            max(highSrc[index] - highSrc[index - 1], 0)
            if highSrc[index] - highSrc[index - 1] > lowSrc[index - 1] - lowSrc[index]
            else 0
        )
        negMovement = (
            max(lowSrc[index - 1] - lowSrc[index], 0)
            if lowSrc[index - 1] - lowSrc[index] > highSrc[index] - highSrc[index - 1]
            else 0
        )
        trSmooth.append(trSmooth[-1] - trSmooth[-1] / length + tr)
        smoothDirectionalMovementPlus.append(
            smoothDirectionalMovementPlus[-1]
            - smoothDirectionalMovementPlus[-1] / length
            + directionalMovementPlus
        )
        smoothnegMovement.append(
            smoothnegMovement[-1] -
            smoothnegMovement[-1] / length + negMovement
        )
        diPositive = smoothDirectionalMovementPlus[-1] / trSmooth[-1] * 100
        diNegative = smoothnegMovement[-1] / trSmooth[-1] * 100

        if index > 3:
            # skip early candles as its division by 0
            dx.append(abs(diPositive - diNegative) /
                      (diPositive + diNegative) * 100)
    dx = np.array(dx)
    adx = calculate_rma(dx, length)
    return rescale(adx, 0, 100, 0, 1)


def regime_filter(
    ohlc4: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    threshold: float,
    use_regime_filter: bool,
) -> np.ndarray:
    data_length = len(ohlc4)
    if not use_regime_filter:
        return np.array([True] * data_length)
    # Calculate the slope of the curve.
    values_1: list = [0.0]
    values_2: list = [0.0]
    klmfs: list = [0.0]
    abs_curve_slope: list = []
    for index in range(1, data_length):
        value_1 = 0.2 * (ohlc4[index] - ohlc4[index - 1]) + 0.8 * values_1[-1]
        value_2 = 0.1 * (highs[index] - lows[index]) + 0.8 * values_2[-1]
        values_1.append(value_1)
        values_2.append(value_2)
        omega = abs(value_1 / value_2)
        alpha = (-pow(omega, 2) +
                 math.sqrt(pow(omega, 4) + 16 * pow(omega, 2))) / 8
        klmfs.append(alpha * ohlc4[index] + (1 - alpha) * klmfs[-1])
        abs_curve_slope.append(abs(klmfs[-1] - klmfs[-2]))
    abs_curve_slope: np.ndarray = np.array(abs_curve_slope)
    exponentialAverageAbsCurveSlope: np.ndarray = tafunc.ema(
        abs_curve_slope, 200
    )
    (
        exponentialAverageAbsCurveSlope,
        abs_curve_slope,
    ) = cut_data_to_same_len(
        (exponentialAverageAbsCurveSlope, abs_curve_slope)
    )
    normalized_slope_decline: np.ndarray = (
        abs_curve_slope - exponentialAverageAbsCurveSlope
    ) / exponentialAverageAbsCurveSlope
    # Calculate the slope of the curve.

    return normalized_slope_decline >= threshold


def filter_adx(
    candle_closes: np.ndarray,
    candle_highs: np.ndarray,
    candle_lows: np.ndarray,
    length: int,
    adx_threshold: int,
    use_adx_filter: bool,
) -> np.ndarray:
    data_length: int = len(candle_closes)
    if not use_adx_filter:
        return np.array([True] * data_length)
    tr_smooths: typing.List[float] = [0.0]
    smoothneg_movements: typing.List[float] = [0.0]
    smooth_directional_movement_plus: typing.List[float] = [0.0]
    dx: typing.List[float] = []
    for index in range(1, data_length):
        tr: float = max(
            max(
                candle_highs[index] - candle_lows[index],
                abs(candle_highs[index] - candle_closes[-2]),
            ),
            abs(candle_lows[index] - candle_closes[-2]),
        )
        directional_movement_plus: float = (
            max(candle_highs[index] - candle_highs[-2], 0)
            if candle_highs[index] - candle_highs[-2]
            > candle_lows[-2] - candle_lows[index]
            else 0
        )
        negMovement: float = (
            max(candle_lows[-2] - candle_lows[index], 0)
            if candle_lows[-2] - candle_lows[index]
            > candle_highs[index] - candle_highs[-2]
            else 0
        )
        tr_smooths.append(tr_smooths[-1] - tr_smooths[-1] / length + tr)

        smooth_directional_movement_plus.append(
            smooth_directional_movement_plus[-1]
            - smooth_directional_movement_plus[-1] / length
            + directional_movement_plus
        )

        smoothneg_movements.append(
            smoothneg_movements[-1] -
            smoothneg_movements[-1] / length + negMovement
        )

        di_positive = smooth_directional_movement_plus[-1] / \
            tr_smooths[-1] * 100
        di_negative = smoothneg_movements[-1] / tr_smooths[-1] * 100
        if index > 3:
            # skip early candles as its division by 0
            dx.append(
                abs(di_positive - di_negative) /
                (di_positive + di_negative) * 100
            )
    dx: np.ndarray = np.array(dx)
    adx: np.ndarray = calculate_rma(dx, length)
    return adx > adx_threshold


def filter_volatility(
    candle_highs: np.ndarray,
    candle_lows: np.ndarray,
    candle_closes: np.ndarray,
    min_length: int = 1,
    max_length: int = 10,
    use_volatility_filter: bool = True,
) -> np.ndarray:
    if not use_volatility_filter:
        return np.array([True] * len(candle_closes)), None, None
    recentAtr = tafunc.atr(candle_highs, candle_lows,
                           candle_closes, min_length)
    historicalAtr = tafunc.atr(
        candle_highs, candle_lows, candle_closes, max_length)
    recentAtr, historicalAtr = cut_data_to_same_len(
        (recentAtr, historicalAtr)
    )
    return recentAtr > historicalAtr, recentAtr, historicalAtr
