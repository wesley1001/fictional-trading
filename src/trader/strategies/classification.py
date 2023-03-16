import math
import typing
import numpy as np
import numpy.typing as npt
from tqsdk import tafunc

from ml_extensions_2 import ml_extensions as ml


class Settings:
    def __init__(self,
                 source: np.ndarray,
                 neighborsCount: int = 8,
                 maxBarsBack: int = 2000,
                 featureCount: int = 5,
                 colorCompression: int = 1,
                 showExits: bool = False,
                 useDynamicExits: bool = False):
        self.source = source  # close
        self.neighborsCount = neighborsCount
        self.maxBarsBack = maxBarsBack
        self.featureCount = featureCount
        self.colorCompression = colorCompression
        self.showExits = showExits
        self.useDynamicExits = useDynamicExits

class TradeStatsSettings:
    def __init__(self, showTradeStats: bool = True, useWorstCase: int = False):
        self.showTradeStats = showTradeStats
        self.useWorstCase = useWorstCase

class Label:
    def __init__(self, long: int, short: int, neutral: int):
        self.long = long
        self.short = short
        self.neutral = neutral


class FeatureArrays:
    def __init__(self, f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, f4: np.ndarray, f5: np.ndarray):
        self.f1: np.ndarray = f1
        self.f2: np.ndarray = f2
        self.f3: np.ndarray = f3
        self.f4: np.ndarray = f4
        self.f5: np.ndarray = f5


class FeatureSeries:
    def __init__(self, f1, f2, f3, f4, f5):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.f5 = f5


class MLModel:
    def __init__(self, firstBarIndex: int, trainingLabels: np.ndarray, loopSize: int, lastDistance: float, distancesArray: np.ndarray, predictionsArray: np.ndarray, prediction: int):
        self.firstBarIndex = firstBarIndex
        self.trainingLabels = trainingLabels
        self.loopSize = loopSize
        self.lastDistance = lastDistance
        self.distancesArray = distancesArray
        self.predictionsArray = predictionsArray
        self.prediction = prediction


class FilterSettings:
    def __init__(self, useVolatilityFilter: bool, useRegimeFilter: bool, useAdxFilter: bool, regimeThreshold: float, adxThreshold: int):
        self.useVolatilityFilter = useVolatilityFilter
        self.useRegimeFilter = useRegimeFilter
        self.useAdxFilter = useAdxFilter
        self.regimeThreshold = regimeThreshold
        self.adxThreshold = adxThreshold


class Filter:
    def __init__(self, volatility: bool, regime: bool, adx: bool):
        self.volatility = volatility
        self.regime = regime
        self.adx = adx

# ==========================
# ==== Helper Functions ====
# ==========================


def series_from(feature_string, _close, _high, _low, _hlc3, f_paramA, f_paramB):
    if feature_string == "RSI":
        return ml.n_rsi(_close, f_paramA, f_paramB)
    elif feature_string == "WT":
        return ml.n_wt(_hlc3, f_paramA, f_paramB)
    elif feature_string == "CCI":
        return ml.n_cci(_close, f_paramA, f_paramB)
    elif feature_string == "ADX":
        return ml.n_adx(_high, _low, _close, f_paramA)


def get_lorentzian_distance(i, featureCount, featureSeries, featureArrays):
    if featureCount == 5:
        return math.log(1+math.fabs(featureSeries.f1 - featureArrays.f1[i])) + \
            math.log(1+math.fabs(featureSeries.f2 - featureArrays.f2[i])) + \
            math.log(1+math.fabs(featureSeries.f3 - featureArrays.f3[i])) + \
            math.log(1+math.fabs(featureSeries.f4 - featureArrays.f4[i])) + \
            math.log(1+math.fabs(featureSeries.f5 - featureArrays.f5[i]))
    elif featureCount == 4:
        return math.log(1+math.fabs(featureSeries.f1 - featureArrays.f1[i])) + \
            math.log(1+math.fabs(featureSeries.f2 - featureArrays.f2[i])) + \
            math.log(1+math.fabs(featureSeries.f3 - featureArrays.f3[i])) + \
            math.log(1+math.fabs(featureSeries.f4 - featureArrays.f4[i]))
    elif featureCount == 3:
        return math.log(1+math.fabs(featureSeries.f1 - featureArrays.f1[i])) + \
            math.log(1+math.fabs(featureSeries.f2 - featureArrays.f2[i])) + \
            math.log(1+math.fabs(featureSeries.f3 - featureArrays.f3[i]))
    elif featureCount == 2:
        return math.log(1+math.fabs(featureSeries.f1 - featureArrays.f1[i])) + \
            math.log(1+math.fabs(featureSeries.f2 - featureArrays.f2[i]))


# ================
# ==== Inputs ====
# ================

# Settings Object: General User-Defined Inputs
