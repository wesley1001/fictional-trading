import numpy as np
import math



def get_lorentzian_distance(self, i: int, featureCount: int, featureSeries: np.ndarray, featureArrays: np.ndarray):
    res = 0
    for j in range(featureCount):
        res += math.log(1+math.fabs(featureSeries[j] - featureArrays[j][i]))
    return res