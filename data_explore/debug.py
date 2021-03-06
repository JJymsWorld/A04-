import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minepy.mine import MINE
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler


def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def calc_condition_ent(x, y):
    """
        calculate ent H(x|y)
    """

    # calc ent(x|y)
    y_value_list = set([y[i] for i in range(y.shape[0])])
    ent = 0.0
    for y_value in y_value_list:
        sub_x = x[y == y_value]
        temp_ent = calc_ent(sub_x)
        ent += (float(sub_x.shape[0]) / x.shape[0]) * temp_ent

    return ent


