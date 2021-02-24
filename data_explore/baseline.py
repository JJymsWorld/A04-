# utf-8
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import LinAlgError
import scipy.stats as stats
from sklearn.model_selection import train_test_split, StratifiedKFold

from passenger_identify.base import reduce_mem_usage, read_csv, datapath, drop_features, discrete_list, tmppath, \
    Box_Cox, train_drop_features, getTrainTest, minmax_target
from passenger_identify.base import evaluation

# 只需要读入训练集并进行交叉验证即可
# train, test = Box_Cox()
train = reduce_mem_usage(read_csv(tmppath + "BOX_train.csv"))  # BOX_cox数据不存在drop_features数据
test = reduce_mem_usage(read_csv(tmppath + 'BOX_test.csv'))
# train = reduce_mem_usage(read_csv(tmppath + "date_train.csv").drop(drop_features, axis=1))
# test = reduce_mem_usage(read_csv(tmppath + 'date_test.csv').drop(drop_features, axis=1))

X_train = train.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)  # 去除部分取值过多的离散型特征
X_test = test.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)
Y_train = train['emd_lable2'].astype(int)

discrete_list = list(set(discrete_list) - set(train_drop_features))  # 训练中需要剔除的特征都是离散型的特征
continue_list = list(set(X_train.columns.tolist()) - set(discrete_list))

X_train, X_test = minmax_target(X_train, X_test, Y_train, continue_list, discrete_list)

# 模型交叉验证！
x_train, x_test, y_train, y_test = getTrainTest(X_train, Y_train)
print(Counter(y_train))
print(Counter(y_test))

evaluation(x_train, x_test, y_train, y_test, 'BOX_Cox')
