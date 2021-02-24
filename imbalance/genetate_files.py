# 生成平衡数据集并保存结果！
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
    Box_Cox, train_drop_features, getTrainTest, minmax_target, target, minmax
from passenger_identify.base import evaluation
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import ClusterCentroids, TomekLinks, NearMiss, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE

train = reduce_mem_usage(read_csv(tmppath + "BOX_train.csv"))
test = reduce_mem_usage(read_csv(tmppath + 'BOX_test.csv'))

X_train = train.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)  # 去除部分取值过多的离散型特征
X_test = test.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)
Y_train = train['emd_lable2'].astype(int)

discrete_list = list(set(discrete_list) - set(train_drop_features))  # 训练中需要剔除的特征都是离散型的特征
continue_list = list(set(X_train.columns.tolist()) - set(discrete_list))

X_train, X_test = target(X_train, X_test, Y_train, discrete_list)

# 模型交叉验证！
x_train, x_test, y_train, y_test = getTrainTest(X_train, Y_train)
count_list = [3000, 5000, 7000, 11000, 13000]
for count in count_list:
    smo = SMOTETomek(random_state=0, sampling_strategy={1: count})  # ratio={1: 300 }，过采样比例可以适当降低一下
    x_train, y_train = smo.fit_sample(x_train, y_train)

    x_train, x_test = minmax(x_train, x_test, continue_list)

    print(Counter(y_train))
    print(Counter(y_test))

    evaluation(x_train, x_test, y_train, y_test, 'SMOTETomek_' + str(count))

