# utf-8
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import LinAlgError
import scipy.stats as stats
from sklearn.model_selection import train_test_split, StratifiedKFold

from passenger_identify.base import reduce_mem_usage, read_csv, datapath, drop_features, tmppath, \
    Box_Cox, train_drop_features, getTrainTest, minmax_target, combine_feature, auto_feature_make
from passenger_identify.base import evaluation

train = reduce_mem_usage(read_csv(tmppath + 'Box_Subtrain.csv'))
test = reduce_mem_usage(read_csv(tmppath + 'Box_Subtest.csv'))

X_train = train.drop(['emd_lable2'], axis=1)  # 去除部分取值过多的离散型特征
Y_train = train['emd_lable2'].astype(int)

discrete_list = ['seg_flight', 'seg_cabin', 'pref_orig_m6_2', 'pref_line_y1_2',
                 'pref_line_y1_3', 'pref_line_y2_2', 'pref_line_y2_3', 'pref_line_y3_3'
    , 'pref_line_y3_4', 'pref_line_y3_5', 'pref_aircraft_y3_3', 'pref_city_y1_2',
                 'pref_city_y3_4', 'pref_dest_city_m6', 'pref_dest_city_y3'
    , 'pref_month_y3_1', 'seg_dep_time_month']  # 训练中需要剔除的特征都是离散型的特征
feature_list = X_train.columns.tolist()
continue_list = list(set(feature_list) - set(discrete_list))

X_train, X_test = minmax_target(X_train, test, Y_train, continue_list, discrete_list)  # 离散值编码与连续特征归一化

x_train, x_test, y_train, y_test = getTrainTest(X_train, Y_train)  # 线下验证，80%训练集，20%验证集

evaluation(x_train, x_test, y_train, y_test, '特征优化结果autoFeatureMake')

# 创建新的特征并对比结果,暂时不用，特征数量非常多，对树类模型可能存在一定的效果
# X_train, X_test = auto_feature_make(X_train, test, continue_list)
# X_train['emd_lable2'] = Y_train
# X_train.to_csv(tmppath + 'feature_make_' + 'train.csv', index=False)
# X_test.to_csv(tmppath + 'feature_make_' + 'test.csv', index=False)

# X_train, X_test = combine_feature(X_train, X_test, discrete_list, continue_list)

# X_train, X_test = minmax_target(X_train, test, Y_train, continue_list, discrete_list)
#
# # 模型交叉验证！
# x_train, x_test, y_train, y_test = getTrainTest(X_train, Y_train)
#
# evaluation(x_train, x_test, y_train, y_test, '特征子集模型验证结果')
