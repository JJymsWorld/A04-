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
    Box_Cox, train_drop_features, getTrainTest, minmax_target, combine_feature, auto_feature_make, xgb_clf, lgb_clf, \
    getTrainTest_np
from passenger_identify.base import evaluation

train = reduce_mem_usage(read_csv(tmppath + 'sub/combine_feature_train.csv'))
test = reduce_mem_usage(read_csv(tmppath + 'sub/combine_feature_test.csv'))

X_train = train.drop(['emd_lable2'], axis=1)  # 去除部分取值过多的离散型特征
Y_train = train['emd_lable2'].astype(int)

discrete_list = ['seg_flight', 'seg_cabin', 'pref_orig_m6_2', 'pref_line_y1_2',
                 'pref_line_y1_3', 'pref_line_y2_2', 'pref_line_y2_3', 'pref_line_y3_3'
    , 'pref_line_y3_4', 'pref_line_y3_5', 'pref_aircraft_y3_3', 'pref_city_y1_2',
                 'pref_city_y3_4', 'pref_dest_city_m6', 'pref_dest_city_y3'
    , 'pref_month_y3_1', 'seg_dep_time_month']  #
feature_list = X_train.columns.tolist()
continue_list = list(set(feature_list) - set(discrete_list))

X_train, test = minmax_target(X_train, test, Y_train, continue_list, discrete_list)  # 离散值编码与连续特征归一化

X_train = X_train.values
Y_train = Y_train.values
del test, train
x_train, x_test, y_train, y_test = getTrainTest_np(X_train, Y_train)  # 线下验证，80%训练集，20%验证集

evaluation(x_train, x_test, y_train, y_test, 'temp')


# 创建新的特征并对比结果,暂时不用，特征数量非常多，对树类模型可能存在一定的效果
# X_train, X_test = auto_feature_make(X_train, test, continue_list)
# X_train['emd_lable2'] = Y_train
# X_train.to_csv(tmppath + 'feature_make_' + 'train.csv', index=False)
# X_test.to_csv(tmppath + 'feature_make_' + 'test.csv', index=False)
#
# X_train, X_test = combine_feature(X_train, test, discrete_list, continue_list)
# X_train['emd_lable2'] = Y_train
# X_train.to_csv(tmppath + 'combine_feature_' + 'train.csv', index=False)
# X_test.to_csv(tmppath + 'combine_feature_' + 'test.csv', index=False)

# stacking分类特征，将这些特征全部合并，做特征选择！！！这里保存的结果是已经经过归一化的数据！！！且离散值已经编码了

# X_train = X_train.values
# X_test = X_test.values
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# clf_list = [lgb_clf, xgb_clf]
# clf_list_col = ['lgb_clf', 'xgb_clf']
#
# column_list = []
# train_data_list = []
# test_data_list = []
# for clf in clf_list:
#     train_data, test_data, clf_name = clf(X_train, Y_train, X_test, kf, label_split=Y_train)
#     train_data_list.append(train_data)
#     test_data_list.append(test_data)
# train_stacking = np.concatenate(train_data_list, axis=1)
# test_stacking = np.concatenate(test_data_list, axis=1)
#
# # # 合并所有特征
# train = pd.DataFrame(np.concatenate([X_train, train_stacking], axis=1))
# test = np.concatenate([X_test, test_stacking], axis=1)
#
# df_train_all = pd.DataFrame(train)
# df_train_all.columns = feature_list + clf_list_col
# df_test_all = pd.DataFrame(test)
# df_test_all.columns = feature_list + clf_list_col
#
# df_train_all['emd_label2'] = Y_train
#
# df_train_all.to_csv('sub_train_stacking.csv', header=True, index=False)
# df_test_all.to_csv('sub_test_stacking.csv', header=True, index=False)
