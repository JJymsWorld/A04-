# utf-8
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from passenger_identify.base import reduce_mem_usage, read_csv, datapath, drop_features, tmppath, \
    Box_Cox, train_drop_features, getTrainTest, minmax_target, target, minmax, getTrainTest_np
from passenger_identify.base import evaluation
from imblearn.combine import SMOTEENN, SMOTETomek

# train = reduce_mem_usage(read_csv(tmppath + "BOX_train.csv"))
# test = reduce_mem_usage(read_csv(tmppath + 'BOX_test.csv'))
#
# X_train = train.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)  # 去除部分取值过多的离散型特征
# X_test = test.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)
# Y_train = train['emd_lable2'].astype(int)
#
# discrete_list = list(set(discrete_list) - set(train_drop_features))  # 训练中需要剔除的特征都是离散型的特征
# continue_list = list(set(X_train.columns.tolist()) - set(discrete_list))
#
# X_train, X_test = minmax_target(X_train, X_test, Y_train, continue_list, discrete_list)
train = reduce_mem_usage(read_csv(tmppath + 'sub/BOX_Subtrain.csv'))
test = reduce_mem_usage(read_csv(tmppath + 'sub/BOX_Subtest.csv'))

X_train = train.drop(['emd_lable2'], axis=1)  # 去除部分取值过多的离散型特征
Y_train = train['emd_lable2'].astype(int)

discrete_list = ['seg_flight', 'seg_cabin', 'pref_orig_m6_2', 'pref_line_y1_2',
                 'pref_line_y1_3', 'pref_line_y2_2', 'pref_line_y2_3', 'pref_line_y3_3'
    , 'pref_line_y3_4', 'pref_line_y3_5', 'pref_aircraft_y3_3', 'pref_city_y1_2',
                 'pref_city_y3_4', 'pref_dest_city_m6', 'pref_dest_city_y3'
    , 'pref_month_y3_1', 'seg_dep_time_month']  # 训练中需要剔除的特征都是离散型的特征
feature_list = X_train.columns.tolist()
continue_list = list(set(feature_list) - set(discrete_list))

X_train, test = minmax_target(X_train, test, Y_train, continue_list, discrete_list)  # 离散值编码与连续特征归一化

del test, train
x_train, x_test, y_train, y_test = getTrainTest(X_train, Y_train)  # 线下验证，80%训练集，20%验证集

# # 模型交叉验证！
smo = SMOTETomek(random_state=0)  # ratio={1: 300 }，过采样比例可以适当降低一下
x_train, y_train = smo.fit_sample(x_train, y_train)
from lightgbm.sklearn import LGBMClassifier

clf = LGBMClassifier()
clf.feature_importances_()
