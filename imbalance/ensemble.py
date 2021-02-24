# utf-8
from collections import Counter
#################模型欠采样集成学习方法，感觉效果很一般#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import LinAlgError
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from passenger_identify.base import reduce_mem_usage, read_csv, datapath, drop_features, discrete_list, tmppath, \
    Box_Cox, train_drop_features, getTrainTest, minmax_target, target, minmax
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier

#
train = reduce_mem_usage(read_csv(tmppath + "BOX_train.csv"))
test = reduce_mem_usage(read_csv(tmppath + 'BOX_test.csv'))

X_train = train.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)  # 去除部分取值过多的离散型特征
X_test = test.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)
Y_train = train['emd_lable2'].astype(int)

discrete_list = list(set(discrete_list) - set(train_drop_features))  # 训练中需要剔除的特征都是离散型的特征
continue_list = list(set(X_train.columns.tolist()) - set(discrete_list))

X_train, X_test = minmax_target(X_train, X_test, Y_train, continue_list, discrete_list)  # 直接归一化，下采样需要直接进行训练与预测

# 模型交叉验证！
x_train, x_test, y_train, y_test = getTrainTest(X_train, Y_train)
print(Counter(y_train))
print(Counter(y_test))

eec = EasyEnsembleClassifier(random_state=42, n_estimators=50)  ###结果只有0.25
eec.fit(x_train, y_train)  # doctest: +ELLIPSIS
y_pred = eec.predict(x_test)

print(f1_score(y_test, y_pred))

# # 使用make_classification生成样本数据
# from collections import Counter
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from imblearn.ensemble import EasyEnsembleClassifier  # doctest: +NORMALIZE_WHITESPACE
#
# X, y = make_classification(n_classes=2, class_sep=2,
#                            weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
#                            n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
# print('Original dataset shape %s' % Counter(y))
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# print(X_train.shape,X_test.shape)
# eec = EasyEnsembleClassifier(random_state=42)
# eec.fit(X_train, y_train)  # doctest: +ELLIPSIS
# y_pred = eec.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
