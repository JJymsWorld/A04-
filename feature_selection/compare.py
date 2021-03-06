# 特征选择对比还有一些其他比较经典的方法，具体如下：
# utf-8
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from passenger_identify.base import reduce_mem_usage, read_csv, datapath, drop_features, tmppath, \
    Box_Cox, train_drop_features, getTrainTest, minmax_target


def evaluation_tree(x_train, x_test, y_train, y_test):
    print('进入评估函数')
    np.random.seed(0)
    from sklearn.tree import DecisionTreeClassifier
    dTree = DecisionTreeClassifier()
    dTree.fit(x_train, y_train)
    y_pred = dTree.predict(x_test)
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    print("balanced_accuracy_score=", balanced_accuracy_score(y_pred=y_pred, y_true=y_test))
    print("f1=", f1_score(y_pred=y_pred, y_true=y_test))
    print("precision_score=", precision_score(y_pred=y_pred, y_true=y_test))
    print("recall_score=", recall_score(y_pred=y_pred, y_true=y_test))
    print("auc=", roc_auc_score(y_true=y_test, y_score=y_pred))
    return


train = reduce_mem_usage(read_csv(tmppath + 'sub/combine_feature_train.csv'))
test = reduce_mem_usage(read_csv(tmppath + 'sub/combine_feature_test.csv'))

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

del train
# 标注：我已经把之前特征选择的结果全部剔除这一部分的特征选择了
x_train, x_test, y_train, y_test = getTrainTest(X_train, Y_train)  # 线下验证，80%训练集，20%验证集

count_list = [50, 100, 150, 250]
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(x_train, y_train)

model = SelectFromModel(clf, prefit=True)
X_train_1 = model.transform(x_train)
X_test_1 = model.transform(x_test)
print('训练数据特征筛选维度后', X_test_1.shape)
evaluation_tree(X_train_1, X_test_1, y_train, y_test)

# 基于树模型的特征选择法
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

gbd = SelectFromModel(DecisionTreeClassifier())
X_train_1 = gbd.fit_transform(x_train, y_train)
X_test_1 = gbd.transform(x_test)
print('Select from model')
print(X_train_1.shape)
evaluation_tree(X_train_1, X_test_1, y_train, y_test)

for count in count_list:
    from minepy import MINE

    print("#######################################" + str(count) + "#########################################")


    def mic(x, y):
        m = MINE()
        m.compute_score(x, y)
        return (m.mic(), 0.5)


    # skb = SelectKBest(
    #     lambda X, Y: np.array(list(map(lambda x: mic(x, Y), X.T))).T[0],
    #     k=count)
    # X_train_1 = skb.fit_transform(x_train, y_train)
    # X_test_1 = skb.transform(x_test)
    # print('SELECT K BEST')
    # evaluation_tree(X_train_1, X_test_1, y_train, y_test)
    #
    # from sklearn.decomposition import PCA
    #
    # pca = PCA(n_components=count)
    # X_train_1 = pca.fit_transform(x_train)
    # X_test_1 = pca.transform(x_test)
    # print('pca                         ')
    # evaluation_tree(X_train_1, X_test_1, y_train, y_test)

    # RBF递归消除特征法

    rbf = RFE(estimator=DecisionTreeClassifier(),
              n_features_to_select=count)
    X_train_1 = rbf.fit_transform(x_train, y_train)
    X_test_1 = rbf.transform(x_test)
    print('RBF                          ')
    evaluation_tree(X_train_1, X_test_1, y_train, y_test)
