# 定义全局函数便于直接调用处理
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from datetime import date
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from chinese_calendar import is_in_lieu, is_holiday, is_workday

############全局参数#################################
discrete_list = ['pax_name_passport', 'seg_route_from', 'seg_route_to', 'seg_flight', 'seg_cabin',
                 'seg_dep_time', 'gender', 'age', 'birth_date', 'residence_country', 'nation_name',
                 'city_name',
                 'province_name', 'marital_stat', 'ffp_nbr', 'member_level', 'often_city', 'enroll_chnl',
                 'pref_aircraft_m3_1', 'pref_aircraft_m3_2', 'pref_aircraft_m3_3', 'pref_aircraft_m3_4',
                 'pref_aircraft_m3_5', 'pref_aircraft_m6_1', 'pref_aircraft_m6_2', 'pref_aircraft_m6_3',
                 'pref_aircraft_m6_4', 'pref_aircraft_m6_5', 'pref_aircraft_y1_1', 'pref_aircraft_y1_2',
                 'pref_aircraft_y1_3', 'pref_aircraft_y1_4', 'pref_aircraft_y1_5', 'pref_aircraft_y2_1',
                 'pref_aircraft_y2_2', 'pref_aircraft_y2_3', 'pref_aircraft_y2_4', 'pref_aircraft_y2_5',
                 'pref_aircraft_y3_1', 'pref_aircraft_y3_2', 'pref_aircraft_y3_3', 'pref_aircraft_y3_4',
                 'pref_aircraft_y3_5', 'pref_orig_m3_1', 'pref_orig_m3_2', 'pref_orig_m3_3', 'pref_orig_m3_4',
                 'pref_orig_m3_5', 'pref_orig_m6_1', 'pref_orig_m6_2', 'pref_orig_m6_3', 'pref_orig_m6_4',
                 'pref_orig_m6_5', 'pref_orig_y1_1', 'pref_orig_y1_2', 'pref_orig_y1_3', 'pref_orig_y1_4',
                 'pref_orig_y1_5', 'pref_orig_y2_1', 'pref_orig_y2_2', 'pref_orig_y2_3', 'pref_orig_y2_4',
                 'pref_orig_y2_5', 'pref_orig_y3_1', 'pref_orig_y3_2', 'pref_orig_y3_3', 'pref_orig_y3_4',
                 'pref_orig_y3_5', 'pref_line_m3_1', 'pref_line_m3_2', 'pref_line_m3_3', 'pref_line_m3_4',
                 'pref_line_m3_5', 'pref_line_m6_1', 'pref_line_m6_2', 'pref_line_m6_3', 'pref_line_m6_4',
                 'pref_line_m6_5', 'pref_line_y1_1', 'pref_line_y1_2', 'pref_line_y1_3', 'pref_line_y1_4',
                 'pref_line_y1_5', 'pref_line_y2_1', 'pref_line_y2_2', 'pref_line_y2_3', 'pref_line_y2_4',
                 'pref_line_y2_5', 'pref_line_y3_1', 'pref_line_y3_2', 'pref_line_y3_3', 'pref_line_y3_4',
                 'pref_line_y3_5', 'pref_city_m3_1', 'pref_city_m3_2', 'pref_city_m3_3', 'pref_city_m3_4',
                 'pref_city_m3_5', 'pref_city_m6_1', 'pref_city_m6_2', 'pref_city_m6_3', 'pref_city_m6_4',
                 'pref_city_m6_5', 'pref_city_y1_1', 'pref_city_y1_2', 'pref_city_y1_3', 'pref_city_y1_4',
                 'pref_city_y1_5', 'pref_city_y2_1', 'pref_city_y2_2', 'pref_city_y2_3', 'pref_city_y2_4',
                 'pref_city_y2_5', 'pref_city_y3_1', 'pref_city_y3_2', 'pref_city_y3_3', 'pref_city_y3_4',
                 'pref_city_y3_5', 'recent_flt_day', 'pit_add_chnl_m3', 'pit_add_chnl_m6', 'pit_add_chnl_y1',
                 'pit_add_chnl_y2', 'pit_add_chnl_y3', 'pref_orig_city_m3', 'pref_orig_city_m6', 'pref_orig_city_y1',
                 'pref_orig_city_y2', 'pref_orig_city_y3', 'pref_dest_city_m3', 'pref_dest_city_m6',
                 'pref_dest_city_y1', 'pref_dest_city_y2', 'pref_dest_city_y3',
                 # 对月份，小时等新增为离散类型数据进行分析，这里一定要详细判断一下数据的分布情况
                 'pref_month_m3_1', 'pref_month_m3_2', 'pref_month_m3_3', 'pref_month_m3_4', 'pref_month_y3_5',
                 'pref_month_m3_5', 'pref_month_m6_1', 'pref_month_m6_2', 'pref_month_m6_3', 'pref_month_m6_4',
                 'pref_month_m6_5', 'pref_month_y1_1', 'pref_month_y1_2', 'pref_month_y1_3', 'pref_month_y1_4',
                 'pref_month_y1_5', 'pref_month_y2_1', 'pref_month_y2_2', 'pref_month_y2_3', 'pref_month_y2_4',
                 'pref_month_y2_5', 'pref_month_y3_1', 'pref_month_y3_2', 'pref_month_y3_3', 'pref_month_y3_4',
                 'seg_dep_time_month', 'seg_dep_time_year', 'seg_dep_time_hour', 'seg_dep_time_is_workday',
                 'seg_dep_time_is_holiday', 'seg_dep_time_is_in_lieu', 'recent_flt_day_month', 'recent_flt_day_year',
                 'recent_flt_day_hour', 'recent_flt_day_is_workday', 'recent_flt_day_is_holiday',
                 'recent_flt_day_is_in_lieu', 'has_ffp_nbr'
                 ]  # 离散型的特征
epsilon = 1e-5
func_dict = {'add': lambda x, y: x + y,
             "mins": lambda x, y: x - y,
             'div': lambda x, y: x / (y + epsilon),
             'multi': lambda x, y: x + y}
func_list = ['mean', 'median', 'min', 'max', 'std', 'var', 'mode']
drop_features = ['emd_lable', 'cabin_hf_cnt_m3', 'cabin_hf_cnt_m6', 'complain_valid_cnt_m3', 'complain_valid_cnt_m6',
                 'bag_cnt_m6',
                 'bag_cnt_y1', 'bag_cnt_y2', 'bag_cnt_y3', 'flt_cancel_cnt_y3', 'cabin_fall_cnt_y3', 'complain_cnt_m3',
                 'complain_cnt_m6', 'pit_out_amt',
                 'pit_add_non_amt_m3', 'pit_add_buy_amt_y2', 'pit_add_buy_amt_y3', 'pit_des_out_amt_m3',
                 'pit_des_out_amt_m6', 'pit_des_out_amt_y1', 'pit_add_non_cnt_m3', 'pit_add_buy_cnt_y2',
                 'pit_add_buy_cnt_y3', 'pit_des_mall_cnt_m6', 'pit_des_mall_cnt_y1', 'pit_des_out_cnt_m6',
                 'pit_des_out_cnt_y1', 'pit_ech_avg_amt_m3',
                 'pit_out_avg_amt_m3', 'pit_out_avg_amt_m6']  # 观察分布后预筛选的特征,都是连续性的
train_drop_features = ['pax_name_passport', 'seg_dep_time', 'recent_flt_day',
                       'birth_date', 'ffp_nbr']  # 训练过程中需要剔除的变量，如用户ID、日期等明显的无关数据！
############目录定义#################################
datapath = 'D:/outsourcing/data/'
featurepath = 'D:/outsourcing/feature/'
resultpath = 'D:/outsourcing/result/'
tmppath = 'D:/outsourcing/tmp/'


###############函数定义################################
# reduce memory
def read_csv(file_name, num_rows=None):
    if num_rows is None:
        return pd.read_csv(file_name)
    return pd.read_csv(file_name, nrows=num_rows)


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def evaluation(x_train, x_test, y_train, y_test, str):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    # knn = KNeighborsClassifier(n_neighbors=5, p=1)
    bayes = GaussianNB()
    tree = DecisionTreeClassifier()
    svm = SVC()
    LR = LogisticRegression()
    model_list = [bayes, tree, svm, LR]
    model_name = ['bayes', 'tree', 'svm', 'LR']
    f = open('../result/' + str + '.txt', mode='x')
    for i in range(len(model_list)):
        np.random.seed(0)
        model = model_list[i]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred2 = model.predict(x_train)
        print("###################" + model_name[i] + "#########################", file=f)
        from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, \
            accuracy_score, roc_auc_score, confusion_matrix
        print("balanced_accuracy_score=", balanced_accuracy_score(y_pred=y_pred, y_true=y_test),
              balanced_accuracy_score(y_pred=y_pred2, y_true=y_train), file=f)
        print("f1=", f1_score(y_pred=y_pred, y_true=y_test), f1_score(y_pred=y_pred2, y_true=y_train), file=f)
        print("precision_score=", precision_score(y_pred=y_pred, y_true=y_test),
              precision_score(y_pred=y_pred2, y_true=y_train), file=f)
        print("recall_score=", recall_score(y_pred=y_pred, y_true=y_test), recall_score(y_pred=y_pred2, y_true=y_train),
              file=f)
        print("accuracy=", accuracy_score(y_pred=y_pred, y_true=y_test), accuracy_score(y_pred=y_pred2, y_true=y_train),
              file=f)
        print("auc=", roc_auc_score(y_true=y_test, y_score=y_pred), roc_auc_score(y_true=y_train, y_score=y_pred2),
              file=f)
        print("#####混淆矩阵#########", file=f)
        print(confusion_matrix(y_true=y_test, y_pred=y_pred), confusion_matrix(y_true=y_train, y_pred=y_pred2), file=f)
    return


def date_transfer(data):
    # 缺失值用-1表示即可
    data['seg_dep_time_month'] = [datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').month for i in
                                  range(len(data))]
    data['seg_dep_time_year'] = [datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').year for i in
                                 range(len(data))]
    data['seg_dep_time_hour'] = [datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').hour for i in
                                 range(len(data))]
    data['seg_dep_time_is_workday'] = [
        is_workday(datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M'))
        for i in
        range(len(data))]
    data['seg_dep_time_is_holiday'] = [
        is_holiday(datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M'))
        for i in
        range(len(data))]
    data['seg_dep_time_is_in_lieu'] = [
        is_in_lieu(datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M'))
        for i in
        range(len(data))]
    data['recent_flt_day_month'] = [
        datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M').month if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['recent_flt_day_year'] = [
        datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M').year if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['recent_flt_day_hour'] = [
        datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M').hour if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]

    data['recent_flt_day_is_workday'] = [
        is_workday(datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M')) if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['recent_flt_day_is_holiday'] = [
        is_holiday(datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M')) if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['recent_flt_day_is_in_lieu'] = [
        is_in_lieu(datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M')) if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['birth_interval_day'] = [
        datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').day - datetime.strptime(data['birth_date'][i],
                                                                                             '%Y/%m/%d %H:%M').day
        if data['birth_date'][i] != '0'
        else -999
        for i in
        range(len(data))]
    data['last_flt_day'] = [
        datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').day - datetime.strptime(data['recent_flt_day'][i],
                                                                                             '%Y/%m/%d %H:%M').day
        if data['recent_flt_day'][i] != '0'
        else -999
        for i in
        range(len(data))]
    data['has_ffp_nbr'] = [0 if data['ffp_nbr'][i] == 0
                           else 1
                           for i in range(len(data))]
    data = data.replace({True: 1, False: 0})
    return data


def Box_Cox():
    # 1.MinMax归一化操作
    # 2.进行Box_Cox转换
    # 对训练集与测试集合并处理即可！
    train = reduce_mem_usage(read_csv(tmppath + "date_train.csv")).drop(drop_features, axis=1)
    test = reduce_mem_usage(read_csv(tmppath + "date_test.csv")).drop(drop_features, axis=1)
    continue_list = list(set(train.columns.tolist()) - set(discrete_list) - {'emd_lable2'})
    from sklearn.preprocessing import MinMaxScaler
    data_all = pd.concat([train, test])
    minmax = MinMaxScaler()
    data_all[continue_list] = minmax.fit_transform(data_all[continue_list])
    # 对所有的连续型特征进行转换
    for column in continue_list:
        data_all[column], lmbda = boxcox(data_all[column] + 1)
    # 连续数据转换完成后保存文件,已经去除了大量特征
    train = data_all[0:23432].to_csv(tmppath + 'Box_train.csv', index=False)
    test = data_all[23432:].to_csv(tmppath + 'Box_test.csv', index=False)
    return train, test


def getTrainTest(X, Y):
    global x_train, x_test, y_train, y_test
    # 会员编号等，等下仔细去查看所有取值数量超过100的特征
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in kfold.split(X, Y):
        x_train = X.loc[train_index]
        x_test = X.loc[test_index]
        y_train = Y.loc[train_index]
        y_test = Y.loc[test_index]
        break
    return x_train, x_test, y_train, y_test


def minmax_target(X_train, X_test, Y_train, continue_list, discrete_list):
    import category_encoders as ce
    from sklearn.preprocessing import MinMaxScaler

    encoder = ce.LeaveOneOutEncoder(cols=discrete_list, drop_invariant=False).fit(X_train, Y_train)
    minmax = MinMaxScaler()
    train = pd.concat([X_train, X_test])
    minmax.fit(train[continue_list])

    X_train = encoder.transform(X_train)  # 基于训练集得到编码器
    X_test = encoder.transform(X_test)
    X_train[continue_list] = minmax.transform(X_train[continue_list])
    X_test[continue_list] = minmax.transform(X_test[continue_list])
    return X_train, X_test


def target(X_train, X_test, Y_train, discrete_list):
    import category_encoders as ce

    encoder = ce.LeaveOneOutEncoder(cols=discrete_list, drop_invariant=False).fit(X_train, Y_train)

    X_train = encoder.transform(X_train)  # 基于训练集得到编码器
    X_test = encoder.transform(X_test)
    return X_train, X_test


def minmax(X_train, X_test, continue_list):
    from sklearn.preprocessing import MinMaxScaler
    minmax = MinMaxScaler()
    train = pd.concat([X_train, X_test])
    minmax.fit(train[continue_list])
    X_train[continue_list] = minmax.transform(X_train[continue_list])
    X_test[continue_list] = minmax.transform(X_test[continue_list])
    return X_train, X_test


# 组合交叉特征，提升模型的线性组合能力（数值特征的变化），特征1+特征2这种类型
def auto_feature_make(train, test, continue_list):
    for col_i in continue_list:
        for col_j in continue_list:
            for func_name, func in func_dict.items():
                for data in [train, test]:
                    func_features = func(data[col_i], data[col_j])
                    col_func_features = '_'.join([col_i, func_name, col_j])
                    data[col_func_features] = func_features
                    print(col_func_features)
    return train, test


# 数值特征与连续特征的组合，这种方法用到中位数，均值等，需要对所有数据操作
def combine_feature(train, test, discrete_list, continue_list):
    data = pd.concat([train, test])
    for col_i in discrete_list:
        for col_j in continue_list:
            for func in func_list:
                col_func_features = '_'.join([col_j, func, col_i])
                data[col_func_features] = data[col_j].groupby(data[col_i]).transform(func)
                print(col_func_features)
    for col_i in discrete_list:
        for col_j in discrete_list:
            col_func_features = '_'.join([col_j, 'freq', col_i])
            data[col_func_features] = data[col_j].groupby(data[col_i]).transform('freq')
            print(col_func_features)
    data[0:23432].to_csv(tmppath + 'combine_feature_' + 'train.csv', index=False)
    data[23432:].to_csv(tmppath + 'combine_feature_' + 'test.csv+', index=False)
    return data[0:23432], data[23432:]

# train = reduce_mem_usage(read_csv(tmppath + "date_train.csv")).drop(drop_features, axis=1).drop(['emd_lable2'],
#                                                                                                 axis=1).drop(
#     train_drop_features, axis=1)
# test = reduce_mem_usage(read_csv(tmppath + "date_test.csv")).drop(drop_features, axis=1).drop(['emd_lable2'],
#                                                                                               axis=1).drop(
#     train_drop_features, axis=1)
# discrete_list = list(set(discrete_list) - set(train_drop_features))  # 训练中需要剔除的特征都是离散型的特征
# continue_list = list(set(train.columns.tolist()) - set(discrete_list))
# train, test = auto_feature_make(train, test, continue_list)
# train.to_csv(tmppath + 'feature_make_' + 'train.csv', index=False)
# test.to_csv(tmppath + 'feature_make_' + 'test.csv', index=False)
#
# combine_feature(train, test, discrete_list, continue_list)

# train = date_transfer(train)
# test = date_transfer(test)
#
# train.to_csv(tmppath + 'date_train.csv', index=False)
# test.to_csv(tmppath + 'date_test.csv', index=False)
