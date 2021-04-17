# encoding:utf-8
from collections import Counter
from copy import copy
from operator import attrgetter
import numpy as np
import pandas as pd
from minepy import MINE
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from passenger_identify.feature_selection.Fitness import Data, Test_Data
from passenger_identify.feature_selection.Partical import Particle
from passenger_identify.base import reduce_mem_usage, read_csv, datapath, drop_features, discrete_list, tmppath, \
    Box_Cox, train_drop_features, getTrainTest, minmax_target, target, minmax, getTrainTest_np, feature_subset_list
from imblearn.combine import SMOTEENN, SMOTETomek


def getTopN(feature_subset_list, N):
    # 遍历10个列表，得出出现次数topN的几个特征
    topN = {}
    for subset in feature_subset_list:
        subset = list(set(subset))
        for feature in subset:
            # feature = list(set(feature))
            if feature in topN:
                topN[feature] += 1
            else:
                topN[feature] = 1
    feature_count_list = []
    for feature, freq in topN.items():
        feature_count_list.append((freq, feature))
    feature_count_list.sort(reverse=True)
    for freq, word in feature_count_list[:N]:
        print(word, freq)
    return np.array(feature_count_list)[:N, 1]


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


def swap(solution, purpose, arr):
    '''定义交换操作SO'''
    temp = solution[arr[0]]
    solution[arr[0]] = purpose[arr[1]]
    purpose[arr[1]] = temp


def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def generateSS(solution, purpose):
    '''核心学习过程：生成学习队列，现采用集合论方式生成，对重复情况有一定取舍'''
    ss = []
    solutionSet = set(solution)
    purposeSet = set(purpose)
    # 求对应的交集
    remove = solutionSet & purposeSet
    # 去除交集，获取差异特征序列
    solutionSet = solutionSet - remove
    purposeSet = purposeSet - remove
    # 针对差异特征较短的集合，进行替换产生so[a,b]
    length = len(solutionSet) if (len(solutionSet) < len(purposeSet)) else len(purposeSet)
    if length == 0: return ss
    solutionSet = list(solutionSet)
    purposeSet = list(purposeSet)
    for i in range(length):
        a = solution.index(solutionSet[i])
        b = purpose.index(purposeSet[i])
        ss.append([a, b])
    return ss


class PSO:
    def __init__(self, iterations, obj, alpha, beta):
        self.iterations = iterations  # max of iterations
        self.particles = []  # list of particles
        self.obj = obj
        self.alpha = alpha
        self.beta = beta
        self.all_feature_size = obj.getDimension()
        self.size_population = 50  # size population
        self.mic_feature_list = self.feature_filter()
        self.choose = 0
        self.solutions = self.RWS(self.mic_feature_list, self.get_subset())
        for solution in self.solutions:
            # creates a new particle
            particle = Particle(solution=solution, cost=obj.getTrainAccuracy(features=solution))
            # add the particle
            self.particles.append(particle)
        # update gbest
        self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))
        self.best = copy(self.gbest)

    def get_subset(self):
        return 25

    def RWS(self, mic_feature_list, size):
        # 权重与对应的选择长度
        constant = np.array([j for j in range(1, 5)]) / 4
        print('第{0}个区间进行更新'.format(self.choose + 1), constant)
        sub_mic_feature_list = mic_feature_list[:int(constant[self.choose] * len(mic_feature_list))]
        P_MIC = sub_mic_feature_list[:, 0]
        P_MIC_feature = sub_mic_feature_list[:, 1].astype(int)
        P_MIC = P_MIC / np.sum(P_MIC)
        for i in range(len(P_MIC)):
            if i != 0:
                P_MIC[i] = P_MIC[i] + P_MIC[i - 1]
        solutions = []
        for par in range(self.size_population):
            solution = []
            for i in range(size):
                rand = np.random.rand()
                for j in range(len(P_MIC)):
                    if rand < P_MIC[j]:
                        solution.append(P_MIC_feature[j])
                        break
            solutions.append(solution)
        return solutions

    def feature_filter(self):
        """基于MIC相关系数，进行特征排序筛选， 产生一个排序后的字典{MIC,feature index}"""
        feature_mic = {}
        mine = MINE(alpha=0.6, c=15)
        for i in range(self.all_feature_size):
            mine.compute_score(self.obj._X_train[:, i], self.obj._Y_train)
            feature_mic[i] = mine.mic()
        feature_mic_list = []
        for feature, mic in feature_mic.items():
            if mic > 0:
                feature_mic_list.append((mic, feature))
        feature_mic_list.sort(reverse=True)

        # feature_mic_list_1 = pd.DataFrame(feature_mic_list.copy())
        # feature_mic_list_1[1] = np.array(feature_list)[np.array(feature_mic_list)[:, 1].astype(int)]
        # feature_mic_list_1.to_csv("D:/outsourcing/result/feature_mic_2.csv", index=False)
        return np.array(feature_mic_list)

    def showParticle(self, t):
        print('迭代次数为{2}   gbest  length={0}   fitness={1}'.format(len(self.gbest.getPBest()), self.gbest.getCostPBest(),
                                                                  t))

    def run(self):
        count = 0
        t = 0
        while t < self.iterations:
            # update each particle's solution
            for particle in self.particles:
                solution_gbest = self.gbest.getPBest()[:]  # gets solution of the gbest
                solution_pbest = particle.getPBest()[:]  # copy of the pbest solution
                solution_particle = particle.getCurrentSolution()[:]

                ss_pbest = generateSS(solution_particle, solution_pbest)
                for so in ss_pbest:
                    alpha = np.random.random()
                    if alpha < self.alpha:
                        swap(solution_particle, solution_pbest, so)
                ss_gbest = generateSS(solution_particle, solution_gbest)
                for so in ss_gbest:
                    beta = np.random.random()
                    if beta < self.beta:
                        swap(solution_particle, solution_gbest, so)
                # update current_solution
                particle.setCurrentSolution(solution_particle)

            # update pbest,gbest
            for particle in self.particles:
                if particle.getPBest() != particle.getCurrentSolution():
                    particle.setCostCurrentSolution(
                        self.obj.getTrainAccuracy(particle.getCurrentSolution()))
                    if particle.getCostCurrentSolution() > particle.getCostPBest():
                        particle.setPBest(particle.getCurrentSolution())
                        particle.setCostPBest(particle.getCostCurrentSolution())
            self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))

            # resize the feature_subset
            if self.gbest.getCostPBest() > self.best.getCostPBest():
                self.best = copy(self.gbest)
                count = 0
                print("best更新成功！！！！！！！！！！！！！！！")
            else:
                count += 1
            if count == 4:
                self.re_size_ent()
            elif count == 10:  # 如果下次更新仍然不能有效跳出局部最优解，则陷入停滞直至循环次数达到100
                t = t - 10 + 4
                self.back()
                if self.choose == -1:
                    break
            self.showParticle(t)
            t = t + 1

    def re_size_ent(self):
        '''对于不是gbest的所有粒子重置，gbest保留历史信息'''
        size = int(2 * calc_ent(self.obj._Y_train) / self.gbest.getCostPBest() + 5)
        self.old_particles = copy(self.particles)
        self.solutions = self.RWS(self.mic_feature_list, size)
        for i in range(len(self.particles)):
            solution = self.particles[i].getCurrentSolution() + self.solutions[i][:size]
            self.particles[i] = Particle(solution=solution,
                                         cost=self.obj.getTrainAccuracy(features=solution))
        self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))
        return

    def back(self):
        print("撤回原来的操作，更新走向下一个区块，返还之前的迭代次数,重新进行count次数计算")
        if self.choose == 3:
            self.choose = -1
            return
        else:
            self.choose = self.choose + 1
        self.particles = self.old_particles
        self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))
        self.re_size_ent()
        return


def testParameters(x_train, x_test, y_train, y_test, clf, file):
    ## 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)
    ## 利用f1评估模型效果
    print('The f1 of the train is:', metrics.f1_score(y_train, train_predict), 'The auc of the train is',
          metrics.roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1]), file=file)
    print('The f1 of the test is:', metrics.f1_score(y_test, test_predict), 'The auc of the test is',
          metrics.roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]), file=file)


def GridSearchSelf(x_train, x_test, y_train, y_test, file):
    import warnings
    warnings.filterwarnings("ignore")
    from lightgbm.sklearn import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV

    print('设置迭代次数', file=file)
    model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                           learning_rate=0.1)
    parameters = {'n_estimators': range(1000, 2500, 50)}
    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    clf = GridSearchCV(model, parameters, cv=kfold, verbose=0)
    clf = clf.fit(x_train, y_train)
    testParameters(x_train, x_test, y_train, y_test, clf, file)
    n_estimators = clf.best_params_['n_estimators']
    print('n_estimators is ', n_estimators)

    print("设置max_depth和num_leaves", file=file)
    model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                           learning_rate=0.1,
                           n_estimators=n_estimators)
    parameters = {'max_depth': range(6, 10, 1), 'num_leaves': range(10, 150, 10)}
    clf = GridSearchCV(model, parameters, cv=kfold, verbose=0)
    clf = clf.fit(x_train, y_train)
    testParameters(x_train, x_test, y_train, y_test, clf, file)
    max_depth = clf.best_params_['max_depth'],
    num_leaves = clf.best_params_['num_leaves']
    print('max_depth is {0},num_leaves is {1}'.format(max_depth, num_leaves), file=file)

    print("确定min_data_in_leaf和max_bin_in", file=file)

    model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                           learning_rate=0.1,
                           n_estimators=n_estimators,
                           max_depth=max_depth,
                           num_leaves=num_leaves)
    parameters = {'max_bin': range(300, 500, 20), 'min_data_in_leaf': range(10, 20, 10)}

    clf = GridSearchCV(model, parameters, cv=kfold, verbose=0)
    clf = clf.fit(x_train, y_train)
    testParameters(x_train, x_test, y_train, y_test, clf, file)
    max_bin = clf.best_params_['max_bin']
    min_data_in_leaf = clf.best_params_['min_data_in_leaf']
    print('max_bin is {0},min_data_in_deaf is {1}'.format(max_bin, min_data_in_leaf), file=file)

    print("确定feature_fraction、bagging_fraction、bagging_freq", file=file)
    model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                           learning_rate=0.1,
                           n_estimators=n_estimators,
                           max_depth=max_depth,
                           num_leaves=num_leaves,
                           max_bin=max_bin,
                           min_data_in_leaf=min_data_in_leaf)
    parameters = {'feature_fraction': [0.7, 0.8, 0.9, 1.0],
                  'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
                  'bagging_freq': range(0, 20, 2)}

    clf = GridSearchCV(model, parameters, cv=kfold, verbose=0)
    clf = clf.fit(x_train, y_train)
    testParameters(x_train, x_test, y_train, y_test, clf, file)

    feature_fraction = clf.best_params_['feature_fraction']
    bagging_fraction = clf.best_params_['bagging_fraction']
    bagging_freq = clf.best_params_['bagging_freq']
    print(
        'feature_fraction is {0},bagging_fraction is {1},bagging_freq is {2}'.format(feature_fraction, bagging_fraction,
                                                                                     bagging_freq), file=file)

    print('确定lambda_l1和lambda_l2', file=file)
    model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                           learning_rate=0.1,
                           n_estimators=n_estimators,
                           max_depth=max_depth,
                           num_leaves=num_leaves,
                           max_bin=max_bin,
                           min_data_in_leaf=min_data_in_leaf,
                           feature_fraction=feature_fraction,
                           bagging_fraction=bagging_fraction,
                           bagging_freq=bagging_freq)
    parameters = {'lambda_l1': [1e-5, 1e-3, 1e-1, 0.0, 0.1],
                  'lambda_l2': [1e-5, 1e-3, 1e-1, 0.0, 0.1]
                  }

    clf = GridSearchCV(model, parameters, cv=kfold, verbose=0)
    clf = clf.fit(x_train, y_train)
    testParameters(x_train, x_test, y_train, y_test, clf, file)
    lambda_l1 = clf.best_params_['lambda_l1']
    lambda_l2 = clf.best_params_['lambda_l2']
    print('lamda1 is {0},lamda2 is {1}'.format(lambda_l1, lambda_l2), file=file)

    clf = GridSearchCV(model, parameters, cv=kfold, verbose=0)
    clf = clf.fit(x_train, y_train)
    testParameters(x_train, x_test, y_train, y_test, clf, file)
    min_split_gain = 0.0
    print('min_split_gain为 ', min_split_gain, file=file)
    print('调低学习率，增加迭代次数并验证模型效果', file=file)

    learing_rates = [0.01, 0.025, 0.05, 0.075, 0.1]
    for learing_rate in learing_rates:
        model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                               learning_rate=learing_rate,
                               n_estimators=n_estimators,
                               max_depth=max_depth,
                               num_leaves=num_leaves,
                               max_bin=max_bin,
                               min_data_in_leaf=min_data_in_leaf,
                               feature_fraction=feature_fraction,
                               bagging_fraction=bagging_fraction,
                               bagging_freq=bagging_freq,
                               lambda_l1=lambda_l1,
                               lambda_l2=lambda_l2,
                               min_split_gain=min_split_gain)
        print('参数为', '学习率', learing_rate, '迭代次数', n_estimators, '深度与叶子数量', max_depth, num_leaves,
              'max_bin与min_data_in_leaf', max_bin, min_data_in_leaf, 'feature_fraction,bagging_fraction,bagging_freq',
              feature_fraction, bagging_fraction, bagging_freq, 'lamda1,lamda2', lambda_l1, lambda_l2, 'min_split',
              min_split_gain, file=file)
        model.fit(x_train, y_train)
        testParameters(x_train, x_test, y_train, y_test, model, file)
        model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                               learning_rate=learing_rate,
                               n_estimators=n_estimators * 2,
                               max_depth=max_depth,
                               num_leaves=num_leaves,
                               max_bin=max_bin,
                               min_data_in_leaf=min_data_in_leaf,
                               feature_fraction=feature_fraction,
                               bagging_fraction=bagging_fraction,
                               bagging_freq=bagging_freq,
                               lambda_l1=lambda_l1,
                               lambda_l2=lambda_l2,
                               min_split_gain=min_split_gain)
        print('参数为', '学习率', learing_rate, '迭代次数', n_estimators * 2, '深度与叶子数量', max_depth, num_leaves,
              'max_bin与min_data_in_leaf', max_bin, min_data_in_leaf, 'feature_fraction,bagging_fraction,bagging_freq',
              feature_fraction, bagging_fraction, bagging_freq, 'lamda1,lamda2', lambda_l1, lambda_l2, 'min_split',
              min_split_gain, file=file)
        model.fit(x_train, y_train)
        testParameters(x_train, x_test, y_train, y_test, model, file)
        model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                               learning_rate=learing_rate,
                               n_estimators=n_estimators * 3,
                               max_depth=max_depth,
                               num_leaves=num_leaves,
                               max_bin=max_bin,
                               min_data_in_leaf=min_data_in_leaf,
                               feature_fraction=feature_fraction,
                               bagging_fraction=bagging_fraction,
                               bagging_freq=bagging_freq,
                               lambda_l1=lambda_l1,
                               lambda_l2=lambda_l2,
                               min_split_gain=min_split_gain)
        print('参数为', '学习率', learing_rate, '迭代次数', n_estimators * 3, '深度与叶子数量', max_depth, num_leaves,
              'max_bin与min_data_in_leaf', max_bin, min_data_in_leaf, 'feature_fraction,bagging_fraction,bagging_freq',
              feature_fraction, bagging_fraction, bagging_freq, 'lamda1,lamda2', lambda_l1, lambda_l2, 'min_split',
              min_split_gain, file=file)
        model.fit(x_train, y_train)
        testParameters(x_train, x_test, y_train, y_test, model, file)
        model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                               learning_rate=learing_rate,
                               n_estimators=n_estimators * 5,
                               max_depth=max_depth,
                               num_leaves=num_leaves,
                               max_bin=max_bin,
                               min_data_in_leaf=min_data_in_leaf,
                               feature_fraction=feature_fraction,
                               bagging_fraction=bagging_fraction,
                               bagging_freq=bagging_freq,
                               lambda_l1=lambda_l1,
                               lambda_l2=lambda_l2,
                               min_split_gain=min_split_gain)
        print('参数为', '学习率', learing_rate, '迭代次数', n_estimators * 5, '深度与叶子数量', max_depth, num_leaves,
              'max_bin与min_data_in_leaf', max_bin, min_data_in_leaf, 'feature_fraction,bagging_fraction,bagging_freq',
              feature_fraction, bagging_fraction, bagging_freq, 'lamda1,lamda2', lambda_l1, lambda_l2, 'min_split',
              min_split_gain, file=file)
        model.fit(x_train, y_train)
        testParameters(x_train, x_test, y_train, y_test, model, file)
        model = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1', is_unbalance=True,
                               learning_rate=learing_rate,
                               n_estimators=n_estimators * 10,
                               max_depth=max_depth,
                               num_leaves=num_leaves,
                               max_bin=max_bin,
                               min_data_in_leaf=min_data_in_leaf,
                               feature_fraction=feature_fraction,
                               bagging_fraction=bagging_fraction,
                               bagging_freq=bagging_freq,
                               lambda_l1=lambda_l1,
                               lambda_l2=lambda_l2,
                               min_split_gain=min_split_gain)
        print('参数为', '学习率', learing_rate, '迭代次数', n_estimators * 10, '深度与叶子数量', max_depth, num_leaves,
              'max_bin与min_data_in_leaf', max_bin, min_data_in_leaf, 'feature_fraction,bagging_fraction,bagging_freq',
              feature_fraction, bagging_fraction, bagging_freq, 'lamda1,lamda2', lambda_l1, lambda_l2, 'min_split',
              min_split_gain, file=file)
        model.fit(x_train, y_train)
        testParameters(x_train, x_test, y_train, y_test, model, file)
    return


def testBaseline(x_train, x_test, y_train, y_test, file):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf.fit(x_train, y_train)
    ## 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)
    ## 利用f1评估模型效果
    print('The f1 of the train is:', metrics.f1_score(y_train, train_predict), 'The auc of the train is',
          metrics.roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1]), file=file)
    print('The f1 of the test is:', metrics.f1_score(y_test, test_predict), 'The auc of the test is',
          metrics.roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]), file=file)
    return 0


if __name__ == "__main__":
    train = reduce_mem_usage(read_csv(tmppath + "BOX_train.csv"))
    test = reduce_mem_usage(read_csv(tmppath + 'BOX_test.csv'))

    X_train = train.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)  # 去除部分取值过多的离散型特征
    X_test = test.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)
    Y_train = train['emd_lable2'].astype(int)

    discrete_list = list(set(discrete_list) - set(train_drop_features))  # 训练中需要剔除的特征都是离散型的特征
    feature_list = X_train.columns.tolist()
    continue_list = list(set(feature_list) - set(discrete_list))

    # 数据归一化与编码
    X_train, X_test = minmax_target(X_train, X_test, Y_train, continue_list, discrete_list)
    # # 模型交叉验证！
    del train, test, X_test
    # file = open('is_unbalance.txt', 'a')
    # kfold = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    # for k, (train_index, test_index) in enumerate(kfold.split(X_train, Y_train)):
    #     feature_subset = list(set(feature_subset_list[k]))
    #     x_train = X_train.loc[train_index][feature_subset]
    #     x_test = X_train.loc[test_index][feature_subset]
    #     y_train = Y_train.loc[train_index]
    #     y_test = Y_train.loc[test_index]
    #     # 对获得的9折训练集交叉验证训练模型，对10%的验证集计算结果，并对10折情况进行平均查看f1值
    #     print('对第{0}折进行自动调参'.format(k), file=file)
    #     GridSearchSelf(x_train, x_test, y_train, y_test, file)
    # file.close()
    file = open('对比随机森林.txt', 'a')
    kfold = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    for k, (train_index, test_index) in enumerate(kfold.split(X_train, Y_train)):
        feature_subset = list(set(feature_subset_list[k]))
        x_train = X_train.loc[train_index][feature_subset]
        x_test = X_train.loc[test_index][feature_subset]
        y_train = Y_train.loc[train_index]
        y_test = Y_train.loc[test_index]
        # 对获得的9折训练集交叉验证训练模型，对10%的验证集计算结果，并对10折情况进行平均查看f1值
        print('对第{0}折进行验证'.format(k), file=file)
        testBaseline(x_train, x_test, y_train, y_test, file)
    file.close()

    # # getTopN(feature_subset_list, 70)  # 前68个，其实相差不大

    # feature_subset_list = []
    # kfold = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    # for k, (train_index, test_index) in enumerate(kfold.split(X_train, Y_train)):  # 验证函数改为3折交叉，如果效果更好则重新设置特征子集
    #     # 十折交叉得到10个特征子集查看一下结果如何？
    #     if k == 1 or k == 3 or k == 8:
    #         print('第{0}折的尝试'.format(k))
    #         x_train = X_train.loc[train_index]
    #         x_test = X_train.loc[test_index]
    #         y_train = Y_train.loc[train_index]
    #         y_test = Y_train.loc[test_index]
    #
    #         obj = Data(x_train, y_train)
    #         pso = PSO(iterations=100, obj=obj, beta=0.2, alpha=0.4)
    #         pso.run()
    #         feature_subset = pso.best.getPBest()
    #         print("得到的特征子集序列为", feature_subset)
    #         print('得到的特征子集为', np.array(feature_list)[feature_subset])
    #         print('特征子集长度为', len(feature_subset))

    # train = reduce_mem_usage(read_csv(tmppath + 'sub_train_all.csv'))
    # # test = reduce_mem_usage(read_csv(tmppath + 'sub_test_all.csv'))
    #
    # X_train = train.drop(['emd_lable2'], axis=1)  # 去除部分取值过多的离散型特征
    # Y_train = train['emd_lable2'].astype(int)
    #
    # discrete_list = ['seg_flight', 'seg_cabin', 'pref_orig_m6_2', 'pref_line_y1_2',
    #                  'pref_line_y1_3', 'pref_line_y2_2', 'pref_line_y2_3', 'pref_line_y3_3'
    #     , 'pref_line_y3_4', 'pref_line_y3_5', 'pref_aircraft_y3_3', 'pref_city_y1_2',
    #                  'pref_city_y3_4', 'pref_dest_city_m6', 'pref_dest_city_y3'
    #     , 'pref_month_y3_1', 'seg_dep_time_month']  # 训练中需要剔除的特征都是离散型的特征
    # feature_list = X_train.columns.tolist()
    # continue_list = list(set(feature_list) - set(discrete_list))
    #
    # # X_train, test = minmax_target(X_train, test, Y_train, continue_list, discrete_list)  # 离散值编码与连续特征归一化
    #
    # del train
    # # 标注：我已经把之前特征选择的结果全部剔除这一部分的特征选择了
    # x_train, x_test, y_train, y_test = getTrainTest(X_train, Y_train)  # 线下验证，80%训练集，20%验证集
