# encoding:utf-8
from copy import copy
from operator import attrgetter
import numpy as np
import pandas as pd
from minepy import MINE
from passenger_identify.feature_selection.Fitness import Data, Test_Data
from passenger_identify.feature_selection.Partical import Particle
from passenger_identify.base import reduce_mem_usage, read_csv, datapath, drop_features, discrete_list, tmppath, \
    Box_Cox, train_drop_features, getTrainTest, minmax_target, target, minmax
from imblearn.combine import SMOTEENN, SMOTETomek


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
        return 15

    def RWS(self, mic_feature_list, size):
        # 权重与对应的选择长度
        constant = np.array([j for j in range(1, 9)]) / 8
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
                                         cost=obj.getTrainAccuracy(features=solution))
        self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))
        return

    def back(self):
        print("撤回原来的操作，更新走向下一个区块，返还之前的迭代次数,重新进行count次数计算")
        if self.choose == 7:
            self.choose = -1
            return
        else:
            self.choose = self.choose + 1
        self.particles = self.old_particles
        self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))
        self.re_size_ent()
        return


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
    x_train, x_test, y_train, y_test = getTrainTest(X_train, Y_train)

    for seed in range(5):
        obj = Data(x_train, y_train)
        obj2 = Test_Data(x_train, x_test, y_train, y_test)
        np.random.seed(seed)
        pso = PSO(iterations=100, obj=obj, beta=0.2, alpha=0.4)
        pso.run()
        feature_subset = pso.best.getPBest()
        print("得到的特征子集序列为", feature_subset)
        print('得到的特征子集为', np.array(feature_list)[feature_subset])
        print('特征子集长度为', len(feature_subset))
        obj2.getTestAccuracy(feature_subset)
