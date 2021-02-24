from sklearn import metrics
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict, cross_val_score, KFold, \
    StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import metrics


def getTrainTest(_X_train, _Y_train):
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    train_test = []
    for train_index, test_index in kfold.split(_X_train, _Y_train):
        train_test.append([train_index, test_index])
    return train_test


class Data:
    _X_train = None
    _Y_train = None

    def __init__(self, x_train, y_train):
        # Extract the no of features
        self.noOfFeatures = x_train.shape[1]
        self._X_train = x_train.to_numpy()
        self._Y_train = y_train.to_numpy()
        self.train_test = getTrainTest(self._X_train, self._Y_train)

    def getTrainAccuracy(self, features):
        acc = []
        for train_index, test_index in self.train_test:
            dTree = DecisionTreeClassifier()
            x_train = self._X_train[train_index]
            x_test = self._X_train[test_index]
            dTree.fit(x_train[:, features], self._Y_train[train_index])
            y_pred = dTree.predict(x_test[:, features])
            acc.append(metrics.roc_auc_score(y_true=self._Y_train[test_index], y_score=y_pred))
        return np.mean(acc)

    def getDimension(self):
        return self.noOfFeatures


class Test_Data:
    _X_test = None
    _Y_test = None

    def __init__(self, x_train, x_test, y_train, y_test):
        # Extract the no of features
        self.noOfFeatures = x_train.shape[1]

        self._X_train = x_train.to_numpy()
        self._X_test = x_test.to_numpy()
        self._Y_train = y_train.to_numpy()
        self._Y_test = y_test.to_numpy()

    def getTestAccuracy(self, features):
        np.random.seed(0)
        knn = DecisionTreeClassifier()
        knn.fit(self._X_train[:, features], self._Y_train)
        # 测试集预测得到的结果存储在列表中统一计算准确度
        y_pred = knn.predict(self._X_test[:, features])
        print("balanced_accuracy_score=", metrics.balanced_accuracy_score(y_pred=y_pred, y_true=self._Y_test))
        print("f1=", metrics.f1_score(y_pred=y_pred, y_true=self._Y_test))
        print("precision_score=", metrics.precision_score(y_pred=y_pred, y_true=self._Y_test))
        print("recall_score=", metrics.recall_score(y_pred=y_pred, y_true=self._Y_test))
        print('auc=', metrics.roc_auc_score(y_score=y_pred, y_true=self._Y_test))

    def getDimension(self):
        return self.noOfFeatures
