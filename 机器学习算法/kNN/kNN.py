import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:
    def __init__(self,k):
       '初始化kNN分类器'
       '断言'
       assert k>=1 ,"kmust be valid"
       self.k = k;
       self._X_train = None
       self._y_train = None

    def fit(self,X_train,y_train):
        '根据训练数据集 X_train 和 y_train 寻来你kNN分类器'
        assert X_train.shape[0] == y_train.shape[0],\
        "the size of X_train must be equal to size of y_train"
        assert self.k <= X_train.shape[0],\
        "the size of X_train must be least k"


        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self,X_predict):
        '给定带预测的数据集 X_predict, 返回表示 X_predict 的结果向量'
        assert self._X_train is not None and self._y_train is not None,\
        "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1],\
        "the feature number of X_predict must be equalto X_train"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):
        '给定单个待测数据 x ,返回 x_predict的预测结果值'
        assert x.shape[0] == self._X_train.shape[1], \
        "the feature number of x must be equal to X_train"
        distances = [sqrt(np.sum((x_train - x)**2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "kNN(k=%d)"%self.k