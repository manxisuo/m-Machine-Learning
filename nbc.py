# -*- coding: utf8 -*-

from math import pi as PI, e as E, exp, sqrt, pow
from functools import reduce

# 计算序列的平均值和方差
def mean_and_variance(values):
    n = len(values)
    mean = reduce(lambda x, y : x + y, values) / n
    variance = reduce(lambda x, y : x + pow(y - mean, 2), values, 0) / (n - 1)
    return (mean, variance)

# 高斯分布
class _GaussianDistr:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def probability(self, v):
        return exp(-pow(v - self.mean, 2) / 2 / self.variance) / sqrt(2 * PI * self.variance)

# 朴素贝叶斯分类
class NBC:
    def __init__(self, features, categories):
    
        # 特征列表
        self.features = features

        # 类别列表
        self.categories = categories

        # 每个类别对应的每个特征的训练数据
        self.training_map = {}

        # 每个类别对应的每个特征的(正态)分布参数
        self.model_map = {}

        # 每个类别的训练数据的总数
        self.categories_amount = {}

        # 训练数据的总数
        self.data_amount = 0

        for c in categories:
            self.training_map[c] = {}
            self.model_map[c] = {}
            self.categories_amount[c] = 0
            for f in features:
                self.training_map[c][f] = []

    # 添加一条训练数据
    def add(self, data, category):
        self.categories_amount[category] += 1
        self.data_amount += 1
        for i, value in enumerate(data):
            feature = self.features[i]
            self.training_map[category][feature].append(value)

    # 开始训练
    def train(self):
        for c in self.categories:
            for f in self.features:
                values = self.training_map[c][f]
                self.model_map[c][f] = _GaussianDistr(*mean_and_variance(values))
    
    # 分类
    def classification(self, data):
        r = (0, None)
        for c in self.categories:
            p = self.categories_amount[c] / self.data_amount
            for i, value in enumerate(data):
                f = self.features[i]
                p *= self.model_map[c][f].probability(value)
            if p > r[0]:
                r = (p, c)
        return r[1]

# 测试
if __name__ == '__main__':
    nbc = NBC(['height', 'weight', 'foot size'], ['male', 'female'])

    nbc.add([6, 180, 12], 'male')
    nbc.add([5.92, 190, 11], 'male')
    nbc.add([5.58, 170, 12], 'male')
    nbc.add([5.92, 165, 10], 'male')
    nbc.add([5, 100, 6], 'female')
    nbc.add([5.5, 150, 8], 'female')
    nbc.add([5.42, 130, 7], 'female')
    nbc.add([5.75, 150, 9], 'female')

    nbc.train()

    print(nbc.classification([6, 130, 8])) # female
    print(nbc.classification([6.2, 170, 13])) # male
