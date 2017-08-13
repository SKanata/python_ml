#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import seed

class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        if random_state:
            seed(random_state)
        
    def fit(self, X, y):
        
        # 初期化
        X_new = self.add_ones(X)
        self._initialize_weights(X_new.shape[1])
        self.cost = []

        # トレーニング回数分トレーニングデータを反復
        for i in range(self.n_iter):
            # トレーニングデータの順番を入れ替える？
            if self.shuffle:
                X_new, y = self._shuffle(X_new, y)

            cost_in_i = []
            for xi, target in zip(X_new, y):
                cost = self._update_weights(xi, target)
                cost_in_i.append(cost)
            avg_cost = sum(cost_in_i) / len(cost_in_i)
            self.cost.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        X_new = self.add_ones(X)
        if not self.w_initialized:
            self._initialize_weights(X_new.shape[1])
        # 複数サンプルが与えられた場合
        if y.shape[0] > 1:
            for xi, target in zip(X_new, y):
                self._update_weights(xi, y)
        # 1つのサンプルが与えられた場合
        else:
            self._update_weights(X_new, y)
        return self

    def _update_weights(self, xi, target):
        error = target - self.activation(xi)
        delta_w = self.eta * np.dot(xi, error)
        self.w_ = self.w_ + delta_w
        cost = 0.5 * error ** 2
        return cost
        
    def _shuffle(self, X, y):
        """ トレーニングデータをシャッフル """
        r = np.random.permutation(len(y))
        # >>> np.random.permutation(10)
        # array([4, 8, 1, 2, 0, 7, 6, 9, 3, 5])
        return X[r], y[r]
        
    def _initialize_weights(self, m):
        """ 重みをゼロに初期化 """
        self.w_ = np.zeros(m)
        self.w_initialized = True

    def net_input(self, X):
        return np.dot(X, self.w_)

    def activation(self, X):
        return self.net_input(X)
    
    def predict(self, X, has_instercept=False):
        if has_instercept:
            X_new = X
        else:
            X_new = self.add_ones(X)
        return np.where(self.activation(X_new) >= 0.0, 1, -1)

    def add_ones(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1 , X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new
