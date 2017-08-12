#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class Perceptron(object):

    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        
        # 重みベクトルを定義。切片 + 特徴量 
        X_new = self.add_ones(X)
        self.w_ = np.zeros(X_new.shape[1])

        # 各イテレーションで何サンプル誤分類したか
        self.errors = []
        
        for i in range(self.n_iter):
            error = 0
            for xi, target in zip(X_new, y):
                predict_y = self.predict(xi)
                update = self.eta * (target - predict_y)
                self.w_ = self.w_ + update * xi
                # int(True) => 1, int(False) => 0
                error += int(update != 0.0)
            self.errors.append(error)

    def net_input(self, x):
        return np.dot(self.w_, x)

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    def add_ones(self, x, how='column'):
        if how == 'column':
            x_new = np.ones((x.shape[0], x.shape[1] + 1))
            x_new[:, 1:] = x
        elif how == 'row':
            x_new = np.ones((x.shape[0] + 1 , x.shape[1]))
            x_new[1:, :] = x
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return x_new
