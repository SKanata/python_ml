#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class AdalineGD(object):

    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        
        # 重みベクトルを定義。切片 + 特徴量 
        X_new = self.add_ones(X)
        self.w_ = np.zeros(X_new.shape[1])

        self.cost = []
        
        for i in range(self.n_iter):
            output = self.net_input(X_new)
            errors = y - self.activation(output)
            delta_w = self.eta * np.dot(X_new.T, errors)
            cost = 0.5 * (errors ** 2).sum()
            self.cost.append(cost)
        return self


    def net_input(self, x):
        return np.dot(x, self.w_)

    def activation(self, x):
        return x
    
    def predict(self, x, has_instercept=False):
        if has_instercept:
            x_new = x
        else:
            x_new = self.add_ones(x)
        return np.where(self.net_input(x_new) >= 0.0, 1, -1)

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
