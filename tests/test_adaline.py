#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import unittest
from adaline import AdalineGD

class TestAdalineGD(unittest.TestCase):
    """test class of adaline.py
    """
    
    def test_tashizan(self):
        """test method for adaline
        """
        # 初期化
        expected = np.array([1.59872116e-16, -1.26256159e-01, 1.10479201e+00])
        
        df = pd.read_csv('../tests/data/iris.data', header=None)
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        X = df.iloc[0:100, [0, 2]].values
        X_std = np.copy(X)
        X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

        # テスト実行
        ad = AdalineGD(n_iter=15, eta=0.01)
        ad.fit(X_std, y)
        actual = ad.w_

        # Assert
        np.testing.assert_array_almost_equal(expected, actual, 2)
        
        # 終了処理
        
