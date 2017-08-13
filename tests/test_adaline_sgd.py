#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import unittest
from adaline_sgd import AdalineSGD

class TestAdalineSGD(unittest.TestCase):
    """test class of adaline_sgd.py
    """
    
    def test_tashizan(self):
        """test method for adaline_sgd
        """
        # 初期化
        expected = np.array([0.01081067, -0.13961527, 1.07501121])
        
        df = pd.read_csv('../tests/data/iris.data', header=None)
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        X = df.iloc[0:100, [0, 2]].values
        X_std = np.copy(X)
        X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

        # テスト実行
        ad = AdalineSGD(n_iter=15, eta=0.01)
        ad.fit(X_std, y)
        actual = ad.w_

        # Assert
        np.testing.assert_array_almost_equal(expected, actual, 2)
        
        # 終了処理
        
