#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import unittest
from perceptron import Perceptron

class TestPerceptronn(unittest.TestCase):
    """test class of perceptron.py
    """
    
    def test_tashizan(self):
        """test method for perceptron
        """
        # 初期化
        expected = np.array([-0.4, -0.68, 1.82])
        
        df = pd.read_csv('../tests/data/iris.data', header=None)
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        X = df.iloc[0:100, [0, 2]].values

        # テスト実行
        pc = Perceptron()
        pc.fit(X, y)
        actual = pc.w_

        # Assert
        np.testing.assert_array_almost_equal(expected, actual, 2)
        
        # 終了処理
        
