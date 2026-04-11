# -*- coding: utf-8 -*-

import numpy as np


class Parameter:
    def __init__(self, init_params):
        if not isinstance(init_params, dict):
            raise TypeError("params must be dict.")
        for i in init_params:
            if not isinstance(init_params[i], np.ndarray):
                raise TypeError("params dict must have np.ndarray.")

        self.params = init_params

    # パラメータが全部0のものを作るメソッド
    def zeros(self):
        zero_params = {}
        for i in self.params:
            zero_params[i] = np.zeros(self[i].shape)
        return parameter(zero_params)

    def _arithmetic(self, other, func):
        res = self.zeros()
        if isinstance(other, self.__class__):
            for i in self.params:
                func(self[i], other[i], out=res[i])
        else:
            for i in self.params:
                func(self[i], other, out=res[i])
        return res

    def map(self, npfunc):
        res = self.zeros()
        for i in self.params:
            npfunc(self[i], out=res[i])
        return res

    def __add__(self, other):
        return self._arithmetic(other, np.add)

    def __mul__(self, other):
        return self._arithmetic(other, np.multiply)

    def __truediv__(self, other):
        return self._arithmetic(other, np.true_divide)

    def __sub__(self, other):
        return self._arithmetic(other, np.subtract)

    def __pow__(self, other):
        return self._arithmetic(other, np.power)

    def __abs__(self):
        return self.map(np.fabs)

    # 2つのパラメータを比べて, 大きい方を残す  "a | b" みたいに使える
    def __or__(self, other):
        if not isinstance(other, np.ndarray):
            TypeError("ndarray only.")
        res = self.zeros()
        for i in self.params:
            np.max(np.stack((self[i], other[i])), axis=0, out=res[i])
        return res

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value
