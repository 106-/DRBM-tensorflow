# -*- coding:utf-8 -*-

import numpy as np


class SGD:
    def __init__(self, learning_rate=None):
        if learning_rate is None:
            raise ValueError("learning_rate is required.")
        self._learning_rate = learning_rate

    def update(self, grad):
        diff = grad.zeros()
        for i in grad.params:
            diff[i] = grad[i] * self._learning_rate[i]
        return diff


class Momentum:
    def __init__(self, learning_rate=None, alpha=None):
        if learning_rate is None or alpha is None:
            raise ValueError("learning_rate and alpha is required.")
        self._learning_rate = learning_rate
        self._alpha = alpha
        self._old_grad = None

    def update(self, grad):
        if self._old_grad is None:
            self._old_grad = grad.zeros()
        diff = grad.zeros()
        for i in grad.params:
            diff[i] = (
                grad[i] * self._learning_rate[i] + self._old_grad[i] * self._alpha[i]
            )
        return diff


class Adamax:
    def __init__(self, alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._t = 0

    def update(self, grad):
        if self._t == 0:
            self._moment = grad.zeros()
            self._norm = grad.zeros()

        self._t += 1
        self._moment = self._moment * self._beta1 + grad * (1 - self._beta1)
        # "|" は2つのパラメータの大きい方だけを抽出する
        self._norm = (self._norm * self._beta2) | abs(grad)
        diff = (
            self._moment
            / (self._norm + self._epsilon)
            * (self._alpha / (1 - np.power(self._beta1, self._t)))
        )
        return diff


class Adam:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._t = 0

    def update(self, grad):
        if self._t == 0:
            self._m = grad.zeros()
            self._v = grad.zeros()

        self._t += 1
        self._m = self._m * self._beta1 + grad * (1 - self._beta1)
        self._v = self._v * self._beta2 + grad**2 * (1 - self._beta2)
        m_hat = self._m / (1 - np.power(self._beta1, self._t))
        v_hat = self._v / (1 - np.power(self._beta2, self._t))
        diff = m_hat / (v_hat.map(np.sqrt) + self._epsilon) * self._alpha
        return diff
