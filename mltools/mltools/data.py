# -*- coding:utf-8 -*-

import numpy as np


class Data:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("only numpy.ndarray")
        self.data = data
        self._minibatch_idx = np.array([])

    def minibatch(self, batchsize, return_idx=False):
        if len(self._minibatch_idx) == 0:
            self._minibatch_idx = np.random.choice(
                np.arange(0, len(self.data)), size=len(self.data), replace=False
            )
        idx, self._minibatch_idx = np.split(self._minibatch_idx, [batchsize])
        if return_idx:
            return self.data[idx], idx
        else:
            return self.data[idx]

    def normalize(self, mu=None, sigma=None, clip_size=4):
        if mu is None:
            mu = np.mean(self.data, axis=0)
        if sigma is None:
            sigma = np.std(self.data, axis=0, ddof=1)
        self.data = (self.data - mu) / (sigma + 1e-8)
        np.clip(self.data, -clip_size, clip_size, out=self.data)
        return mu, sigma

    def __len__(self):
        return len(self.data)


class CategoricalData:
    def __init__(self, data, answer):
        if not isinstance(data, np.ndarray) or not isinstance(answer, np.ndarray):
            raise TypeError("only numpy.ndarray")
        self.data = Data(data)
        self.answer = Data(answer)

    def minibatch(self, batchsize):
        batch_data, idx = self.data.minibatch(batchsize, return_idx=True)
        return batch_data, self.answer.data[idx]

    def normalize(self, **kwargs):
        mu, sigma = self.data.normalize(**kwargs)
        return mu, sigma

    def __len__(self):
        return len(self.data)
