#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/11/12 10:18
# @Author  : liuxin22
# @File    : MainProj.py
# @Software: PyCharm

from tqdm import *
import numpy as np
import random
import mmh3
import math
from scipy.linalg import hadamard


class FreqEst:
    def __init__(self, k, m, epsilon, datastream):
        """
        Parameters
        ----------
        k: the number of hash_funcs.
        m: the domain size of hash_funcs.
        epsilon: privacy budget, should be positive.
        datastream: dataset for estimation.
        """
        self.k = k
        self.m = m
        self.epsilon = epsilon

        self.probability = 1 / (math.pow(math.e, self.epsilon) + 1)
        self.c = (math.pow(math.e, self.epsilon) + 1) / (math.pow(math.e, self.epsilon) - 1)
        self.had = hadamard(self.m)
        self.sketch = np.zeros((self.k, self.m))  # initialize k*m sketch

        self.hash_seed = 1000
        self.datastream = datastream
        # self.datastream = np.concatenate(([1]*10000, [2]*15000, [3]*750, [4]*350, [5]*50, [6]*150, [7]*100,
        #                                   [8]*150, [9]*100, [10]*100))
        self.unique_set, self.true_freq = np.unique(self.datastream, return_counts=True)

    def _insert(self, item):
        """
        Parameters
        ----------
        item: item being inserted to sketch.

        Returns
        -------
        (j,l): the index that item being inserted to.
        y: the one-bit sent to the server.
        """
        j: int = np.random.randint(0, self.k)
        l: int = np.random.randint(0, self.m)
        v = [0] * self.m
        index_j = mmh3.hash(str(item), self.hash_seed + j, signed=False) % self.m          # k different hash functions.
        v[index_j] = 2 * (mmh3.hash(str(item), self.hash_seed + j + self.k, signed=False) % 2) - 1
        w = self.had[:, index_j] * v[index_j]
        b = random.choices([-1, 1], k=1, weights=[self.probability, 1 - self.probability])           # LDP perturbation.
        y = b[0] * w[l]
        return j, l, y

    def _frequency_est(self, sketch, value):
        """
        Parameters
        ----------
        sketch: the aggregated and calibrated sketch.
        value: value of unique item.

        Returns
        -------
        est_freq: estimated frequency of unique item.
        """
        sums = 0
        for i in range(self.k):
            index_i = mmh3.hash(str(value), self.hash_seed + i, signed=False) % self.m
            xi = 2 * (mmh3.hash(str(value), self.hash_seed + i + self.k, signed=False) % 2) - 1
            sums += sketch[i, index_i] * xi
        est_freq = 1/self.k * sums
        return est_freq

    def insert_all(self):
        for i in trange(len(self.datastream), desc='Inserting items to sketch'):
            j, l, y = self._insert(self.datastream[i])
            self.sketch[j, l] += self.k * self.c * y

    def calibrating(self):
        self.sketch = np.matmul(self.sketch, np.transpose(self.had))

    def frequency_est_all(self):
        est_freq = []  # use Python list to record estimated frequency.
        for uniq_item in self.unique_set:
            ef = self._frequency_est(self.sketch, uniq_item)
            est_freq.append(ef)
        return est_freq

    def calculate_mse(self, ef):
        """
        Parameters
        ----------
        ef: estimated frequency vector.

        Returns
        -------
        mse: mean square error result.
        """
        ef = np.array(ef)
        mse = np.mean((ef - self.true_freq) ** 2)
        return mse

    def find_freq_itemset(self, est_f, thet):
        """
        Parameters
        ----------
        est_f: estimated frequency vector.
        thet: high-frequency items threshold theta.

        Returns
        -------
        fi, fi_freq: frequent items set and corresponding frequency.
        """
        thresh = len(self.datastream) * thet
        _fi = []
        _fi_freq = []
        for i in range(len(est_f)):
            if est_f[i] > thresh:
                _fi_freq.append(est_f[i])
                _fi.append(self.unique_set[i])
        return _fi, _fi_freq
