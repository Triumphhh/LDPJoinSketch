#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 10:04
# @Author  : liuxin22
# @File    : AppleHCMS.py
# @Software: PyCharm

from collections import defaultdict
from tqdm import *
import numpy as np
import random
import mmh3
import math
from scipy.linalg import hadamard

class AppleHCMS:
    def __init__(self, k, m, epsilon, datastream):
        """
        Parameters
        ----------
        k: the number of hash_funcs.
        m: the domain size of hash_funcs.
        epsilon: privacy budget.
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
        self.unique_set, self.true_freq = np.unique(self.datastream, return_counts=True)
        self.n = len(self.datastream)

    def true_frequency(self):  # type defaultdict to compute join size
        tr_fre = defaultdict(int)
        tr_fre.clear()
        for i in range(self.n):
            tr_fre[self.datastream[i]] += 1
        return tr_fre

    # client-side
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
        index_j = mmh3.hash(str(item), self.hash_seed + j, signed=False) % self.m
        v[index_j] = 1
        w = self.had[:,index_j]
        b = random.choices([-1, 1], k=1, weights=[self.probability, 1 - self.probability])
        y = b[0] * w[l]
        return j, l, y

    # server-side
    def insert_all(self):
        for i in trange(len(self.datastream), desc='Inserting items to sketch'):
            j, l, y = self._insert(self.datastream[i])
            self.sketch[j, l] += self.k * self.c * y

    def calibrating(self):
        self.sketch = np.matmul(self.sketch, np.transpose(self.had))

    def _freq_est(self, sketch, value):
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
            index = mmh3.hash(str(value), self.hash_seed + i, signed=False) % self.m
            sums += sketch[i][index]
        est_freq = (self.m / (self.m - 1)) * ((1 / self.k) * sums - (self.n / self.m))
        return est_freq

    def freq_est_all(self):
        f_res = defaultdict(float)
        for item in self.unique_set:
            f_res[item] = self._freq_est(self.sketch, item)
        return f_res

    def calculate_mse(self, ef):
        mse = 0
        for i in range(len(self.unique_set)):
            mse += (self.true_freq[i] - ef[self.unique_set[i]]) ** 2
        mse /= len(self.unique_set)
        return mse
