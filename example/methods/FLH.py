#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/12/15 15:01
# @Author  : liuxin22
# @File    : FLH.py
# @Software: PyCharm

import numpy as np
import time
import math
import mmh3
import csv
import xxhash
import random
from tqdm import *
from collections import defaultdict

class FLH:
    def __init__(self, epsilon, datastream):
        """
        Parameters
        ----------
        epsilon: privacy budget, should be positive.
        datastream: dataset for estimation.
        """
        self.epsilon = epsilon
        self.datastream = datastream

        self.n = len(datastream)
        self.unique_set, self.true_freq = np.unique(self.datastream, return_counts=True)

        self.g = round(math.exp(self.epsilon) + 1)   # Optimal Local Hashing setting.
        self.prob_p = math.exp(self.epsilon) / (self.g + math.exp(self.epsilon) - 1)
        self.prob_q = 1.0 / (self.g + math.exp(self.epsilon) - 1)
        self.k = 18  # used in FLH. Larger k results in a more accurate oracle at the expense of computation time.

        self.hash_matrix = np.empty((self.k, len(self.unique_set)))  # constructing pre-computed hash_matrix.
        for i in range(self.k):
            for j in range(len(self.unique_set)):
                # self.hash_matrix[i][j] = xxhash.xxh32(str(self.unique_set[j]), seed=i).intdigest() % self.g
                self.hash_matrix[i][j] = mmh3.hash(str(self.unique_set[j]), seed=i) % self.g
        self.hash_counter = np.zeros((self.k, self.g))

    def true_frequency(self):
        tr_fre = defaultdict(int)
        tr_fre.clear()
        for i in range(self.n):
            tr_fre[self.datastream[i]] += 1
        return tr_fre

    def _perturb(self, item):
        """
        Parameters
        ----------
        item: item being perturbed.
        """
        hash_seed = np.random.randint(0, self.k)
        x = mmh3.hash(str(item), seed=hash_seed) % self.g
        y = x
        p_sample = np.random.random_sample()
        if p_sample > self.prob_p - self.prob_q:
            # perturb
            y = np.random.randint(0, self.g)
        self.hash_counter[hash_seed][y] += 1

    def perturbing_all(self):
        for i in trange(self.n, desc="Perturbing items in datastream"):
            self._perturb(self.datastream[i])

    def _func(self, x):
        """
        Parameters
        ----------
        x: a column vector in hash_matrix.

        Returns
        -------
        sums: the sum of mapped result of item in hash_counter.
        """
        sums = 0
        for index, val in enumerate(x):
            sums += self.hash_counter[index, int(val)]
        return sums

    def aggregate(self):
        priv_freq = list(np.apply_along_axis(self._func, 0, self.hash_matrix))
        return priv_freq

    def calibrating(self, priv_freq):
        calibrated_freq = defaultdict(float)
        a = 1.0 * self.g / (self.prob_p * self.g - 1)
        b = 1.0 * self.n / (self.prob_p * self.g - 1)
        for i in range(len(priv_freq)):
            cal_p = a * priv_freq[i] - b
            calibrated_freq[self.unique_set[i]] = cal_p
        return calibrated_freq

