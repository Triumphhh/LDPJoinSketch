#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 14:22
# @Author  : liuxin22
# @File    : KRR.py
# @Software: PyCharm
import numpy as np
import math
from tqdm import *
from collections import defaultdict

class KRR:
    def __init__(self, epsilon, datastream):
        self.epsilon = epsilon
        self.datastream = datastream

        self.n = len(datastream)
        self.unique_set, self.true_freq = np.unique(self.datastream, return_counts=True)
        self.GRR_k = len(self.unique_set)
        self.prob_p = math.pow(math.e, self.epsilon) / (self.GRR_k + math.pow(math.e, self.epsilon) - 1)
        self.prob_q = 1 / (self.GRR_k + math.pow(math.e, self.epsilon) - 1)

    def get_frequency(self):
        fre = defaultdict(int)
        fre.clear()
        for i in range(self.n):
            fre[self.datastream[i]] += 1
        return fre

    def _perturb(self, item, index):
        if np.random.random() > self.prob_p:
            while True:
                t = np.random.choice(self.unique_set)
                if t != item:
                    self.datastream[index] = t
                    break

    def perturbing_all(self):
        for i in trange(self.n, desc="Perturbing items in datastream"):
            self._perturb(self.datastream[i], i)

    def calibrating(self, priv_freq):
        priv_freq = {key: (value - self.n * self.prob_q) / (self.prob_p - self.prob_q)
                     for key, value in priv_freq.items()}
        return priv_freq

