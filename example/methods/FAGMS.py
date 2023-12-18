#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/12/15 11:47
# @Author  : liuxin22
# @File    : FAGMS.py
# @Software: PyCharm
from tqdm import *
import numpy as np
import mmh3
from collections import defaultdict

class FAGMS:
    def __init__(self, k, m, datastream):
        self.k = k
        self.m = m
        self.datastream = datastream

        self.hash_seed = 1000
        self.sketch = np.zeros((self.k, self.m))
        self.n = len(datastream)

    def true_frequency(self):
        tr_fre = defaultdict(int)
        tr_fre.clear()
        for i in range(self.n):
            tr_fre[self.datastream[i]] += 1
        return tr_fre

    def _insert(self, item):
        for i in range(self.k):
            index = mmh3.hash(str(item), self.hash_seed + i, signed=False) % self.m
            g = 2 * (mmh3.hash(str(item), self.hash_seed + i + self.k, signed=False) % 2) - 1
            self.sketch[i][index] += g

    def insert_all(self):
        for i in trange(len(self.datastream), desc="Inserting items to sketch"):
            self._insert(self.datastream[i])
        return self.sketch

