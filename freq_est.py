#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/11/04 10:35
# @Author  : liuxin22
# @File    : freq_est.py
# @Software: PyCharm

from tqdm import *
import numpy as np
import time
import random
import csv
import mmh3
import math
from scipy.linalg import hadamard
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""

Description: Frequency estimation program via LDPJoinSketch.

Usage: python freq_est.py.

"""

def data_gen(file):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_vector = [int(row[0]) for row in reader]
    return data_vector

class FreqEst:
    def __init__(self, k, m, epsilon, datastream):
        """
        Parameters
        ----------
        k: the number of hash_funcs.
        m: the domain size of hash_funcs.
        epsilon: privacy budget.
        """
        self.k = k
        self.m = m
        self.epsilon = epsilon

        self.probability = 1 / (math.pow(math.e, self.epsilon) + 1)
        self.c = (math.pow(math.e, self.epsilon) + 1) / (math.pow(math.e, self.epsilon) - 1)
        self.had = hadamard(self.m)
        self.sketch = np.zeros((self.k, self.m))

        self.hash_seed = 1000
        self.datastream = datastream
        # self.datastream = np.concatenate(([1]*100, [2]*1500, [3]*1750, [4]*3500, [5]*50, [6]*150, [7]*1000,
        #                                   [8]*1500, [9]*1000, [10]*100))
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


if __name__ == '__main__':
    print(f"*------ Frequency Estimation via LDPJoinSketch ------*")
    hash_k = 18
    hash_m = 1024
    eps = 10
    filename = 'data/twitter/twitter.csv'
    data = data_gen(filename)
    testcycles = 3
    MSE = 0
    st_time = time.time()
    for tc in range(testcycles):
        print(f"Processing test round {tc+1}:")
        frequency = FreqEst(hash_k, hash_m, eps, data)  # parameters: k, m, epsilon
        frequency.insert_all()
        print(f"Calibrating and Estimating...")
        frequency.calibrating()
        r = frequency.frequency_est_all()
        m = frequency.calculate_mse(r)
        print(f"Processing round {tc+1} done!\n")
        MSE += m
    MSE /= testcycles
    ed_time = time.time()
    print(f"Mse of estimated frequency: {MSE}.")
    print(f"Running time: {ed_time-st_time} s.")
