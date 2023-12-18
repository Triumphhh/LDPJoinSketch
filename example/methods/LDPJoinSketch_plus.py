#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/11/18 14:22
# @Author  : liuxin22
# @File    : LDPJoinSketch+.py
# @Software: PyCharm
from tqdm import *
import numpy as np
from collections import defaultdict, Counter
import random
import mmh3
import math
from scipy.linalg import hadamard
from .frequency_est import FreqEst

class LDPJoinSketch_Plus:
    def __init__(self, k, m, mt, epsilon, r, theta, datastream):
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
        self.mt = mt
        self.epsilon = epsilon
        self.sample_rate = r
        self.theta = theta

        self.probability = 1 / (math.pow(math.e, self.epsilon) + 1)
        self.c = (math.pow(math.e, self.epsilon) + 1) / (math.pow(math.e, self.epsilon) - 1)
        self.had = hadamard(self.m)

        self.sketch_H = np.zeros((self.k, self.m))  # initialize k*m sketch for high_frequency items.
        self.sketch_L = np.zeros((self.k, self.m))  # for low_frequency items.

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

    def divide_clients(self):
        """
        Divide the datastream into three parts.
        Returns
        -------
        sp: sample data for calculating FI.
        rm1, rm2: two groups of remain data for FAP.
        """
        ns = int(self.n * self.sample_rate)   # phase1采样的用户数量
        sp = random.sample(self.datastream, ns)  # 根据采样数量对数据集进行均匀随机采样
        data_counter = Counter(self.datastream)
        sp_counter = Counter(sp)
        uniq_sp, uniq_fre = np.unique(sp, return_counts=True)  # 采样数据的真实频数向量
        rm = list((data_counter - sp_counter).elements())  # 再将剩余客户端数据分组用于高低频项估计，这里采样均分
        random.shuffle(rm)
        rm1 = rm[:len(rm)//2]
        rm2 = rm[len(rm)//2:]
        return sp, rm1, rm2

    def get_high_frequency_set(self, data):
        """
        Parameters
        ----------
        data: sampled data.
        Returns
        -------
        fi: estimated high-frequency item set.
        fi_freq: estimated high frequency.
        """
        freq_est = FreqEst(self.k, self.mt, self.epsilon, data)
        freq_est.insert_all()  # using LDPJoinSketch
        freq_est.calibrating()
        freq = freq_est.frequency_est_all()
        fi, fi_freq = freq_est.find_freq_itemset(freq, self.theta)
        if len(fi) == 0:
            print(f"the high frequency threshold theta={self.theta} is too large that no frequent items selected!")
        return fi, fi_freq

    def _inserting(self, item):
        """
        For target item, inserting to LDPJoinSketch.
        Parameters
        ----------
        item: item to be inserted.

        Returns
        -------
        j, l, y: index (j,l) and perturbed value y.
        """
        j: int = np.random.randint(0, self.k)
        l: int = np.random.randint(0, self.m)
        v = [0] * self.m
        index_j = mmh3.hash(str(item), self.hash_seed + j, signed=False) % self.m  # k different hash functions.
        v[index_j] = 2 * (mmh3.hash(str(item), self.hash_seed + j + self.k, signed=False) % 2) - 1
        w = self.had[:, index_j] * v[index_j]
        b = random.choices([-1, 1], k=1, weights=[self.probability, 1 - self.probability])  # LDP perturbation.
        y = b[0] * w[l]
        return j, l, y

    def _uniform_inserting(self):
        """
        For non-target item, uniformly and randomly inserting to the sketch.
        Returns
        -------
        j, l, y: random index (j, l) and perturbed value y.
        """
        j: int = np.random.randint(0, self.k)
        l: int = np.random.randint(0, self.m)
        r: int = np.random.randint(0, self.m)
        v = [0] * self.m
        v[r] = 1  # set to 1
        w = self.had[:, r]
        b = random.choices([-1, 1], k=1, weights=[self.probability, 1 - self.probability])  # LDP perturbation
        y = b[0] * w[l]
        return j, l, y

    def _inserting_all(self, item, is_target, mode):
        """
        Parameters
        ----------
        item: item to be inserting.
        is_target: whether the item is a target one in  current mode.
        mode: {0,1} for high-frequent join and low-frequent join.

        """
        if is_target:
            j, l, y = self._inserting(item)
        else:
            j, l, y = self._uniform_inserting()

        if mode == 1:
            self.sketch_H[j, l] += self.k * self.c * y
        else:
            self.sketch_L[j, l] += self.k * self.c * y

    def _frequency_aware_perturbing(self, item, freq_set, mode):
        """
        Parameters
        ----------
        item: item to be perturbed.
        freq_set: frequent item set.
        mode: {0,1} for high-frequent join and low-frequent join.
        """
        if mode == 1:
            if item in freq_set:
                is_target = True
            else:
                is_target = False
            self._inserting_all(item, is_target, mode)
        elif mode == 0:
            if item in freq_set:
                is_target = False
            else:
                is_target = True
            self._inserting_all(item, is_target, mode)

    def perturb(self, data, freq_set, mode):
        for i in trange(len(data), desc=f"Perturbing items for mode {mode}."):
            self._frequency_aware_perturbing(data[i], freq_set, mode)
        if mode == 1:
            return self.sketch_H
        else:
            return self.sketch_L

    @staticmethod
    def get_FI_intersection(fi0, hf0, fi1, hf1):
        """
        Parameters
        ----------
        fi0: frequent item set of SA.
        hf0: frequency vector of fi0.
        fi1: frequent item set of SB.
        hf1: frequency vector of fi1.

        Returns
        -------
        fi: intersection frequent item set of SA and SB.
        f0, f1: corresponding frequency vector of fi.
        """
        fi = list(set(fi0).intersection(set(fi1)))
        f0, f1 = [], []
        for item in fi:
            index0 = fi0.index(item)
            index1 = fi1.index(item)
            f0.append(hf0[index0])
            f1.append(hf1[index1])
        return fi, f0, f1

    @staticmethod
    def join_est(k, m, sketch_a, sketch_b, size_a, size_b, hf_a, hf_b, mode):
        """
        Algorithm 5: JoinEst
        Parameters
        ----------
        k: number of hash functions.
        m: domain size of hash functions.
        sketch_a: sketch from remained data groupA for HEst/LEst.
        sketch_b: sketch from remained data groupB for HEst/LEst.
        size_a: size of datastream A.
        size_b: size of datastream B.
        hf_a: sum of high frequency in SA.
        hf_b: sum of high frequency in SB.
        mode: {0,1} for high-frequent join and low-frequent join.

        Returns
        -------
        est: sub-join_size HEst or LEst.
        """
        if mode == 0:
            sketch_a -= hf_a / m
            sketch_b -= hf_b / m
        else:
            sketch_a -= (size_a - hf_a) / m
            sketch_b -= (size_b - hf_b) / m
        k_est = np.zeros(k)
        for i in range(k):
            k_est[i] = np.dot(sketch_a[i], sketch_b[i])
        est = np.median(k_est)
        return est

    @staticmethod
    def get_join_size(hest, lest, size_A, size_A0, size_A1, size_B, size_B0, size_B1):
        """
        Parameters
        ----------
        hest: sub-join_size of high frequency items in groupA0 and groupB0.
        lest: sub-join_size of low frequency items in groupA1 and groupB1.
        size_A: size of datastream A.
        size_A0: size of first remained data group of A for HEst.
        size_A1: size of second remained data group of A for LEst.
        size_B: size of datastream B.
        size_B0: size of first remained data group of B for HEst.
        size_B1: size of second remained data group of B for LEst.

        Returns
        -------
        re: final estimated join size of datastream A and B using LDPJoinSketch+.
        """
        re = ((size_A * size_B) / (size_A0 * size_B0)) * hest + \
             ((size_A * size_B) / (size_A1 * size_B1)) * lest
        return re
