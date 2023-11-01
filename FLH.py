#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/10/24 11:12
# @Author  : liuxin22
# @File    : FLH.py
# @Software: PyCharm
import random
import statistics
from tqdm import *
from pure_ldp.frequency_oracles import *
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import csv
import xxhash
from collections import Counter
from collections import defaultdict


def read_csv_to_vector(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_vector = [int(row[0]) for row in reader]
    return data_vector

def get_frequencies(datastream, unique_values):
    freq_dict = defaultdict(int)
    for val in datastream:
        freq_dict[val] += 1
    return [freq_dict[val] for val in unique_values]

def calibrating(priv_freq, prob_p, g, n):
    calibrated_freq = []
    a = 1.0 * g / (prob_p * g - 1)
    b = 1.0 * n / (prob_p * g - 1)
    for i in range(len(priv_freq)):
        cal_p = a * priv_freq[i] - b
        calibrated_freq.append(cal_p)

    return calibrated_freq

def func1(x):
    summ = 0
    for index, val in enumerate(x):
        summ += hash_counts1[index, int(val)]
    return summ

def func2(x):
    summ = 0
    for index, val in enumerate(x):
        summ += hash_counts2[index, int(val)]
    return summ


# -------- choose dataset ------
# filename = 'data/zipf/zipf_10M_1_1.csv'
# filename = 'data/gaussian/gaussian.csv'
filename = 'data/tpc_ds/tpc_ds.csv'
# filename = 'data/twitter/twitter.csv'
# filename = 'data/facebook/facebook.csv'

data = read_csv_to_vector(filename)
# data = list(np.concatenate(([1]*1000,[2]*800,[3]*600,[4]*800,[5]*600,[6]*400,[7]*200,[8]*500,[9]*100,[10]*1000)))

datastream_1 = data
datastream_2 = data
print("Generate data streams done!")

unique_values = np.unique(np.concatenate((datastream_1, datastream_2)))
domain = len(unique_values)

st_time = time.time()
freq_vector_1 = get_frequencies(datastream_1, unique_values)  # true frequencies. the same item order as unique_values.
freq_vector_2 = get_frequencies(datastream_2, unique_values)
Join_ground_truth = sum(x * y for x, y in zip(freq_vector_1, freq_vector_2))

epsilon = 4  # 设置隐私预算
g = round(math.exp(epsilon) + 1)   # Optimal Local Hashing setting.
p = math.exp(epsilon) / (g + math.exp(epsilon) - 1)
q = 1.0 / (g + math.exp(epsilon) - 1)
k = 18  # used in FLH. Larger k results in a more accurate oracle at the expense of computation time.
print("*----------Experiment Information----------*")
print(f"Using dataset:{filename} with size={len(data)} and domain={domain}")
print(f"Number of hashfuncs k of FLH = {k}, Epsilon = {epsilon}")
print(f"Optimal g={g}, prob_p={p}, prob_q={q}")
print("True Join Size of two datastreams = ", Join_ground_truth)
print("*------------------------------------------*")

hash_matrix = np.empty((k, domain))  # constructing pre-computed hash_matrix.
for i in range(k):
    for j in range(domain):
        hash_matrix[i][j] = xxhash.xxh32(str(unique_values[j]), seed=i).intdigest() % g

hash_counts1 = np.zeros((k, g))
hash_counts2 = np.zeros((k, g))
AE = 0
RE = 0
MSE1 = 0
MSE2 = 0
Est_JS = 0
testcycles = 10
print(f"FLH perturbing with testcycles = {testcycles}:")
for rd in range(testcycles):
    hash_counts1 = np.zeros((k, g))
    hash_counts2 = np.zeros((k, g))
    print(f"Processing round {rd+1}.")
    tp_st_time = time.time()
    n1 = 0  # number of aggregated items.
    n2 = 0
    tp_MSE1 = 0
    tp_MSE2 = 0
    Y1 = np.zeros(len(datastream_1))
    Y2 = np.zeros(len(datastream_2))
    for i in trange(len(datastream_1), desc="Perturbing first data stream:"):
        v = datastream_1[i]
        hash_seed = random.randint(0, k-1)
        x = (xxhash.xxh32(str(v), seed=hash_seed).intdigest() % g)
        y = x
        p_sample = np.random.random_sample()
        if p_sample > p - q:
            # perturb
            y = np.random.randint(0, g)
        Y1[i] = y
        # server aggregation
        hash_counts1[hash_seed][y] += 1
        n1 += 1

    for i in trange(len(datastream_2), desc="Perturbing second data stream"):
        v = datastream_2[i]
        hash_seed = random.randint(0, k-1)
        x = (xxhash.xxh32(str(v), seed=hash_seed).intdigest() % g)
        y = x
        p_sample = np.random.random_sample()
        if p_sample > p - q:
            # perturb
            y = np.random.randint(0, g)
        Y2[i] = y
        # server aggregation
        hash_counts2[hash_seed][y] += 1
        n2 += 1

    aggregated_data1 = list(np.apply_along_axis(func1, 0, hash_matrix))
    aggregated_data2 = list(np.apply_along_axis(func2, 0, hash_matrix))
    # print(aggregated_data1)
    # print(aggregated_data2)
    Est_dist1 = calibrating(aggregated_data1, p, g, n1)
    Est_dist2 = calibrating(aggregated_data2, p, g, n2)

    # calculating frequency estimation MSE.
    sum1 = 0
    for tf1, ef1 in zip(freq_vector_1, Est_dist1):
        sum1 += (tf1 - ef1) ** 2
    tp_MSE1 = sum1 / domain
    sum2 = 0
    for tf2, ef2 in zip(freq_vector_2, Est_dist2):
        sum2 += (tf2 - ef2) ** 2
    tp_MSE2 = sum2 / domain

    Est_Join_Size = sum(x*y for x, y in zip(Est_dist1, Est_dist2))

    tp_AAE = abs(Est_Join_Size - Join_ground_truth)
    tp_ARE = abs(Est_Join_Size - Join_ground_truth) / Join_ground_truth
    tp_ed_time = time.time()
    tp_total_time = tp_ed_time-tp_st_time

    # Writer to result file.
    re_row = ["FLH", filename.split('/')[-1], filename.split('/')[-1], k, epsilon, Join_ground_truth,
              Est_Join_Size, tp_AAE, tp_ARE, tp_MSE1, tp_MSE2, tp_total_time]
    res_filename = 'Results/FLH_res.csv'
    with open(res_filename, mode='a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(re_row)

    Est_JS += Est_Join_Size
    AE += tp_AAE
    RE += tp_ARE
    MSE1 += tp_MSE1
    MSE2 += tp_MSE2

AE /= testcycles
RE /= testcycles
MSE1 /= testcycles
MSE2 /= testcycles
Est_JS /= testcycles
ed_time = time.time()
total_time = ed_time-st_time

# Writer to result file.
re_row = ["FLH", filename.split('/')[-1], filename.split('/')[-1], k, epsilon, Join_ground_truth,
          Est_JS, AE, RE, MSE1, MSE2, total_time]
res_filename = 'Results/kRR_res.csv'
with open(res_filename, mode='a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(re_row)

print(f"*----------FLH Experiment Results----------*")
print(f"Dataset:{filename.split('/')[-1]}, Epsilon:{epsilon}, k:{k}.")
print(f"True Join Size = {Join_ground_truth}")
print("Est Join size:", Est_JS)
print("Absolute Error=", AE)
print("Relative Error=", RE)
print("MSE for data1=", MSE1)
print("MSE for data2=", MSE2)
print(f"Running time: {total_time} s")
