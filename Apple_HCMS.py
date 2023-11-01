#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 10:21
# @Author  : liuxin22
# @File    : Apple_CMS.py.py
# @Software: PyCharm

import numpy as np
from collections import defaultdict
import time
import random
from tqdm import *
import csv
import mmh3
import math
from collections import Counter
from scipy.linalg import hadamard
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Apple-HCMS Example.
"""

# global parameters definition

k = 18
m = 1024

epsilon = 4
print(f"k = {k}, m = {m}, Epsilon = {epsilon}")
hash_seed_time = 1000
Heavy_Thes = 100  # 设置频繁项的阈值

c = (math.pow(math.e, epsilon) + 1) / (math.pow(math.e, epsilon) - 1)
print(f"Calibration c = {c}")
had = hadamard(m)  # 构建大小为m*m的hadamard矩阵

# -------- choose dataset ------
# filename = 'data/zipf/zipf_10M_1_1.csv'
# filename = 'data/gaussian/gaussian.csv'
filename = 'data/tpc_ds/tpc_ds.csv'
# filename = 'data/twitter/twitter.csv'
# filename = 'data/facebook/facebook.csv'

est_join_size = 0
probability = 1 / (math.pow(math.e, epsilon) + 1)
print(f"Perturbation probability:{probability}")
Est = 0  # estimate result by frequency vector

counter1 = [[0] * m for _ in range(k)]
counter2 = [[0] * m for _ in range(k)]
items = []
freq1 = defaultdict(int)
freq0 = defaultdict(int)
estf1 = defaultdict(float)
estf2 = defaultdict(float)
estimate_freq1 = defaultdict(float)
estimate_freq2 = defaultdict(float)
data_delimiter = 0
thres_large = 0  # 大于阈值的频繁项数量
item_num = 0
flow_num = 0
Join_Ground_Truth = 0
ARE = 0
AAE = 0
MSE1 = 0  # 第一个数据流的频率估计均方误差
MSE2 = 0  # 第二个数据流的频率估计均方误差

def read_csv_to_vector(filename):  # read data from csv file as data streams
    global data_delimiter, thres_large, item_num, flow_num, Join_Ground_Truth, items, freq0, freq1
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_vector = [int(row[0]) for row in reader]

    for i in range(len(data_vector)):
        items.append(data_vector[i])

    data_delimiter = len(items) // 2
    freq0.clear()
    freq1.clear()

    for i in range(len(items)):  # two different half as two data streams
        # if i < data_delimiter:
        #     freq1[items[i]] += 1
        # else:
        #     freq0[items[i]] += 1

        freq0[items[i]] += 1
        freq1[items[i]] += 1
    print(f"len of freq0 = {len(freq0)}, len of freq1 = {len(freq1)}")
    max_freq = max([max(freq0.values()), max(freq1.values())])

    for k, v in freq0.items():
        if k in freq1:
            if v > Heavy_Thes:
                thres_large += 1
            Join_Ground_Truth += v * freq1[k]

    item_num = len(items)
    flow_num = len(freq0) + len(freq1)

    print(f"*-------- Information of Original Dataset ------*")
    print(f"dataset name: {filename}")
    print(f"{flow_num} flows, {len(items)} items read")
    print(f"max freq = {max_freq}")
    print(f"large flow sum = {thres_large}")
    print(f"Join Ground Truth = {Join_Ground_Truth}\n")

def init_sketch(k, m, counter):  # sketch initializing operation
    for i in range(k):
        counter[i] = [0] * m
    return counter

def insert(k, m, item, id, hash_seed, prob):  # sketch inserting operation
    global counter1, counter2

    # client side
    j = np.random.randint(0, k)
    l = np.random.randint(0, m)
    v = [0] * m
    index_j = mmh3.hash(str(item), hash_seed + j, signed=False) % m  # k different hash functions.
    #v[index_j] = 2 * (mmh3.hash(str(item), hash_seed + j + k, signed=False) % 2) - 1
    v[index_j] = 1  # Apple-HCMS
    # w = had[:, index_j] * v[index_j]
    w = had[:, index_j]  # Apple-HCMS
    b = random.choices([-1, 1], k=1, weights=[prob, 1 - prob])  # LDP perturbation
    y = b[0] * w[l]

    # server side
    if id == 1:
        counter1[j][l] = counter1[j][l] + k * c * y
    elif id == 2:
        counter2[j][l] = counter2[j][l] + k * c * y


read_csv_to_vector(filename)
n = item_num  # len(items)  # 数据流的大小
uniq_item = list(np.unique(items))

testcycles = 10  # 设置测试轮数
st_time = time.time()
for t in range(testcycles):  # main procedure
    tp_stime = time.time()
    print(f"Processing round {t+1}:")
    est_join_size = 0
    counter1 = init_sketch(k, m, counter1)
    counter2 = init_sketch(k, m, counter2)
    estf1.clear()
    estf2.clear()
    for i in trange(len(items), desc='perturbing data1'):

        # if i < data_delimiter:
        #     insert(k, m, items[i], 1, hash_seed_time, probability)
        # else:
        #     insert(k, m, items[i], 2, hash_seed_time, probability)
        insert(k, m, items[i], 1, hash_seed_time, probability)
    for i in trange(len(items), desc='perturbing data2'):
        insert(k, m, items[i], 2, hash_seed_time, probability)
    counter1 = np.matmul(counter1, np.transpose(had))  # server calibration.
    counter2 = np.matmul(counter2, np.transpose(had))

    # Join Size Estimator.
    for item in uniq_item:
        sums = 0
        for i in range(k):
            index = mmh3.hash(str(item), hash_seed_time + i, signed=False) % m
            sums += counter1[i][index]
        estf1[item] = (m/(m-1))*((1/k)*sums-(n/m))

    for item in uniq_item:
        sums = 0
        for i in range(k):
            index = mmh3.hash(str(item), hash_seed_time + i, signed=False) % m
            sums += counter2[i][index]
        estf2[item] = (m / (m - 1)) * ((1 / k) * sums - (n / m))

    EE1 = 0
    EE2 = 0
    for key, val in estf1.items():
        if key in estf2:
            est_join_size += val * estf2[key]
            #print(f"{key},{val},{estf2[key]}")
        EE1 += (freq0[key]-estf1[key])**2
        EE2 += (freq1[key]-estf2[key])**2

    # Record the result of each round.
    tp_MSE1 = EE1 / len(uniq_item)
    tp_MSE2 = EE2 / len(uniq_item)
    tp_AAE = 1.0 * abs(est_join_size - Join_Ground_Truth)
    tp_ARE = 1.0 * abs(est_join_size - Join_Ground_Truth) / Join_Ground_Truth
    tp_etime = time.time()
    tp_totaltime = tp_etime - tp_stime
    # Writer to result file.
    re_row = ["HCMS", filename.split('/')[-1], filename.split('/')[-1], k, m, epsilon, Join_Ground_Truth,
              est_join_size, tp_AAE, tp_ARE, tp_MSE1, tp_MSE2, tp_totaltime]
    res_filename = 'Results/HCMS_res.csv'
    with open(res_filename, mode='a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(re_row)

    print(f"Est result:{est_join_size}")
    print(f"AE of round {t + 1}:{1.0 * abs(est_join_size - Join_Ground_Truth)}")
    print(f"RE of round {t + 1}:{1.0 * abs(est_join_size - Join_Ground_Truth) / Join_Ground_Truth}")
    print(f"MSE1 of round {t + 1}:{tp_MSE1}")
    print(f"MSE2 of round {t + 1}:{tp_MSE2}")
    ARE += 1.0 * abs(est_join_size - Join_Ground_Truth) / Join_Ground_Truth
    AAE += abs(est_join_size - Join_Ground_Truth)
    MSE1 += tp_MSE1
    MSE2 += tp_MSE2

AAE /= testcycles
ARE /= testcycles
MSE1 /= testcycles
MSE2 /= testcycles

ed_time = time.time()
total_time = ed_time - st_time

# Writer to result file.
re_row = ["HCMS", filename.split('/')[-1], filename.split('/')[-1], k, m, epsilon, Join_Ground_Truth,
          est_join_size, AAE, ARE, MSE1, MSE2, total_time]
res_filename = 'Results/HCMS_res.csv'
with open(res_filename, mode='a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(re_row)

print(f"*------Apple-HCMS Experiment Result ------*")
print(f"Result under testcycles={testcycles}, k={k},m={m}, eps={epsilon}:")
print(f"Estimated Join Size={est_join_size}")
print(f"AE = {AAE}")
print(f"RE = {ARE}")
print(f"MSE1 = {MSE1}")
print(f"MSE2 = {MSE2}")
print(f"RunningTime = {total_time}s")