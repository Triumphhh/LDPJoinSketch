# -*- coding: utf-8 -*-
import random
import statistics
from tqdm import *
from pure_ldp.frequency_oracles import *
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import csv
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

def calibrating(priv_freq, prob_p, prob_q, k, n):
    calibrated_freq = []
    for i in range(len(priv_freq)):
        # cal_p = ((priv_freq[i] / n + (prob - 1) / (k - 1)) / (2 * prob - 1)) * n
        cal_p = (priv_freq[i]-n*prob_q)/(prob_p-prob_q)
        calibrated_freq.append(cal_p)

    return calibrated_freq


# -------- choose dataset ------
# filename = 'data/zipf/zipf_10M_1_1.csv'
# filename = 'data/gaussian/gaussian.csv'
filename = 'data/tpc_ds/tpc_ds.csv'
# filename = 'data/twitter/twitter.csv'
# filename = 'data/facebook/facebook.csv'

data = read_csv_to_vector(filename)
datastream_1 = data
datastream_2 = data
print("Generate data streams done!")


unique_values = np.unique(np.concatenate((datastream_1, datastream_2)))
GRR_k = len(unique_values)  #kRR的范围，即数据域大小

st_time = time.time()
freq_vector_1 = get_frequencies(datastream_1, unique_values)
freq_vector_2 = get_frequencies(datastream_2, unique_values)
Join_ground_truth = sum(x * y for x, y in zip(freq_vector_1, freq_vector_2))

epsilon = 4  # 设置隐私预算
p = math.pow(math.e, epsilon) / (GRR_k + math.pow(math.e, epsilon) - 1)
q = 1 / (GRR_k + math.pow(math.e, epsilon) - 1)
print("*------kRR experiment result------*")
print(f"Using dataset:{filename} with epsilon={epsilon}, k={GRR_k}")
print(f"prob_p={p}, prob_q={q}")
print("True Join Size=", Join_ground_truth)

AE = 0
RE = 0
Est_JS = 0
testcycles = 1
for rd in range(testcycles):
    print(f"Processing round {rd+1}.")
    tp_st_time = time.time()
    for i in trange(len(datastream_1), desc="Perturbing first data stream:"):
        if np.random.random() > p:
            while True:
                t = np.random.choice(unique_values)
                if t != datastream_1[i]:
                    datastream_1[i] = t
                    break

    for i in trange(len(datastream_2), desc="Perturbing second data stream"):
        if np.random.random() > p:
            while True:
                t = np.random.choice(unique_values)
                if t != datastream_2[i]:
                    datastream_2[i] = t
                    break

    priv_freq_1 = get_frequencies(datastream_1, unique_values)
    priv_freq_2 = get_frequencies(datastream_2, unique_values)

    cali_freq_1 = calibrating(priv_freq_1, p, q, GRR_k, len(datastream_1))
    cali_freq_2 = calibrating(priv_freq_2, p, q, GRR_k, len(datastream_2))

    print(freq_vector_1[:10])
    print(priv_freq_1[:10])
    print(cali_freq_1[:10])
    Est_Join_Size = sum(x * y for x, y in zip(cali_freq_1, cali_freq_2))

    tp_AAE = abs(Est_Join_Size - Join_ground_truth)
    tp_ARE = abs(Est_Join_Size - Join_ground_truth) / Join_ground_truth
    tp_ed_time = time.time()
    tp_total_time = tp_ed_time-tp_st_time

    # Writer to result file.
    re_row = ["k-RR", filename.split('/')[-1], filename.split('/')[-1], epsilon, Join_ground_truth,
              Est_Join_Size, tp_AAE, tp_ARE, tp_total_time]
    res_filename = 'Results/kRR_res.csv'
    with open(res_filename, mode='a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(re_row)

    Est_JS += Est_Join_Size
    AE += abs(Est_Join_Size - Join_ground_truth)
    RE += abs(Est_Join_Size - Join_ground_truth) / Join_ground_truth

AE /= testcycles
RE /= testcycles
Est_JS /= testcycles
ed_time = time.time()
total_time = ed_time-st_time

# Writer to result file.
re_row = ["k-RR", filename.split('/')[-1], filename.split('/')[-1], epsilon, Join_ground_truth,
          Est_JS, AE, RE, total_time]
res_filename = 'Results/kRR_res.csv'
with open(res_filename, mode='a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(re_row)


print("Est Join size:", Est_JS)
print("Absolute Error=", AE)
print("Relative Error=", RE)