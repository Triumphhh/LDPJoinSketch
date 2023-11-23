#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
from tqdm import *
import numpy as np
from collections import defaultdict, Counter
import time
import random
import csv
import mmh3
import math
from collections import Counter
from scipy.linalg import hadamard
import warnings
import sys
from pympler import asizeof
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
LDPJoinSketch_plus Example.
"""

k = 18
m = 512
mt = 512
epsilon = 4
sample_rate = 0.1      # 采样率
theta = 0.0001        # 频繁项阈值,需要根据数据集进行修改
print(f"k = {k}, m = {m}, Epsilon = {epsilon}")
print(f"sample rate = {sample_rate}, highfreq threshold rate = {theta}")
hash_seed_time = 1000
# hash_seed_time = int(time.time())
c = (math.pow(math.e, epsilon) + 1) / (math.pow(math.e, epsilon) - 1)
probability = 1 / (math.pow(math.e, epsilon) + 1)

# -------- choose dataset ------
filename = 'data/zipf/zipf_20M_2_0.csv'
# filename = 'data/gaussian/gaussian.csv'
# filename = 'data/movielens/movielens.csv'
# filename = 'data/tpc_ds/tpc_ds.csv'
# filename = 'data/twitter/twitter.csv'
# filename = 'data/facebook/facebook.csv'
print(f"filename = {filename}")

had = hadamard(m)   # construct m*m hadamard matrix.
hadt = hadamard(mt)
# 实验一共用到6个sketch
# phase1: two sketches for sampled users.
# sketch1_phase1 = [[0] * mt for _ in range(k)]
# sketch2_phase1 = [[0] * mt for _ in range(k)]
sketch1_phase1 = np.zeros((k, mt))
sketch2_phase1 = np.zeros((k, mt))
# phase2: four sketches for remaining users after group dividing.
# sketch1_phase2_P = [[0] * m for _ in range(k)]
# sketch1_phase2_Q = [[0] * m for _ in range(k)]
# sketch2_phase2_P = [[0] * m for _ in range(k)]
# sketch2_phase2_Q = [[0] * m for _ in range(k)]
sketch1_phase2_P = np.zeros((k, m))
sketch1_phase2_Q = np.zeros((k, m))
sketch2_phase2_P = np.zeros((k, m))
sketch2_phase2_Q = np.zeros((k, m))


data_stream1 = []
data_stream2 = []
Join_Ground_Truth = 0
# Sampled_Join_Ground_Truth = 0                   # Join Ground Truth of S1 and S2
data_delimiter = 0
thres_large = 0
item_num = 0
flow_num = 0
freq_ground_truth1 = defaultdict(int)   # 用于记录真实数据流的各项频数
freq_ground_truth2 = defaultdict(int)
Est_Join_Size = 0

def read_csv_to_vector(filename):  # read data from csv file as data streams
    global data_delimiter, thres_large, item_num, flow_num, Join_Ground_Truth, data_stream1, data_stream2, \
        freq_ground_truth1, freq_ground_truth2
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_vector = [int(row[0]) for row in reader]

    for i in range(len(data_vector)):
        data_stream1.append(data_vector[i])   # 这里考虑自连接，因此两个数据流都为整个表数据向量
        data_stream2.append(data_vector[i])

    data_delimiter = len(data_stream1)   # 均分数据流时使用，自连接情况不考虑
    freq_ground_truth1.clear()
    freq_ground_truth2.clear()

    for i in range(len(data_stream1)):  # two different half as two data streams
        # if i < data_delimiter:
        #     freq1[items[i]] += 1
        # else:
        #     freq0[items[i]] += 1
        freq_ground_truth1[data_stream1[i]] += 1      # 自连接
    for i in range(len(data_stream2)):
        freq_ground_truth2[data_stream2[i]] += 1      # 自连接
    # 报告两个数据流中不同项的数量
    # print(f"len of freq_1 = {len(freq_ground_truth1)},\n len of freq_2 = {len(freq_ground_truth2)}")
    # 报告两个数据流中最大的频数max_freq
    max_freq = max([max(freq_ground_truth1.values()), max(freq_ground_truth2.values())])

    Heavy_Thes = theta * len(data_vector)  # 根据theta计算原始数据流中频繁项应达到的频数
    for key, val in freq_ground_truth1.items():
        if key in freq_ground_truth2:
            if val > Heavy_Thes:
                thres_large += 1
            Join_Ground_Truth += val * freq_ground_truth1[key]

    item_num = len(data_stream1)
    flow_num = len(freq_ground_truth1) + len(freq_ground_truth2)

    print(f"*-------- Information of Original Dataset ------*")
    print(f"dataset name: {filename}")
    print(f"len of freq_1 = {len(freq_ground_truth1)}, len of freq_2 = {len(freq_ground_truth2)}")
    print(f"{int(flow_num/2)} flows, {item_num} items read")
    print(f"max freq in datastream = {max_freq}")
    print(f"freq > {Heavy_Thes} large flow numbers = {thres_large}")
    print(f"Join Ground Truth = {Join_Ground_Truth}\n")


def init_sketch(k, m, counter):  # sketch initializing operation
    # for i in range(k):
    #     counter[i] = [0] * m
    # return counter
    return np.zeros((k, m))

def insert(k, m, item, id, hash_seed, prob):  # LDPJoinSketch. Used in phase1 for FI.
    global sketch1_phase1, sketch2_phase1
    # client side
    j = np.random.randint(0, k)
    l = np.random.randint(0, m)
    v = [0] * m
    index_j = mmh3.hash(str(item), hash_seed + j, signed=False) % m  # k different hash functions.
    v[index_j] = 2 * (mmh3.hash(str(item), hash_seed + j + k, signed=False) % 2) - 1
    w = hadt[:, index_j] * v[index_j]
    b = random.choices([-1, 1], k=1, weights=[prob, 1 - prob])  # LDP perturbation
    y = b[0] * w[l]

    # server side
    if id == 1:
        sketch1_phase1[j, l] = sketch1_phase1[j, l] + k * c * y
    elif id == 2:
        sketch2_phase1[j, l] = sketch2_phase1[j, l] + k * c * y

def fre_est(uniq_items, sketch, m):   # 用JoinSketch估计频率
    est_freq = []
    for item_i in uniq_items:
        sums = 0
        for ik in range(k):
            index_ik = mmh3.hash(str(item_i), hash_seed_time + ik, signed=False) % m
            xi = 2 * (mmh3.hash(str(item_i), hash_seed_time + ik + k, signed=False) % 2) - 1
            sums += sketch[ik][index_ik] * xi
        f = 1/k * sums
        # f = (m/(m-1)) * (f - (item_num*theta)/(2*m))
        est_freq.append(f)
    return est_freq

def find_freq_item(uniq_items, estf, thres):
    freq_item = []
    for i in range(len(estf)):
        if estf[i] > thres:
            freq_item.append(uniq_items[i])
    return freq_item

def find_heavyhitters(uniq_items, estf, topk):
    sorted_indices = np.argsort(estf)[::-1]
    freq_item_id = []
    freq_item = []
    for ii in range(topk):
        index = sorted_indices[ii]
        freq_item_id.append(index)
    for iid in freq_item_id:
        freq_item.append(uniq_items[iid])
    return freq_item

def find_from_dic(freq_dic, thres):
    freq_item = []
    for ky, val in freq_dic.items():
        if val > thres:
            freq_item.append(ky)
    return freq_item

def get_true_frequencies(datastream, unique_values):
    freq_dict = defaultdict(int)
    for val in datastream:
        freq_dict[val] += 1
    return [freq_dict[val] for val in unique_values]

def freq_aware_perturb(item, mode, id, k, m, freq_set, hash_seed, prob):   # LDPJoinSketch+ FAP
    global sketch1_phase2_P, sketch1_phase2_Q, sketch2_phase2_P, sketch2_phase2_Q
    if (mode == 1 and item not in freq_set) or (mode == 0 and item in freq_set):  # mode和item两个条件同时为真或同时为否
        j = np.random.randint(0, k)
        l = np.random.randint(0, m)
        r = np.random.randint(0, m)
        v = [0] * m
        v[r] = 1
        w = had[:,r]
        b = random.choices([-1, 1], k=1, weights=[prob, 1 - prob])  # LDP perturbation
        y = b[0] * w[l]
    else:
        j = np.random.randint(0, k)
        l = np.random.randint(0, m)
        v = [0] * m
        index_j = mmh3.hash(str(item), hash_seed + j, signed=False) % m  # k different hash functions.
        v[index_j] = 2 * (mmh3.hash(str(item), hash_seed + j + k, signed=False) % 2) - 1
        w = had[:, index_j] * v[index_j]
        b = random.choices([-1, 1], k=1, weights=[prob, 1 - prob])  # LDP perturbation
        y = b[0] * w[l]
    # server-side the same as LDPJoinSketch:
    # mode == H
    if id == 11:
        sketch1_phase2_P[j, l] = sketch1_phase2_P[j, l] + k * c * y
    elif id == 21:
        sketch2_phase2_P[j, l] = sketch2_phase2_P[j, l] + k * c * y
    # mode == L
    elif id == 12:
        sketch1_phase2_Q[j, l] = sketch1_phase2_Q[j, l] + k * c * y
    else:
        sketch2_phase2_Q[j, l] = sketch2_phase2_Q[j, l] + k * c * y

def joinest(sketch1, sketch2, size1, sample_size1, size2, sample_size2, freq_set, hfs1_set, hfs2_set, mode): # JoinEst
    # global sketch1_phase2_P, sketch1_phase2_Q, sketch2_phase2_P, sketch2_phase2_Q
    hf1 = 0
    hf2 = 0
    for t in range(len(freq_set)):
        # hf1 += hfs1_set[t] * (size1/sample_size1)
        # hf2 += hfs2_set[t] * (size2/sample_size2)
        hf1 = sum(hfs1_set) * (size1/sample_size1)
        hf2 = sum(hfs2_set) * (size2/sample_size2)
    if mode == 0:
        for ii in range(k):
            for jj in range(m):
                sketch1[ii, jj] = sketch1[ii, jj] - hf1 / m
                sketch2[ii, jj] = sketch2[ii, jj] - hf2 / m
    else:
        for ii in range(k):
            for jj in range(m):
                sketch1[ii, jj] = sketch1[ii, jj] - (size1 - hf1) / m
                sketch2[ii, jj] = sketch2[ii, jj] - (size2 - hf2) / m
    # sketch1 = np.matmul(sketch1, np.transpose(had))
    # sketch2 = np.matmul(sketch2, np.transpose(had))
    k_est = np.zeros(k)
    for ii in range(k):
        # k_est[ii] = sum(x * y for x, y in zip(sketch1[ii], sketch2[ii]))
        k_est[ii] = np.dot(sketch1[ii], sketch2[ii])
    est = np.median(k_est)
    return est


read_csv_to_vector(filename)   # 读取文件并构造数据流

st_time = time.time()
testcycles = 10

data_stream1_counter = Counter(data_stream1)  # 构建type类型，用于后面排除采样数据，保留剩余数据
data_stream2_counter = Counter(data_stream2)

# ------------ phase1 start ------------
print(f"*-------- Phase1: Sampling for finding FI. -------*")
ns1 = int(len(data_stream1) * sample_rate)    # 第1个表采样用户（项目）数量
ns2 = int(len(data_stream2) * sample_rate)    # 第2个表采样用户（项目）数量
print(f"Sampled number for data1:{ns1}, for data2:{ns2}.")
thresh1 = theta * ns1   # 通过采样数据判断项目是否为频繁项
thresh2 = theta * ns2
print(f"Threshold1 = {thresh1}, threshold2 = {thresh2}.")
S1 = random.sample(data_stream1, ns1)         # 根据采样数量对数据流进行随机抽样
S2 = random.sample(data_stream2, ns2)
S1_counter = Counter(S1)
S2_counter = Counter(S2)
uniq_S1 = list(np.unique(S1))       # 采样数据中的不同项，按逻辑频繁项一定能被采样到
uniq_S2 = list(np.unique(S2))
true_freqvec_S1 = get_true_frequencies(S1, uniq_S1)
true_freqvec_S2 = get_true_frequencies(S2, uniq_S2)
# aso1 = np.argsort(true_freqvec_S1)[::-1]
# aso2 = np.argsort(true_freqvec_S2)[::-1]
# ts1 = []
# ts2 = []
# tsf1 = []
# tsf2 = []
# for it in range(20):
#     ts1.append(aso1[it])
#     ts2.append(aso2[it])
#     tsf1.append(true_freqvec_S1[aso1[it]])
#     tsf2.append(true_freqvec_S2[aso2[it]])
#
# print(ts1)
# print(tsf1)
# print(ts2)
# print(tsf2)

acc_freq1 = defaultdict(int)
acc_freq2 = defaultdict(int)

print(f"len of uniq_S1={len(uniq_S1)}, uniq_S2={len(uniq_S2)}.")
for i in range(len(S1)):             # 客户端将采样数据第一阶段的两个sketch，SA1和SA2
    acc_freq1[S1[i]] += 1   # 尝试准确记录所有项频数
    insert(k, mt, S1[i], 1, hash_seed_time, probability)
for i in range(len(S2)):
    acc_freq2[S2[i]] += 1
    insert(k, mt, S2[i], 2, hash_seed_time, probability)
sketch1_phase1 = np.matmul(sketch1_phase1, np.transpose(hadt))  # server calibration.
sketch2_phase1 = np.matmul(sketch2_phase1, np.transpose(hadt))

# 通过采样数据的频率估计， Est_F与uniq_S对应位置元素相同
Est_F1 = fre_est(uniq_S1, sketch1_phase1, mt)  # fd of SA
Est_F2 = fre_est(uniq_S2, sketch2_phase1, mt)  # fd of SB


# Freq_id1 = find_freq_item(uniq_S1, Est_F1, thresh1)  # 根据thresh条件查找两个表中的频繁项项目
# Freq_id2 = find_freq_item(uniq_S2, Est_F2, thresh2)
# Freq_id1 = find_heavyhitters(uniq_S1, Est_F1, thres_large)
# Freq_id2 = find_heavyhitters(uniq_S2, Est_F2, thres_large)
Freq_id1 = find_from_dic(acc_freq1, thresh1)
Freq_id2 = find_from_dic(acc_freq2, thresh2)
print(f"Number of high freq items in datastream1: {len(Freq_id1)}, datastream2: {len(Freq_id2)}")
# print(Freq_id1)
# print(Freq_id2)
# set1 = set(Freq_id1)
# set2 = set(Freq_id2)
# buji_id = list(set1.union(set2) - set1.intersection(set2))  # 计算两个采样流频繁项的补集
FI = list(set(Freq_id1).intersection(set(Freq_id2)))  # 先求两个频繁项集合的交集FI
# FI = list(set(Freq_id1).union(set(Freq_id2)))
# for iid in buji_id:  # 向交集中添加在两个采样数据中都存在的频繁项
#     if iid in uniq_S1 and iid in uniq_S2:
#         FI.append(iid)

Est_Fd1 = []  # 用于保存频繁项并集在两个采样数据中的频数
Est_Fd2 = []
for item in FI:
    Est_Fd1.append(acc_freq1[item])  # 尝试准确记录高频项频数
    for t1 in range(len(uniq_S1)):
        if uniq_S1[t1] == item:
            Est_Fd1.append(Est_F1[t1])
            break
for item in FI:
    Est_Fd2.append(acc_freq2[item])
    for t2 in range(len(uniq_S2)):
        if uniq_S2[t2] == item:
            Est_Fd2.append(Est_F2[t2])
            break
print(f"Est_Fd1:{Est_Fd1}")
print(f"Est_Fd2:{Est_Fd2}")

print("true high frequent join size:", sum(a*b for a,b in zip(Est_Fd1, Est_Fd2))/(sample_rate**2))
print("true low frequent join size:", Join_Ground_Truth-sum(a*b for a,b in zip(Est_Fd1, Est_Fd2))/(sample_rate**2))

print(f"len of FI={len(FI)}, Est_Fd1={len(Est_Fd1)}, Est_Fd2={len(Est_Fd2)}.")
if not (len(FI) == len(Est_Fd1) and len(Est_Fd1) == len(Est_Fd2)):
    print(f"----Warring: the frequent threshold theta={theta} is too low thus causing item only sampled in one table!---")
    sys.exit(1)
elif len(Est_Fd1) == 0 and len(Est_Fd2) == 0:
    print(f"---Note: the frequent threshold theta={theta} is too high, no frequent items found!---")
print(f"*-------- Phase1 Finished! --------*")
# ----------------------------------phase1 finished--------------------------------------------#

# ----------------------------------phase2 started---------------------------------------------#
print(f"*-------- Phase2 Started! --------*")
Remain_data1 = list((data_stream1_counter-S1_counter).elements())  # 排除采样数据后的数据表
Remain_data2 = list((data_stream2_counter-S2_counter).elements())
random.shuffle(Remain_data1)   # 打乱列表数据
random.shuffle(Remain_data2)
Remain_data1_P = Remain_data1[:len(Remain_data1)//2]   # 分别对两个表的剩余数据进行随机均匀分组
Remain_data1_Q = Remain_data1[len(Remain_data1)//2:]
Remain_data2_P = Remain_data2[:len(Remain_data2)//2]
Remain_data2_Q = Remain_data2[len(Remain_data2)//2:]
print(f"len of 4 table: {len(Remain_data1_P)}, {len(Remain_data2_P)}, {len(Remain_data1_Q)}, {len(Remain_data2_Q)}")
# mode=1 indicates H, mode=0 indicates L.
# M1 -----  M2   # sketches of remain data.
# M11(1P)      M21(2P)   # mode == H
#      *    *
#       *  *
#        *
#       *  *
#      *    *
# M12(1Q)      M22(2Q)   # mode == L
# Mode == H
for i in trange(len(Remain_data1_P), desc="FAP for A1:"):
    freq_aware_perturb(Remain_data1_P[i], 1, 11, k, m, FI, hash_seed_time, probability)
for i in trange(len(Remain_data2_P), desc="FAP for B1:"):
    freq_aware_perturb(Remain_data2_P[i], 1, 21, k, m, FI, hash_seed_time, probability)
# Mode == L
for i in trange(len(Remain_data1_Q), desc="FAP for A2:"):
    freq_aware_perturb(Remain_data1_Q[i], 0, 12, k, m, FI, hash_seed_time, probability)
for i in trange(len(Remain_data2_Q), desc="FAP for B2:"):
    freq_aware_perturb(Remain_data2_Q[i], 0, 22, k, m, FI, hash_seed_time, probability)

# server side calibration.
sketch1_phase2_P = np.matmul(sketch1_phase2_P, np.transpose(had))
sketch1_phase2_Q = np.matmul(sketch1_phase2_Q, np.transpose(had))
sketch2_phase2_P = np.matmul(sketch2_phase2_P, np.transpose(had))
sketch2_phase2_Q = np.matmul(sketch2_phase2_Q, np.transpose(had))

HEst = joinest(sketch1_phase2_P, sketch2_phase2_P, len(Remain_data1_P), ns1, len(Remain_data2_P), ns2, FI, Est_Fd1,
               Est_Fd2, 1)
LEst = joinest(sketch1_phase2_Q, sketch2_phase2_Q, len(Remain_data1_Q), ns1, len(Remain_data2_Q), ns2, FI, Est_Fd1,
               Est_Fd2, 0)
print(f"HEst = {HEst}, LEst = {LEst}")

A = len(data_stream1)
B = len(data_stream2)
A1 = len(Remain_data1_P)
A2 = len(Remain_data1_Q)
B1 = len(Remain_data2_P)
B2 = len(Remain_data2_Q)
"""
Est_Join_Size = (len(data_stream1)*len(data_stream2)) * \
                ((HEst / (len(Remain_data1_P)*len(Remain_data2_P))) + \
                (LEst / ((len(Remain_data1_Q)) * len(Remain_data2_Q))))
"""
Est_Join_Size = ((A*B)/(A1*B1))*HEst + ((A+B)/(A2*B2))*LEst
ed_time = time.time()
total_time = ed_time-st_time
AE = abs(Join_Ground_Truth-Est_Join_Size)
RE = abs(Join_Ground_Truth-Est_Join_Size)/Join_Ground_Truth

size_in_bytes1 = asizeof.asizeof(sketch1_phase1)  # space cost in phase 1
size_in_bytes2 = asizeof.asizeof(sketch2_phase1)
space1 = (size_in_bytes1 + size_in_bytes2)/1000

space_cost_1 = asizeof.asizeof(sketch1_phase2_P)
space_cost_2 = asizeof.asizeof(sketch2_phase2_P)
space_cost_3 = asizeof.asizeof(sketch1_phase2_Q)
space_cost_4 = asizeof.asizeof(sketch2_phase2_Q)
space2 = (space_cost_1 + space_cost_2 + space_cost_3 + space_cost_4)/1000

total_space_cost = (size_in_bytes1 + size_in_bytes2 + space_cost_1 + space_cost_2 + space_cost_3 + space_cost_4)/1000

print("------LDPJoinSketch+ Experiment Results------")
print(f"Estimated Join Size={Est_Join_Size}")
print(f"AE = {AE}")
print(f"RE = {RE}")
print(f"RunningTime = {total_time}s")
print(f"Total Space cost = {total_space_cost} KB")
print(f"Space cost in phase 1: {space1} KB")
print(f"Space cost in phase 2: {space2} KB")
