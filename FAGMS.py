from collections import defaultdict
import numpy as np
import mmh3
import csv
import statistics
import time
"""
Fast-AGMS Example.
"""

# global parameters definition
k = 18
m = 1024
print(f"(k,m)=({k},{m})")
hash_seed_time = 1000  # same hash_seed to ensure same hash functions in each sketch.


# -------- choose dataset ------
# filename = 'data/zipf/zipf_10M_1_1.csv'
# filename = 'data/gaussian/gaussian.csv'
# filename = 'data/tpc_ds/tpc_ds.csv'
# filename = 'data/twitter/twitter.csv'
filename = 'data/facebook/facebook.csv'

counter1 = [[] * m for _ in range(k)]
counter2 = [[] * m for _ in range(k)]
Heavy_Thes = 100
items = []
freq1 = defaultdict(int)
freq0 = defaultdict(int)
data_delimiter = 0
thres_large = 0
item_num = 0
flow_num = 0
Join_Ground_Truth = 0
est_join_size = 0
ARE = 0
AAE = 0

def read_csv_to_vector(filename):                # read data from csv file as data streams
    global data_delimiter, thres_large, item_num, flow_num, Join_Ground_Truth
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_vector = [int(row[0]) for row in reader]

    for i in range(len(data_vector)):
        items.append(data_vector[i])
    data_delimiter = len(items) // 2
    freq0.clear()
    freq1.clear()

    for i in range(len(items)):
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

    print(f"dataset name: {filename}")
    print(f"{flow_num} flows, {len(items)} items read")
    print(f"Heavy threshold = {Heavy_Thes}")
    print(f"max freq = {max_freq}")
    print(f"large flow sum = {thres_large}")
    print(f"Join Ground Truth = {Join_Ground_Truth}\n")

def init_sketch(k, m, counter):                       # sketch initializing operation
    for i in range(k):
        counter[i] = [0] * m
    return counter

def insert(k, m, item, id, hash_seed):      # using mmh3 ,the same as 'JoinSketch'
    global counter1, counter2
    index = [0] * k
    if id == 1:
        for i in range(k):
            index[i] = mmh3.hash(str(item), hash_seed+i, signed=False) % m
            g = mmh3.hash(str(item), hash_seed+i+k, signed=False) % 2
            if g == 0:
                counter1[i][index[i]] += 1
            if g == 1:
                counter1[i][index[i]] -= 1
    if id == 2:
        for i in range(k):
            index[i] = mmh3.hash(str(item), hash_seed+i, signed=False) % m
            g = mmh3.hash(str(item), hash_seed+i+k, signed=False) % 2
            if g == 0:
                counter2[i][index[i]] += 1
            if g == 1:
                counter2[i][index[i]] -= 1


read_csv_to_vector(filename)

testcycles = 1
st_time = time.time()
for t in range(testcycles):
    sketch1_phase1 = init_sketch(k, m, counter1)
    sketch2_phase1 = init_sketch(k, m, counter2)
    for i in range(len(items)):
        # if i < data_delimiter:
        #     insert(k, m, items[i], 1, hash_seed_time)
        # else:
        #     insert(k, m, items[i], 2, hash_seed_time)

        insert(k, m, items[i], 1, hash_seed_time)
        insert(k, m, items[i], 2, hash_seed_time)

    # Join Size Estimator.
    k_est_join_size = np.zeros(k, dtype=np.int64)
    for i in range(k):
        k_est_join_size[i] = sum(x*y for x, y in zip(counter1[i], counter2[i]))
    est_join_size = statistics.median(k_est_join_size)

    ARE += 1.0*abs(est_join_size-Join_Ground_Truth)/Join_Ground_Truth
    AAE += abs(est_join_size-Join_Ground_Truth)

AAE /= testcycles
ARE /= testcycles
ed_time = time.time()
total_time = ed_time-st_time


# Write to results file.

re_row = ["FAGMS", filename.split('/')[-1], filename.split('/')[-1], k, m, Join_Ground_Truth, est_join_size, AAE, ARE, total_time]
res_filename = 'Results/Result_1.csv'
with open(res_filename, mode='a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(re_row)

print(f"Result under k={k},m={m}.")
print(f"Est re = {est_join_size}")
print(f"ARE = {ARE}")
print(f"AAE = {AAE}")
print(f"RunningTime = {total_time}s")

