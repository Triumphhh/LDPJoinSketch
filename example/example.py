#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/11/12 10:29
# @Author  : liuxin22
# @File    : minitest.py
# @Software: PyCharm
import time
import csv
import numpy as np
import statistics
import parameter_setting as ps
from methods.frequency_est import FreqEst
from methods.AppleHCMS import AppleHCMS
from methods.FAGMS import FAGMS
from methods.KRR import KRR
from methods.FLH import FLH
from methods.LDPJoinSketch import LDPJoinSketch
from methods.LDPJoinSketch_plus import LDPJoinSketch_Plus

# usage: python example.py --k 4 --m 64 --epsilon 0.5 --theta 0.01 --dataset facebook --method ljs

def data_gen(file):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_vector = [int(row[0]) for row in reader]
    return data_vector

def ldp_join_sketch_example(k, m, eps, datastream0, datastream1):
    ljs0 = LDPJoinSketch(k, m, eps, datastream0)
    trf0 = ljs0.true_frequency()
    ljs0.insert_all()
    sk0 = ljs0.calibrating()

    ljs1 = LDPJoinSketch(k, m, eps, datastream1)
    trf1 = ljs1.true_frequency()
    ljs1.insert_all()
    sk1 = ljs1.calibrating()
    k_est_join_size = np.zeros(k, dtype=np.int64)
    for i in range(k):
        k_est_join_size[i] = sum(x * y for x, y in zip(sk0[i], sk1[i]))  # 直接利用sketch的内容计算连接基数
    est_joinsize = np.median(k_est_join_size)
    true_joinsize = 0
    for key, val in trf0.items():
        if key in trf1:
            true_joinsize += val * trf1[key]
    ab_error = abs(est_joinsize - true_joinsize)
    re_error = ab_error / true_joinsize
    print(f"The AE of joinsize_est using LDPJoinSketch: {ab_error}, RE: {re_error}")

def ldp_join_sketch_plus_example(k, m, mt, eps, sample_rate, theta, datastream0, datastream1):
    LA, LB = len(datastream0), len(datastream1)
    print(f"--Phase 1: Calculating high frequency item set from sampled data using LDPJoinSketch:")
    # handle data A
    ljspA = LDPJoinSketch_Plus(k, m, mt, eps, sample_rate, theta, datastream0)
    sampled_clients_A, group_A0, group_A1 = ljspA.divide_clients()
    FI_A, FI_freq_A = ljspA.get_high_frequency_set(sampled_clients_A)
    # handle data B
    ljspB = LDPJoinSketch_Plus(k, m, mt, eps, sample_rate, theta, datastream1)
    sampled_clients_B, group_B0, group_B1 = ljspB.divide_clients()
    FI_B, FI_freq_B = ljspB.get_high_frequency_set(sampled_clients_B)
    LRA0, LRA1 = len(group_A0), len(group_A1)
    LRB0, LRB1 = len(group_B0), len(group_B1)
    print(f"--Phase 2: Estimating join size by frequency-aware-perturbation:")
    FI, FA, FB = LDPJoinSketch_Plus.get_FI_intersection(FI_A, FI_freq_A, FI_B, FI_freq_B)
    hfA = sum(FA) * LA / len(sampled_clients_A)
    hfB = sum(FB) * LB / len(sampled_clients_B)
    A_sketch_H = ljspA.perturb(group_A0, FI, 1)
    A_sketch_L = ljspA.perturb(group_A1, FI, 0)
    B_sketch_H = ljspB.perturb(group_B0, FI, 1)
    B_sketch_L = ljspB.perturb(group_B1, FI, 0)

    HEst = LDPJoinSketch_Plus.join_est(k, m, A_sketch_H, B_sketch_H, LA, LB, hfA, hfB, 1)
    LEst = LDPJoinSketch_Plus.join_est(k, m, A_sketch_L, B_sketch_L, LA, LB, hfA, hfB, 0)

    est_join_size = LDPJoinSketch_Plus.get_join_size(HEst, LEst, LA, LRA0, LRA1, LB, LRB0, LRB1)
    trf0 = ljspA.true_frequency()
    trf1 = ljspB.true_frequency()
    true_joinsize = 0
    for key, val in trf0.items():
        if key in trf1:
            true_joinsize += val * trf1[key]
    ab_error = abs(true_joinsize - est_join_size)
    re_error = ab_error / true_joinsize
    print(f"The AE of joinsize_est using LDPJoinSketch+: {ab_error}, RE: {re_error}")

def fagms_example(k, m, datastream0, datastream1):
    fagms0 = FAGMS(k, m, datastream0)
    fagms1 = FAGMS(k, m, datastream1)
    sketch0 = fagms0.insert_all()
    sketch1 = fagms1.insert_all()
    k_est_join_size = np.zeros(k, dtype=np.int64)
    for i in range(k):
        k_est_join_size[i] = np.dot(sketch0[i], sketch1[i])
    est_join_size = statistics.median(k_est_join_size)
    trf0 = fagms0.true_frequency()
    trf1 = fagms1.true_frequency()
    true_joinsize = 0
    for key, val in trf0.items():
        if key in trf1:
            true_joinsize += val * trf1[key]
    ab_error = abs(true_joinsize - est_join_size)
    re_error = ab_error / true_joinsize
    print(f"The AE of joinsize_est using FAGMS: {ab_error}, RE: {re_error}")

def frequency_est_example(k, m, eps, theta, datastream):
    frequency = FreqEst(k, m, eps, datastream)  # parameters: k, m, epsilon, datastream
    frequency.insert_all()
    frequency.calibrating()
    r = frequency.frequency_est_all()
    fi, fi_freq = frequency.find_freq_itemset(r, theta)  # finding HH based on theta.
    mse = frequency.calculate_mse(r)  # calculating MSE of frequency estimation.
    print(f"The MSE of freq_est using LDPJoinSketch: {mse}. ")

def apple_hcms_example(k, m, eps, datastream0, datastream1):
    # for convenience, datastream1 = datastream2.
    hcms0 = AppleHCMS(k, m, eps, datastream0)
    hcms0.insert_all()
    hcms0.calibrating()
    hcms1 = AppleHCMS(k, m, eps, datastream1)
    hcms1.insert_all()
    hcms1.calibrating()
    r0 = hcms0.freq_est_all()
    r1 = hcms1.freq_est_all()
    trf0 = hcms0.true_frequency()
    trf1 = hcms1.true_frequency()
    true_joinsize = 0
    est_joinsize = 0
    for key, val in r0.items():
        if key in r1:
            est_joinsize += val * r1[key]
    for key, val in trf0.items():
        if key in trf1:
            true_joinsize += val * trf1[key]
    ab_error = abs(est_joinsize - true_joinsize)
    re_error = ab_error / true_joinsize
    print(f"The AE of joinsize_est using AppleHCMS: {ab_error}, RE: {re_error}")

def krr_example(eps, datastream0, datastream1):
    krr0 = KRR(eps, datastream0)
    krr1 = KRR(eps, datastream1)
    true_joinsize = 0
    trf0 = krr0.get_frequency()
    trf1 = krr1.get_frequency()
    for key, val in trf0.items():
        if key in trf1:
            true_joinsize += val * trf1[key]
    krr0.perturbing_all()
    priv_f0 = krr0.get_frequency()
    r0 = krr0.calibrating(priv_f0)
    krr1.perturbing_all()
    priv_f1 = krr1.get_frequency()
    r1 = krr1.calibrating(priv_f1)
    est_joinsize = 0
    for key, val in r0.items():
        if key in r1:
            est_joinsize += val * r1[key]
    ab_error = abs(est_joinsize - true_joinsize)
    re_error = ab_error / true_joinsize
    print(f"The AE of joinsize_est using KRR: {ab_error}, RE: {re_error}")

def flh_example(eps, datastream0, datastream1):
    flh0 = FLH(eps, datastream0)
    flh1 = FLH(eps, datastream1)
    true_joinsize = 0
    trf0 = flh0.true_frequency()
    trf1 = flh1.true_frequency()
    for key, val in trf0.items():
        if key in trf1:
            true_joinsize += val * trf1[key]
    flh0.perturbing_all()
    priv_f0 = flh0.aggregate()
    r0 = flh0.calibrating(priv_f0)
    flh1.perturbing_all()
    priv_f1 = flh1.aggregate()
    r1 = flh1.calibrating(priv_f1)
    est_joinsize = 0
    for key, val in r0.items():
        if key in r1:
            est_joinsize += val * r1[key]
    ab_error = abs(est_joinsize - true_joinsize)
    re_error = ab_error / true_joinsize
    print(f"The AE of joinsize_est using FLH: {ab_error}, RE: {re_error}")


def main():
    args = ps.get_args()
    hash_k = args.k
    hash_m = args.m
    eps = args.epsilon
    r = args.r
    theta = args.theta
    method = args.method

    # the file path needs to be modified according to your own environment.
    filename = '../LDPJoinSketch/SourceCode/data/'+args.dataset+'/'+args.dataset+'.csv'
    data = data_gen(filename)

    st_time = time.time()
    if method in ["freqest", "FreqEst"]:
        # example of frequency estimation using LDPJoinSketch.
        frequency_est_example(hash_k, hash_m, eps, theta, data)
    elif method in ["applehcms", "AppleHCMS"]:
        # example of Apple's HCMS
        apple_hcms_example(hash_k, hash_m, eps, data, data)
    elif method in ["fagms", "FAGMS"]:
        # example of FAGMS
        fagms_example(hash_k, hash_m, data, data)
    elif method in ["krr", "KRR"]:
        krr_example(eps, data, data)
    elif method in ["flh", "FLH"]:
        flh_example(eps, data, data)
    elif method in ["ljs", "LDPJoinSketch"]:
        ldp_join_sketch_example(hash_k, hash_m, eps, data, data)
    elif method in ["ljsp", "LDPJoinSketch_plus"]:
        ldp_join_sketch_plus_example(hash_k, hash_m, hash_m, eps, r, theta, data, data)

    ed_time = time.time()
    print(f"Running time: {ed_time - st_time} s.")


if __name__ == "__main__":
    main()

