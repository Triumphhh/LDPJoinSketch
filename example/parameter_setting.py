#!/usr/bin/python3.10.6
# -*- coding: utf-8 -*-
# @Time    : 2023/11/12 10:09
# @Author  : liuxin22
# @File    : parameter_setting.py.py
# @Software: PyCharm

import argparse
import warnings

def get_args():
    """
    Initializing parameters.
    """
    parser = argparse.ArgumentParser()

    # type int
    parser.add_argument("--k", type=int, default=16, help="number of hash functions")
    parser.add_argument("--m", type=int, default=256, help="domain size of sketch")

    # type float
    parser.add_argument("--epsilon", type=float, default=0.1, help="privacy budget")
    parser.add_argument("--r", type=float, default=0.1, help="sample rate for LDPJoinSketch+")
    parser.add_argument("--theta", type=float, default=0.001, help="threshold of high frequency")

    # type str
    parser.add_argument("--dataset", type=str, default="", help="dataset name")
    parser.add_argument("--method", type=str, default="", help="method used")

    args = parser.parse_args()
    if args.epsilon < 0:
        raise argparse.ArgumentTypeError('epsilon should be greater than 0!')
    if args.epsilon > 10:
        warnings.warn('epsilon is too large to protect privacy!')

    return args
