#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_test(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k]=w_avg[k]*0.5
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]*0.5
        # w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_weight(w,weight):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k]=w_avg[k]*weight[0]
        for i in range(1, len(w)):
            w_avg[k] += (w[i][k]*weight[i])
        w_avg[k] = torch.div(w_avg[k], sum(weight))
        # print(sum(weight))
    return w_avg



