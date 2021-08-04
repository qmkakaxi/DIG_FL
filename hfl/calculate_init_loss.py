#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :calculate_init_loss.py.py
# @Time      :2020/12/28 3:30 下午
# @Author    :wangjunhao


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
from utils.sampling import mnist_iid, mnist_noniid_new, cifar_iid, mnist_remove, mnist_noniid
from utils.options import args_parser
from models.Update import LocalUpdate,DatasetSplit
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg,FedAvg_weight,FedAvg_test
from models.test import test_img



if __name__ == '__main__':
    # parse args
    args = args_parser()

    # 设置参数
    args.dataset= 'MNIST'
    args.num_channels=1
    args.model="cnn"
    args.iid=True
    args.epochs=20
    args.local_bs=4
    args.verbose=True
    args.num_users=5
    args.frac=0.6
    settime=0
    args.gpu=3
    args.local_ep=1


    #设置cpu
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('data/', train=True, download=True,transform=trans_mnist)
    dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)

    dict_users=torch.load('log/mnist_2/mnist_fed_noisy_1/dict_users')



    print("begin train...")
    net_train=CNNMnist(args=args).to(args.device)

    net_train.load_state_dict(torch.load('log/mnist_2/mnist_fed_noisy_1/init'))
    print(net_train)
    net_train.train()
    net_test=CNNMnist(args=args).to(args.device)

    test_accuracy,test_loss = test_img(net_train, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(test_accuracy))
    print("Testing loss: {:.2f}".format(test_loss))
    torch.save(test_accuracy,'log/mnist_2/mnist_fed_noisy_1/init_acc')
    torch.save(test_loss,'log/mnist_2/mnist_fed_noisy_1/init_loss')
    print(test_loss)
