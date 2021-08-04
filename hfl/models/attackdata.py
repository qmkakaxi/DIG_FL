import matplotlib
import argparse
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
from models.Update import LocalUpdate
import torch

from utils.sampling import mnist_iid
from utils.options import args_parser
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

from torch.utils.data import DataLoader
import pickle




def generate_attack_data(dataName, dict_users, dataset, noisy_client, noisy_rate):
    """
    将一部分用户数据中的2与6的数据混合，形成标签为2的新数据
    noisy_rate是加的噪音比例
    """
    if dataName=='MNIST':
        originTargets=dataset.train_labels.numpy()
    else:
        originTargets=np.array(dataset.targets)

    if noisy_client>len(dict_users.keys()):
        print('too many noisy client')
        raise NameError('noisy_client')
        exit()
    noisyDataList = []
    all6Index = np.where(originTargets==6)[0]
    all2Index = np.where(originTargets==2)[0]
    noisyAddRelation={}
    for userIndex in range(noisy_client):
        for dataIndex in dict_users[userIndex]:
            if dataIndex in all2Index:
                maskDataIndex = np.random.choice(list(all6Index), 1, replace=False)
                dataset.train_data[dataIndex]=(1-noisy_rate)*dataset.train_data[dataIndex]+noisy_rate*dataset.train_data[maskDataIndex]
                noisyDataList.append(dataIndex)
                noisyAddRelation[dataIndex] = maskDataIndex

    return dataset, noisyDataList, noisyAddRelation


def generate_attack_data_mnist( dataset,  noisyAddRelation,noisy_rate):
    """
    将一部分用户数据中的2与6的数据混合，形成标签为2的新数据
    noisy_rate是加的噪音比例
    """
    dataset_train_noisy=copy.deepcopy(dataset)
    for key in noisyAddRelation.keys():
        # dataset_train_noisy.train_data[key]=torch.tensor((1-noisy_rate)*dataset_train_noisy.train_data.numpy()[key]+noisy_rate*dataset_train_noisy.train_data.numpy()[noisyAddRelation[key]])
        dataset_train_noisy.train_data[key] = torch.tensor(
            (1 - noisy_rate) * dataset_train_noisy.train_data.numpy()[key] + noisy_rate * dataset_train_noisy.train_data.numpy()[
                noisyAddRelation[key][0]])
    return dataset_train_noisy


def generate_attack_data_cifar( dataset,  noisyAddRelation,noisy_rate):
    """
    将一部分用户数据中的2与6的数据混合，形成标签为2的新数据
    noisy_rate是加的噪音比例
    """
    dataset_train_noisy=copy.deepcopy(dataset)
    for key in noisyAddRelation.keys():
        # dataset_train_noisy.train_data[key]=torch.tensor((1-noisy_rate)*dataset_train_noisy.train_data.numpy()[key]+noisy_rate*dataset_train_noisy.train_data.numpy()[noisyAddRelation[key]])
        dataset_train_noisy.train_data[key] = torch.tensor(
            (1 - noisy_rate) * dataset_train_noisy.train_data.numpy()[key] + noisy_rate * dataset_train_noisy.train_data.numpy()[
                noisyAddRelation[key][0]])
    return dataset_train_noisy

def generate_attack_data2(dataName, dict_users, dataset, noisy_client, noisy_rate):
    
    """
    将一部分client的数据的标签置换为错误标签，选取前noisy_client的client进行处理
    不需要返回值，直接修改了原始数据集，故调用后，如需原始数据集，需要再次读取
    """
    if dataName == 'mnist':
        originTargets = dataset.targets.numpy()
    else:
        originTargets = np.array(dataset.targets)
    allorigin_targets = set(originTargets)

    if noisy_client > len(dict_users):
        print('too many noisy client')
        raise NameError('noisy_client')
        exit()
    noisyDataList = []
    for userIndex in range(noisy_client):
        noisyDataList.extend(list(
            np.random.choice(list(dict_users[userIndex]), int(len(dict_users[userIndex]) * noisy_rate), replace=False)))

    for index in noisyDataList:
        all_targets = allorigin_targets
        all_targets = all_targets - set([originTargets[index]])
        new_label = np.random.choice(list(all_targets), 1, replace=False)
        originTargets[index] = new_label[0]
    dataset.targets = torch.tensor(originTargets)
    return dataset, noisyDataList
