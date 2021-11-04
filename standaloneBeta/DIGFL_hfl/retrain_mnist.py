#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :main_fed_mnist_retrain.py.py
# @Time      :2020/12/26 3:35 下午
# @Author    :wangjunhao





import copy
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid
from utils.options import args_parser
from models.Update import LocalUpdate,DatasetSplit
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg,FedAvg_weight
from models.test import test_img
from models.noisylabel import noisy_label_change_client



def train(idxs_users):
    # parse args
    args = args_parser()

    args.epochs=20
    args.num_channels=1
    args.local_bs=4
    settime=0
    args.gpu=3
    args.local_ep=1
    args.num_users=10



    #设置cpu
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('data/', train=True, download=True,transform=trans_mnist)

    dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)

    dict_users=mnist_iid(dataset_train,args.num_users)


    #Generate low-contributing participants
    dataset_train,noisyDataList,newTargets = noisy_label_change_client('MNIST', dict_users, dataset_train, 1, 0.5)


    print("begin train...")
    net_train=CNNMnist(args=args).to(args.device)
    print(net_train)
    net_train.train()



    for iter in range(args.epochs):
        w_locals, loss_locals = [], []

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_train).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_train
        net_train.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        if iter==args.epochs-1:

            net_train.eval()
            train_accuracy,train_loss = test_img(net_train, dataset_train, args)
            test_accuracy,test_loss = test_img(net_train, dataset_test, args)
            print("Training accuracy: {:.2f}".format(train_accuracy))
            print("Training loss: {:.2f}".format(train_loss))

            print("Testing accuracy: {:.2f}".format(test_accuracy))
            print("Testing loss: {:.2f}".format(test_loss))
    return test_accuracy,test_loss

if __name__=='__main__':
    args = args_parser()
    acc=[]
    loss=[]
    num_participant = 10
    for i in range(1,2**num_participant):
        b= bin(i+32)
        idxs_users=[]
        for j in range(num_participant):
            if b[3+j]=="1":
                idxs_users.append(j)
        temp_1,temp_2=train(idxs_users)
        print("id:", idxs_users)
        print(" test acc:", temp_1)
        print(" test loss:", temp_2)
        acc.append(temp_1)
        loss.append(temp_2)



