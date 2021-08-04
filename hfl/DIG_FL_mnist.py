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
from models.noisylabel import noisy_label_change_client, noisy_label,noisy_label_
from utils.sampling import set_user
import pickle
from models.attackdata import generate_attack_data_mnist
from models.noisylabel import Gaussian_noise,Gaussian_noise_dict
import numpy

def cal_grad(w_local,w_g):
    grad = copy.deepcopy(w_local)
    for k in w_local.keys():
            grad[k] =w_g[k]-w_local[k]
            grad[k]=grad[k].cpu().numpy()
    return grad


#计算每个client的权重，使用进行梯度下降
def cal_weight(w_local, w_g, net):

    grad_client = []
   #计算client的本地梯度
    for i in range(len(w_local)):
        grad_client.append(cal_grad(w_local[i],w_g))

    net.load_state_dict(w_g)
    grad_weight = []
    #求测试数据的导数
    test_update = LocalUpdate(args=args,dataset=dataset_train,idxs=dict_users[10])
    w_t,_ = test_update.train(net=copy.deepcopy(net.to(args.device)))

    grad_test = (cal_grad(w_t,w_g))
    # print(grad_test)
    #计算权重
    for j in range(len(w_local)):
        temp=0
        s=grad_client[j]
        t=grad_test
        for k in s.keys():
            temp=temp+np.sum(s[k] * t[k])
        grad_weight.append(max(temp,0))

    weight=grad_weight
    sum_weight=sum(weight)+0.0001
    for j in range(len(weight)):
        weight[j]=weight[j]/sum_weight
    print("weight:", weight)

    return weight


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
    dataset_train = datasets.MNIST('data/', train=True, download=True,transform=trans_mnist)
    dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)

    dict_users=mnist_iid(dataset_train,args.num_users)


    print("begin train...")
    net_train=CNNMnist(args=args).to(args.device)

    print(net_train)
    net_train.train()
    net_test=CNNMnist(args=args).to(args.device)

    w_glob = net_train.state_dict()
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    grad_client={0:[],1:[]}
    grad_test=[]
    weights=[]

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []

        idxs_users= [0,1,2,3,4,5,6,7,8,9]

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_train).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))


        weight = cal_weight(w_locals, w_glob, net_train)
        weights.append(weight)

        w_glob = FedAvg(w_locals)
        # copy weight to net_train
        net_train.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)


        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        test_accuracy,test_loss = test_img(net_train, dataset_test, args)

        print("Testing accuracy: {:.2f}".format(test_accuracy))
        print("Testing loss: {:.4f}".format(test_loss))
        loss_test.append(test_loss)
        acc_test.append(test_accuracy)

        if iter==args.epochs-1:
            w_wag=net_train.state_dict()

            net_train.eval()
            train_accuracy,train_loss = test_img(net_train, dataset_train, args)

            print("Training accuracy: {:.2f}".format(train_accuracy))
            print("Training loss: {:.2f}".format(train_loss))


