import torch
import numpy as np
import collections
from math import ceil
from torch.autograd import Variable
import torch.nn.functional as F
from models.deliver import deliver
from torch.utils.data import DataLoader
from torch import nn
import copy

def merge(nets):
    """
    :param nets: client的网络参数列表。list of dictionary，维度不限，key不限，可适用于任意多个网络和任意维度的网络。
    :return: 合并后的网络参数，dictionary
    """
    """
    应对不同机器的网络参数的key 名字不同的情况：默认网络的 key不同时顺序仍然形同，可一一对应。
    """
    net = {}
    for i in range(len(nets[0].keys())):
        keys = [list(net.keys())[i] for net in nets]
        net[keys[0]] = np.mean([np.array(nets[j][keys[0]]) for j in range(len(nets))], axis=0)
        net[keys[0]] = net[keys[0]].tolist()
    return net

def cut_grad(w_local,w_g):
    grad = copy.deepcopy(w_local)
    for k in w_local.keys():
            grad[k] =w_g[k]-w_local[k]
    return grad


class calculate_gradient(object):
    def __init__(self, dataset=None):

        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.local_bs = 16
        self.ldr_train = DataLoader(dataset, batch_size=self.local_bs, shuffle=True)


    def calcluate(self,net,device):
        net.eval()

        loss_all = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_train):

            images, labels = images.to(device), labels.to(device)

            log_probs = net(images)
            # loss = self.loss_func(log_probs, labels)
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(log_probs, labels)
            loss.backward()

        grad_test = []
        for g in net.parameters():
            grad_test.append(g.grad)

        return grad_test


def DIG_FL(w_local, w_g, net,validation_dataset,device):

    contirbution_epoch = []
    cal_gradient_temp = calculate_gradient(dataset=validation_dataset)
    grad_test = cal_gradient_temp.calcluate(net=copy.deepcopy(net.to(device)),device=device)
    for i in range(len(w_local)):
        temp = cut_grad(w_local[i], w_g)
        net.load_state_dict(temp)
        grad_client=(list(net.parameters()))
        product = 0
        for (g,v) in zip(grad_client,grad_test):
            product += torch.sum(torch.mul(g,v))
        contirbution_epoch.append(max(product.cpu().item(),0))


    sum_contirbution = sum(contirbution_epoch)+0.00001
    for j in range(len(contirbution_epoch)):
        contirbution_epoch[j]=contirbution_epoch[j]/sum_contirbution

    print("contirbution_epoch:", contirbution_epoch)

    return contirbution_epoch


def DIGFL_learning(HOST,PORT, world_size, partyid, net,optimizer,dataset,
                      lossfunction=F.nll_loss,device=torch.device('cpu'),epoch=10,BUFSIZ=1024000000,batch_size=64,iter=5):

    """
    HOST:联邦学习server的ip
    PORT:端口号
    world_size:client的数量
    partyid:当前的id，id为0是server
    net:神经网络模型
    optimizer:神经网络训练优化器
    epoch:总训练的迭代次数
    device:训练选择的设备
    lossfunction:损失函数
    BUFSIZ:数据传输的buffer_size
    batch_size:神经网络训练的batch_size
    iter:每个client的内循环
    """



    # server
    if partyid == 0:

        server=deliver(HOST,PORT,partyid=partyid,world_size=world_size)

        for j in range(epoch):

            recDatas = []
            for i in range(world_size):
                recData=server.rec(id=i+1)
                recDatas.append(recData)

            # aggregation
            if len(recDatas) > 1:
                new_net = merge([data["net"] for data in recDatas])
            else:
                new_net = recDatas[0]["net"]


            # calculate contribution
            if len(recDatas) > 1:
                w_local = [data["net"] for data in recDatas]
                #list to tensor
                for i in range(len(w_local)):
                    for key in w_local[i]:
                        w_local[i][key] = torch.tensor((new_net[key])).to(device)
                w_glob = net.state_dict()
                DIG_FL(w_local, w_glob, net,dataset,device)

            # send model updates to all client
            for i in range(world_size):
                server.send(new_net,id=i+1)

            #  list to tensor
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
            new_net = collections.OrderedDict(new_net)

            # 加载参数到网络
            net.load_state_dict(new_net)
            if j == epoch - 1:
                return net



    # client
    else:

        client=deliver(HOST,PORT,partyid=partyid,world_size=world_size)

        train_set = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        optimizer = optimizer
        num_batches = ceil(len(train_set.dataset) / float(batch_size))

        for epoch in range(epoch):
            # train client network
            lf=lossfunction
            for i in range(iter):
                epoch_loss = 0.0
                for data, target in train_set:
                    data, target = data.to(device), target.to(device)
                    data, target = Variable(data), Variable(target)
                    optimizer.zero_grad()
                    output = net(data)
                    loss = lf(output, target)
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                if i == iter - 1:
                    print('partyid: ', partyid, ', epoch ', epoch, ': ', epoch_loss / num_batches)

            # 取出网络参数
            client_net = copy.deepcopy(net.state_dict())

            # tensor to list
            client_net = dict(client_net)
            for key in client_net:
                client_net[key] = client_net[key].cpu().numpy().tolist()

            # 拼接传输的数据内容
            data = {}
            data["net"] = client_net
            data["partyid"] = partyid

            client.send(data)

            new_net=client.rec()

            # 加载新的网络参数
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
            new_net = collections.OrderedDict(new_net)
            net.load_state_dict(new_net)
        return net

