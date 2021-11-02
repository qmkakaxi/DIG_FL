#计算每个client的权重，使用进行梯度下降
import torch
from torch.utils.data import DataLoader
from torch import nn
import copy

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
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

        # train and update
        loss_all = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_train):

            images, labels = images.to(device), labels.to(device)
            #初始化
            #训练
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
        # grad_client.append(list(net.parameters()))
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
