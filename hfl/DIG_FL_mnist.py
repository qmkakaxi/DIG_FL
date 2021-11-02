import copy
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid
from utils.options import args_parser
from models.Update import LocalUpdate,DatasetSplit
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg,FedAvg_weight
from models.test import test_img
from models.DIG_FL import DIG_FL
from models.noisylabel import noisy_label_change_client



if __name__ == '__main__':
    # parse args
    args = args_parser()

    # 设置参数
    args.epochs=20
    args.num_channels=1
    args.local_bs=4
    settime=0
    args.gpu=3
    args.local_ep=1
    args.num_users=5


    #设置cpu
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('data/', train=True, download=True,transform=trans_mnist)
    dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)

    dict_users=mnist_iid(dataset_train,args.num_users)


    #Generate low-contributing participants
    dataset_train,noisyDataList,newTargets = noisy_label_change_client('MNIST', dict_users, dataset_train, 1, 0.5)

    print("begin...")
    net_train=CNNMnist(args=args).to(args.device)

    print(net_train)
    net_train.train()


    w_glob = net_train.state_dict()
    contributions=[]

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []

        idxs_users= [i for i in range(args.num_users)]

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_train).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))


        contribution_epoch = DIG_FL(w_locals, w_glob, net_train,dataset_test,args.device)
        contributions.append(contribution_epoch)

        w_glob = FedAvg(w_locals)
        # copy weight to net_train
        net_train.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)


        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        test_accuracy,test_loss = test_img(net_train, dataset_test, args)

        print("Testing accuracy: {:.2f}".format(test_accuracy))
        print("Testing loss: {:.4f}".format(test_loss))


        if iter==args.epochs-1:
            w_wag=net_train.state_dict()

            net_train.eval()
            train_accuracy,train_loss = test_img(net_train, dataset_train, args)

            print("Training accuracy: {:.2f}".format(train_accuracy))
            print("Training loss: {:.2f}".format(train_loss))


