from torchvision import datasets, transforms
import torch
from utils.options import args_parser
from models.Nets import CNNMnist, CNNCifar
from models.test import test
import torch.nn.functional as F
from models.FederatedLearning import FederatedLearning


class Partition(object):
	""" Dataset-like object, but only access a subset of it. """

	def __init__(self, data, index):
		self.data = data
		# self.index = index
		self.index = list(index)

	def __len__(self):
		return len(self.index)

	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]


if __name__ == '__main__':

    args = args_parser()
    data = []
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    #
    HOST=args.HOST
    PORT=args.PORT_
    world_size=2
    net = CNNMnist().to(device)
    optimizer=torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.5)
    lossfunction=F.nll_loss
    net=FederatedLearning(HOST=HOST,PORT=PORT, world_size=world_size, partyid=0, net=net,optimizer=optimizer,
                      dataset=data,lossfunction=lossfunction,device=device)


    # 
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)
    test_set = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs)
    args.device=device
    test_accuracy, test_loss = test(net, test_set, args)
    print("Testing accuracy: {:.2f}".format(test_accuracy))
    print("Testing loss: {:.2f}".format(test_loss))

    #
    w_wag=net.state_dict()
    torch.save(w_wag,'w_wag')
