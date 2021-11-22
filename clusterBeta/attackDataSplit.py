from torchvision import datasets, transforms
from utils.sampling import mnist_iid
import torch
from models.attackdata import generate_attack_data1,generate_attack_data2

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
       self.data = data
       # self.index = index
       self.index=list(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]



if __name__=='__main__':


     # Divide the dataset
    num_participants=2
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('data/', train=True, download=True, transform=trans_mnist)
    dict_users = mnist_iid(dataset_train, num_participants)
    noisy_participant=1
    noisy_rate=0.5
    dataset_attack,noisylist=generate_attack_data2('mnist', dict_users, dataset_train, noisy_participant, noisy_rate)
    torch.save(dict_users, 'dict_user')
    torch.save(noisylist,'noisylist')


    for i in range(num_participants):
        partition = Partition(dataset_attack, dict_users[i])
        torch.save(partition,'data_of_client{}'.format(i+1))
