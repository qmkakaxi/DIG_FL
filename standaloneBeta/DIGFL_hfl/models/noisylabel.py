import numpy as np
import torch





def noisy_label_change_client(dataName, dict_users, dataset, noisy_client, noisy_rate):
    """
    change correct label into noisy label
    dataName:'MNIST' or 'cifar'
    """
    if dataName == 'MNIST':
        originTargets = dataset.train_labels.numpy()
    else:
        originTargets = dataset.targets
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
    return dataset, noisyDataList,torch.tensor(originTargets)







