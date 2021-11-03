import pickle
from itertools import combinations as cb
import numpy as np

def calculate_combinations(input):
    all_cb = []
    num = len(input)
    for j in range(1,num+1):
        for i in cb(input,j):
            temp=list(i)
            all_cb.append(temp)
    return all_cb

def list_to_index(up,input):
    up = up-1


    #去掉重复元素
    input = set(input)
    input = list(input)

    index = 0
    num = len(input)
    for i in range(num):
        index = index + 2**(up-input[i])

    return index-1


def marginal_contribution(i, S, reward, num_client):

    index = list_to_index(num_client, S)
    reward_1 = reward[index]

    S.append(i)
    index = list_to_index(num_client, S)
    reward_2 = reward[index]

    return reward_2-reward_1


def calculate_shapleyvalue(reward, num_client):
    shapleyvalue = []

    for i in range(num_client):
        temp = 0
        temp_list = []

        for j in range(num_client):
            if j != i :
                temp_list.append(j)

        #计算边际贡献
        all_cb = calculate_combinations(input=temp_list)
        for j in range(len(all_cb)):
            l=len(all_cb[j])
            mc=marginal_contribution(i, S=all_cb[j], reward=reward, num_client=num_client)
            mc=max(0,mc)
            temp = temp + 2**(l-1)*mc
        #加上自己的
        index = list_to_index(num_client,[i])
        temp = temp + 2**(num_client-2)*max(reward[index],0)
        temp=temp/120
        shapleyvalue.append(temp)

    #归一化
    sum = 0
    for j in range(len(shapleyvalue)):
        sum = sum+ shapleyvalue[j]
    for j in range(len(shapleyvalue)):
        shapleyvalue[j]= shapleyvalue[j]/sum

    return shapleyvalue

def MC_shapleyvalue(reward, num_client, num_sample):

    shapleyvalue = []

    for i in range(num_client):
        temp = 0
        temp_list = []

        for j in range(num_client):
            if j != i :
                temp_list.append(j)

        #计算边际贡献
        all_cb = calculate_combinations(input=temp_list)
        # print(len(all_cb))
        sample = np.random.choice(len(all_cb), num_sample, replace=False)
        t = 0
        for j in sample:
            l=len(all_cb[j])
            mc=marginal_contribution(i, S=all_cb[j], reward=reward, num_client=num_client)
            mc=max(0,mc)
            temp = temp + 2**(l-1)*mc
            t=t+2**(l-1)
        #加上自己的
        index = list_to_index(num_client,[i])
        temp = temp + 2**(num_client-2)*max(reward[index],0)
        temp=temp/t
        shapleyvalue.append(temp)

    #归一化
    sum = 0
    for j in range(len(shapleyvalue)):
        sum = sum+ shapleyvalue[j]
    for j in range(len(shapleyvalue)):
        shapleyvalue[j]= shapleyvalue[j]/sum

    return shapleyvalue

def calculate_shapleyvalue_2(reward, num_client):

    shapleyvalue = []

    for i in range(num_client):
        temp = 0
        temp_list = []

        for j in range(num_client):
            if j != i :
                temp_list.append(j)

        #计算边际贡献
        all_cb = calculate_combinations(input=temp_list)
        t = 0
        for j in range(len(all_cb)):
            l=len(all_cb[j])
            mc=marginal_contribution(i, S=all_cb[j], reward=reward, num_client=num_client)
            mc=max(0,mc)
            temp = temp + mc
        #加上自己的

        temp=temp/len(all_cb)
        shapleyvalue.append(temp)

    #归一化
    sum = 0
    for j in range(len(shapleyvalue)):
        sum = sum+ shapleyvalue[j]
    for j in range(len(shapleyvalue)):
        shapleyvalue[j]= shapleyvalue[j]/sum

    return shapleyvalue


if __name__=="__main__":

    path = "LinR/house"
    # path = "LR/credit"
    with  open(r'data/{}/loss_retrain.pickle'.format(path),'rb')  as f1:
        loss = pickle.load(f1)
    with  open(r'data/{}/loss_test.pickle'.format(path),'rb')  as f1:
        loss_test = pickle.load(f1)

    init_loss = loss_test[0]
    print(init_loss)
    reward = []
    for i in range(len(loss)):
        reward.append(init_loss-loss[i])
    print(reward)
    num_participant = 8

    shapleyvalue = calculate_shapleyvalue(reward, num_participant)
    print("actual shapley value = ",shapleyvalue)


