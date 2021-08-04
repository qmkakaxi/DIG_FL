


import torch
if __name__=="__main__":

    path = "LinReg/house"
    weight = torch.load("data/{}/weight".format(path))
    loss_test = torch.load("data/{}/loss".format(path))
    delta_loss = []

    for i in range(1,len(loss_test)):
        temp=loss_test[i-1]-loss_test[i]
        delta_loss.append((temp))


    print(delta_loss)
    con=[]

    for i in range(len(weight[0])):
    # for i in range():
        temp=0
        for j in range(1):
            t = delta_loss[j]*weight[j][i]
            temp = temp+t

        con.append(temp)

    sum = 0.00001
    for j in range(len(con)):
        sum = sum+ con[j]
    for j in range(len(con)):
        con[j]= float(con[j]/sum)
    print("Shapley value from DIG-FL = ",con)
