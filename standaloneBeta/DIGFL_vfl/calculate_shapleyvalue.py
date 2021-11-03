import pickle

if __name__=="__main__":

    # path = "LinR/house"
    path = "LR/credit"
    with  open(r'data/{}/contribution_epoch.pickle'.format(path),'rb')  as f1:
     contribution_epoch = pickle.load(f1)

    con=[]


    for i in range(len(contribution_epoch[0])):
    # for i in range():
        temp=0
        for j in range(1):
            t = contribution_epoch[j][i]
            temp = temp+t
        con.append(temp)
    sum = 0.00001
    for j in range(len(con)):
        sum = sum+ con[j]
    for j in range(len(con)):
        con[j]= float(con[j]/sum)
    print("Shapley value from DIG-FL = ",con)
