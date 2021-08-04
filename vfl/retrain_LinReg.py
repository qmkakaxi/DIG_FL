import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing




class LinearRegression():


    def __init__(self, n_iterations=2000, learning_rate=0.00001, num_participant=1, gradient=True):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.num_participant = num_participant



    def initialize_weights(self, X):

        n_features = 0
        for i in range(self.num_participant):
            _,temp = X[i].shape
            n_features = n_features +temp
        self.w = []
        limit = np.sqrt(1 / n_features)
        for i in range(self.num_participant):
            _,n_feature = X[i].shape
            w = np.random.uniform(0, 0, (n_feature, 1))
            self.w.append(w)
        b = np.array(0)
        self.w.append(b)

    def fit(self, X, y, X_test, y_test):

        self.initialize_weights(X)

        y = np.reshape(y, (len(y), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))
        self.training_errors = []

        if self.gradient == True:

            for i in range(self.n_iterations):


                y_pred = np.zeros(y.shape)
                for j in range(self.num_participant):
                    y_pred = y_pred+X[j].dot(self.w[j])

                train_loss = np.mean(0.5 * (y_pred -y) ** 2)  #计算loss
                z = y_pred - y + self.w[self.num_participant]

                y_pred = np.zeros(y_test.shape)
                for j in range(self.num_participant):
                    y_pred = y_pred+X_test[j].dot(self.w[j])
                test_loss = np.mean(0.5 * (y_pred -y_test) ** 2)  #计算loss

                if i == self.n_iterations-1:
                    print("iteration: ",i," train loss :",train_loss)
                    print("test loss :",test_loss)
                    return test_loss

                for j in range(self.num_participant):
                    w_grad = X[j].T.dot(z)
                    self.w[j] = self.w[j] - self.learning_rate * w_grad

                d_grad = np.ones([len(y),1]).T.dot(z)

                self.w[self.num_participant] = self.w[self.num_participant] - self.learning_rate * d_grad

        else:
            # 正规方程
            X = np.matrix(X)
            y = np.matrix(y)
            X_T_X = X.T.dot(X)
            X_T_X_I_X_T = X_T_X.I.dot(X.T)
            X_T_X_I_X_T_X_T_y = X_T_X_I_X_T.dot(y)
            self.w = X_T_X_I_X_T_X_T_y





def main(X_train, y_train, X_test, y_test):

    # 可自行设置模型参数，如正则化，梯度下降轮数学习率等
    model = LinearRegression(n_iterations=1200, num_participant=len(X_train))

    print(len(X_train))
    test_loss = model.fit(X_train, y_train, X_test, y_test)
    return test_loss


if __name__ == "__main__":

    # boston = load_boston()
    # boston = load_diabetes()
    num_participant = 8
    wine_quality = pd.read_csv('data/LinReg/house/house_data.csv')

    data = wine_quality.iloc[:,:-1]
    target = wine_quality["Price"]
    data = preprocessing.scale(data)
    target = np.array(target)

    X_train = np.split(data,(num_participant), axis=1)
    y_train = target

    X_test= X_train
    y_test = y_train

    num_participant = 8
    loss_retrain = []
    for i in range(1,2**num_participant):
        b= bin(i+2**num_participant)
        idxs_users=[]
        for j in range(num_participant):
            if b[3+j]=="1":
                idxs_users.append(j)
        train_data = []
        # test_data = []
        for j in (idxs_users):
            train_data.append(X_train[j])
            test_data = train_data


        temp_1 = main(train_data,y_train,test_data,y_test)
        print("id:", idxs_users)
        print(" test loss:", temp_1)
        loss_retrain.append(temp_1)

    # print(loss)
    torch.save(loss_retrain,"data/LinReg/house/loss_retrain")

