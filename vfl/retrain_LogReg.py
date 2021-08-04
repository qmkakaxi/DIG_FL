

import numpy as np
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
# Import helper functions
from utils import make_diagonal, normalize, train_test_split, accuracy_score
from utils import Plot
import torch
from sklearn import preprocessing
import time

def sigmoid(x):
    x = np.array(x, dtype = np.float64)
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    """
        Parameters:
        -----------
        n_iterations: int
            梯度下降的轮数
        learning_rate: float
            梯度下降学习率
    """
    def __init__(self, learning_rate=0.0001, n_iterations=100,num_client=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.num_client = num_client

    def initialize_weights(self, X):
        # 初始化参数
        # 参数范围[-1/sqrt(N), 1/sqrt(N)]
        # limit = np.sqrt(1 / n_features)
        # w = np.random.uniform(-limit, limit, (n_features, 1))
        # b = 0
        # self.w = np.insert(w, 0, b, axis=0)
        n_features = 0
        for i in range(self.num_client):
            # _,temp = X[i].shape
            temp = len(X[i])
            n_features = n_features +temp
        self.w = []
        # limit = np.sqrt(1 / n_features)
        for i in range(self.num_client):
            _,n_feature = X[i].shape
            w = np.random.uniform(0, 0, (n_feature, 1))
            # w = np.random.uniform(0, 0, (n_feature, 1))
            self.w.append(w)
        # b = np.array(0)
        # self.w.append(b)


    def fit(self, X, y, X_test, y_test):
        # m_samples, n_features = X.shape
        m_samples = len(y)
        self.initialize_weights(X)
        # 为X增加一列特征x1，x1 = 0
        # X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))

        # 梯度训练n_iterations轮
        weight = []
        loss_test = []
        for i in range(self.n_iterations):
            # h_x = X.dot(self.w)
            h_x = np.zeros(y.shape)
            for j in range(self.num_client):
                h_x = h_x+X[j].dot(self.w[j])
            y_pred = sigmoid(h_x)
            z = y_pred - y
            loss = np.mean(y * np.log(1+ np.exp(-y_pred)) + (1-y) * np.log(1+np.exp(y_pred)))
            # print("iierations:",i,"train loss:",loss)
            # w ,loss_test_ = self.calculate_weights(X,X_test,y_test)
            # weight.append(w)
            # loss_test.append(loss_test_)
            for j in range(self.num_client):
                w_grad = X[j].T.dot(z)
                self.w[j] = self.w[j] - self.learning_rate * w_grad
            if i == self.n_iterations-1:
                return loss
    # def predict(self, X):
    #     X = np.insert(X, 0, 1, axis=1)
    #     h_x = X.dot(self.w)
    #     y_pred = np.round(sigmoid(h_x))
    #     return y_pred.astype(int)




def main(X_train, y_train, X_test, y_test):
    # Load dataset


    clf = LogisticRegression(num_client=len(X_train))
    loss = clf.fit(X_train, y_train,X_test,y_test)
    # y_pred = clf.predict(X)
    # y_pred = np.reshape(y_pred, y.shape)
    #
    # accuracy = accuracy_score(X, y_pred)
    # print("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    # Plot().plot_in_2d(X, y_pred, title="Logistic Regression", accuracy=accuracy)
    return loss

if __name__ == "__main__":
    path = "credit"
    # data = datasets.load_iris()
    # # X = normalize(data.data[data.target != 0])
    # X = preprocessing.scale(data.data[data.target != 0])
    # y = data.target[data.target != 0]
    # train = np.split(X,(4), axis=1)
    # y[y == 1] = 0
    # y[y == 2] = 1
    loss=[]
    # cancer = load_breast_cancer()
    cancer = load_wine()
    # X = cancer['data']
    # # print(X[0])
    # y = cancer['target']
    X = torch.load("data/LogReg/credit/data")
    y = torch.load("data/LogReg/credit/target")
    # X = preprocessing.scale(X)
    # train = np.split(X,15, axis=1)
    # X = torch.load("data/LogReg/{}/data".format(path))
    # y =  torch.load("data/LogReg/{}/target".format(path))
    # X = X.values
    X = preprocessing.scale(X)
    #
    # y = y.values
    train = np.split(X,(2,6,10,14,18,20,), axis=1)
    print(len(train))
    s = time.time()
    num_feature = 7

    for i in range(1,2**num_feature):
        b= bin(i+2**num_feature)
        idxs_users=[]
        for j in range(num_feature):
            if b[3+j]=="1":
                idxs_users.append(j)
        train_data = []
        # test_data = []
        for j in (idxs_users):
            train_data.append(train[j])
            test_data = train_data
            # test_data.append(test[j])

        temp_1 = main(train_data,y,test_data,y)
        print("id:", idxs_users)
        print(" test loss:", temp_1)
        loss.append(temp_1)
        # if i >=2**7:
        #     break
    e = time.time()
    print("cost time:",e-s)
    print(loss)
    torch.save(loss,"data/LogReg/{}/loss_retrain_7".format(path))
