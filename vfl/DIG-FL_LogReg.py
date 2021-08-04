import numpy as np
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch
# Import helper functions
from utils import make_diagonal, normalize, train_test_split, accuracy_score
from utils import Plot
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer, load_wine
def sigmoid(x):
    x = np.array(x, dtype=np.float64)
    return 1 / (1 + np.exp(-x))
import time

class LogisticRegression():
    """
        Parameters:
        -----------
        n_iterations: int
            梯度下降的轮数
        learning_rate: float
            梯度下降学习率
    """
    def __init__(self, learning_rate=0.00001, n_iterations=100,num_client=1):
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

    def calculate_weights(self,X,X_test,y_test):

        if self.num_client >1 :
            X_c = np.concatenate(X,axis=1)
            X_test_c = np.concatenate(X_test,axis=1)
            # w = np.concatenate(self.w,axis=0)
            t = [ self.w[i] for i in range(self.num_client)]
            w = np.concatenate(t,axis=0)
        else:
            X_c = X[0]
            X_test_c = X_test[0]
            w = self.w[0]
        #计算测试数据集的梯度
        y_test = np.reshape(y_test, (len(y_test), 1))
        y_pred = np.zeros(y_test.shape)
        for j in range(self.num_client):
            y_pred = y_pred+X_test[j].dot(self.w[j])
        # loss = np.mean(0.5 * (y_pred -y+self.w[self.num_client]) ** 2)  #计算loss
        y_pred = np.array(y_pred,dtype=np.float64)
        y_pred = sigmoid(y_pred)
        z = y_pred - y_test
        #计算各个client的梯度
        grad = []
        for j in range(self.num_client):
            temp = z.T.dot(X[j])
            # print(z.shape)
            grad.append(temp)
        # print("z",z)
        loss_test = np.mean(y_test * np.log(1+ np.exp(-y_pred)) + (1-y_test) * np.log(1+np.exp(y_pred)))
        # print("test loss:",loss_test)
        # grad_test = []
        # for j in range(self.num_client):
        #     w_grad = X_test[j].T.dot(z)
        #     grad_test.append(w_grad)
        y_pred = X_test_c.dot(w)
        y_pred = sigmoid(y_pred)
        z = y_pred - y_test
        grad_test = []

        #计算每个用户的权重
        we = []
        for j in range(len(grad)):
            # w_ = ((grad[j].T.dot(grad_test[j])))
            w_ = z.T.dot(X[j])*grad[j]

            we.append(max((sum(w_))))


        we_sum = (sum(we))
        # print(we_sum)
        if we_sum != 0 :
            for i in range(self.num_client):
                we[i] = we[i]/we_sum
        return we, loss_test

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
            h_x = np.array(h_x,dtype=np.float64)
            y_pred = sigmoid(h_x)
            z = y_pred - y
            loss = np.mean(y * np.log(1+ np.exp(-y_pred)) + (1-y) * np.log(1+np.exp(y_pred)))
            print("iierations:",i,"train loss:",loss)
            w ,loss_test_ = self.calculate_weights(X,X_test,y_test)
            weight.append(w)
            loss_test.append(loss_test_)
            for j in range(self.num_client):
                w_grad = X[j].T.dot(z)
                self.w[j] = self.w[j] - self.learning_rate * w_grad
            # self.w = self.w - self.learning_rate * w_grad
        torch.save(weight,"data/LogReg/{}/weight".format(path))
        torch.save(loss_test,"data/LogReg/{}/loss".format(path))

    def predict(self, X,y):

        h_x = np.zeros(y.shape)
        for j in range(self.num_client):
            h_x = h_x+X[j].dot(self.w[j])
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)




def main():
    # Load dataset

    X = torch.load("data/LogReg/credit/data")
    y = torch.load("data/LogReg/credit/target")
    X = preprocessing.scale(X)

    X = np.split(X,(2,6,10,14,18,20,), axis=1)
    clf = LogisticRegression(num_client=len(X))
    clf.fit(X, y,X,y)


if __name__ == "__main__":
    path = "credit"
    main()
