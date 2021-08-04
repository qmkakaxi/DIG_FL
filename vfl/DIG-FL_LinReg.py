import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



class LinearRegression():

    def __init__(self, n_iterations=3000, learning_rate=0.00001, num_participant=1, gradient=True):
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
        for i in range(self.num_participant):
            _,n_feature = X[i].shape
            w = np.random.uniform(0, 0, (n_feature, 1))
            self.w.append(w)

    def calculate_weights(self,X,X_test,y,y_test):

        if self.num_participant >1 :
            X_c = np.concatenate(X,axis=1)
            X_test_c = np.concatenate(X_test,axis=1)
            t = [ self.w[i] for i in range(self.num_participant)]
            w = np.concatenate(t,axis=0)
        else:
            X_c = X[0]
            X_test_c = X_test[0]
            w = self.w[0]
        y_test = np.reshape(y_test, (len(y_test), 1))
        y_pred = np.zeros(y_test.shape)
        for j in range(self.num_participant):
            y_pred = y_pred+X_test[j].dot(self.w[j])
        z = y_pred - y_test

        grad = []
        for j in range(self.num_participant):
            temp = z.T.dot(X_test[j])
            grad.append(temp)
        loss_test = np.mean(0.5*(z)**2)

        y_pred = X_c.dot(w)
        z = y_pred - y

        we = []
        for j in range(len(grad)):
            w_ = z.T.dot(X[j])*grad[j]
            we.append(sum(sum(w_)))

        we_sum = (sum(we))

        if we_sum != 0 :
            for i in range(self.num_participant):
                we[i] = we[i]/we_sum
        print("weights:",we)
        return we, loss_test


    def fit(self, X, y, X_test, y_test):
        m_samples = len(y)
        print(m_samples)
        self.initialize_weights(X)
        y = np.reshape(y, (m_samples, 1))
        self.training_errors = []

        weight = []
        loss_test = []
        if self.gradient == True:
            for i in range(self.n_iterations):

                y_pred = np.zeros(y.shape)
                for j in range(self.num_participant):
                    y_pred = y_pred+X[j].dot(self.w[j])

                loss = np.mean(0.5 * (y_pred -y )** 2)
                z = y_pred - y
                print("iteration: ",i," train loss :",loss)
                self.training_errors.append(loss)
                w ,loss_test_ = self.calculate_weights(X,X_test,y,y_test)
                weight.append(w)
                loss_test.append(loss_test_)

                for j in range(self.num_participant):
                    w_grad = X[j].T.dot(z)
                    self.w[j] = self.w[j] - self.learning_rate * w_grad
            torch.save(weight,"data/LinReg/house/weight")
            torch.save(loss_test,"data/LinReg/house/loss")
        else:
            # 正规方程
            X = np.matrix(X)
            y = np.matrix(y)
            X_T_X = X.T.dot(X)
            X_T_X_I_X_T = X_T_X.I.dot(X.T)
            X_T_X_I_X_T_X_T_y = X_T_X_I_X_T.dot(y)
            self.w = X_T_X_I_X_T_X_T_y




def main():
    
    wine_quality = pd.read_csv('data/LinReg/house/house_data.csv')

    wine_quality = wine_quality.drop(index=[0])

    num_participant = 8

    data = wine_quality.iloc[:,:-1]
    target = wine_quality["Price"]
    data = preprocessing.scale(data)
    target = np.array(target)

    X_train,X_test, y_train, y_test = train_test_split(data,target,test_size=0.1, random_state=0)

    X_train = np.split(np.array(X_train),num_participant, axis=1)
    y_train = y_train

    X_test= np.split(np.array(X_test),num_participant, axis=1)
    y_test = y_test

    model = LinearRegression(n_iterations=1200, num_participant=num_participant)

    model.fit(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    
    main()
