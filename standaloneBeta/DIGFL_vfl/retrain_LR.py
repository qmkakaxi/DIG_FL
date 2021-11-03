import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def sigmoid(x):
    x = np.array(x, dtype = np.float64)
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    """
        Parameters:
        -----------
        n_iterations: int
        learning_rate: float
    """
    def __init__(self, learning_rate=0.0001, n_iterations=100,num_client=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.num_client = num_client

    def initialize_weights(self, X):

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



    def fit(self, X, y, X_test, y_test):
        # m_samples, n_features = X.shape
        m_samples = len(y)
        self.initialize_weights(X)

        y = np.reshape(y, (m_samples, 1))

        for i in range(self.n_iterations):
            # h_x = X.dot(self.w)

            #calculate train loss
            h_x = np.zeros(y.shape)
            for j in range(self.num_client):
                h_x = h_x+X[j].dot(self.w[j])
            y_pred = sigmoid(h_x)
            z = y_pred - y
            loss = np.mean(y * np.log(1+ np.exp(-y_pred)) + (1-y) * np.log(1+np.exp(y_pred)))

            #calculate test loss
            h_x = np.zeros(y_test.shape)
            for j in range(self.num_client):
                h_x = h_x+X_test[j].dot(self.w[j])
            y_pred = sigmoid(h_x)
            z_test = y_pred - y_test
            loss_test = np.mean(y_test * np.log(1+ np.exp(-y_pred)) + (1-y_test) * np.log(1+np.exp(y_pred)))

            for j in range(self.num_client):
                w_grad = X[j].T.dot(z)
                self.w[j] = self.w[j] - self.learning_rate * w_grad

            if i == self.n_iterations-1:
                return loss_test




def main(X_train, y_train, X_test, y_test):
    # Load dataset


    clf = LogisticRegression(num_client=len(X_train))
    loss = clf.fit(X_train, y_train,X_test,y_test)

    return loss


if __name__ == "__main__":
    path = "credit"
    # Load dataset
    with  open(r'data/LR/credit/data.pickle'.format(path),'rb')  as f1:
        data = pickle.load(f1)
    with  open(r'data/LR/credit/target.pickle'.format(path),'rb')  as f2:
        target = pickle.load(f2)
    data = preprocessing.scale(data)

    X_train,X_test, y_train, y_test = train_test_split(data,target,test_size=0.1, random_state=0)

    num_participant = 7
    X_train = np.split(np.array(X_train),(2,6,10,14,18,20,), axis=1)
    y_train = y_train

    X_test= np.split(np.array(X_test),(2,6,10,14,18,20,), axis=1)
    y_test = y_test

    loss = []
    for i in range(1,2**num_participant):
        b= bin(i+2**num_participant)
        idxs_users=[]
        for j in range(num_participant):
            if b[3+j]=="1":
                idxs_users.append(j)
        train_data = []
        test_data = []
        for j in (idxs_users):
            train_data.append(X_train[j])
            test_data.append(X_test[j])

        temp_1 = main(train_data,y_train,test_data,y_test)
        print("id:", idxs_users)
        print(" test loss:", temp_1)
        loss.append(temp_1)

    # print(loss)
    f1 = open(r"data/LR/credit/loss_retrain.pickle",'wb')
    pickle.dump(loss,f1)
    f1.close()

