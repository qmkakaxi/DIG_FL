import numpy as np
from sklearn import preprocessing
import pickle
from sklearn.model_selection import train_test_split


def sigmoid(x):
    x = np.array(x, dtype=np.float64)
    return 1 / (1 + np.exp(-x))

class LogisticRegression_DIGFL():
    """
        Parameters:
        -----------
        n_iterations: int

        learning_rate: float

    """
    def __init__(self, learning_rate=0.00001, n_iterations=100,num_client=1):
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
        # b = np.array(0)
        # self.w.append(b)

    def calculate_contribution(self,X,y,X_test,y_test):

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

        #calculate gradient of validation dataset
        y_test = np.reshape(y_test, (len(y_test), 1))
        y_pred = np.zeros(y_test.shape)
        for j in range(self.num_client):
            y_pred = y_pred+X_test[j].dot(self.w[j])


        y_pred = np.array(y_pred,dtype=np.float64)
        y_pred = sigmoid(y_pred)
        loss_test = np.mean(y_test * np.log(1+ np.exp(-y_pred)) + (1-y_test) * np.log(1+np.exp(y_pred)))
        z_test = y_pred - y_test

        #calculate gradient of all participants


        y_pred = np.zeros(y.shape)
        for j in range(self.num_client):
            y_pred = y_pred+X[j].dot(self.w[j])

        y_pred = np.array(y_pred,dtype=np.float64)
        y_pred = sigmoid(y_pred)
        z = y_pred - y
        grad = []
        for j in range(self.num_client):
            temp = z.T.dot(X[j])
            # print(z.shape)
            grad.append(temp)



        contribution_epoch = []
        for j in range(len(grad)):
            # w_ = ((grad[j].T.dot(grad_test[j])))
            c_temp = z_test.T.dot(X_test[j])*grad[j]

            contribution_epoch.append(sum((sum(c_temp))))

        # sum = (sum(contribution_epoch))
        # # print(we_sum)
        # if we_sum != 0 :
        #     for i in range(self.num_client):
        #         contribution_epoch[i] = contribution_epoch[i]/we_sum
        return contribution_epoch, loss_test

    def fit(self, X, y, X_test, y_test):
        # m_samples, n_features = X.shape
        m_samples = len(y)
        self.initialize_weights(X)

        y = np.reshape(y, (m_samples, 1))

        contribution = []
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
            contribution_epoch ,loss_test_ = self.calculate_contribution(X,y,X_test,y_test)
            contribution.append(contribution_epoch)
            loss_test.append(loss_test_)
            for j in range(self.num_client):
                w_grad = X[j].T.dot(z)
                self.w[j] = self.w[j] - self.learning_rate * w_grad
            f1 = open(r"data/LR/credit/contribution_epoch.pickle",'wb')
            pickle.dump(contribution,f1)
            f1.close()
            f2 = open(r"data/LR/credit/loss_test.pickle",'wb')
            pickle.dump(loss_test,f2)
            f2.close()
            # self.w = self.w - self.learning_rate * w_grad


    def predict(self, X,y):

        h_x = np.zeros(y.shape)
        for j in range(self.num_client):
            h_x = h_x+X[j].dot(self.w[j])
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)




def main():

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
    # divide 7 paricipants

    clf = LogisticRegression_DIGFL(num_client=num_participant)
    clf.fit(X_train, y_train,X_test,y_test)


if __name__ == "__main__":
    path = "credit"
    main()
