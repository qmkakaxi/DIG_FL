import pandas as pd
import pickle


import numpy as np
if __name__ == "__main__":

    credit = pd.read_excel ("credit.xls")

    data = credit.iloc[:,1:-1]
    data_ = data.values
    data_ = np.delete(data_,0,0)

    target = credit.iloc[:,23:24]
    target_ = target.values
    target_ = np.delete(target_,0,0)

    f1 = open(r"data.pickle",'wb')
    pickle.dump(data_,f1)
    f1.close()

    f2 = open(r"target.pickle",'wb')
    pickle.dump(target_,f2)
    f2.close()


