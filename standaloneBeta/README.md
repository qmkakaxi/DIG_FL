
#                     **standaloneBeta use Guide**

StandaloneBeta simulates federated learning in a standalone case to test how close our method is to the actual shapley value.


### DIG-FL V.S. Actual Shapley value for HFL.

In [DIG_hfl](https://github.com/qmkakaxi/DIG_FL/tree/master/standaloneBeta/DIGFL_hfl)
#### calculate estimated Shapley value

 ```shell
 $python DIG_FL_mnist.py.
 ```
 ### calculate Actual Shapley value
 
Calculate the utility function of all participants through retraining.
  ```shell
 $python retrain_mnist.py
 ```
 
### DIG-FL V.S. Actual Shapley value for VFL.
