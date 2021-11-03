##                     **DIGFL use Guide**
Federated learning and participant contribution evaluation system based on pytorch and socket (can be deployed on servers and various edge nodes).
 
 ### Federatedlearning
 
The federated optimization algorithm is [FedSGD](https://arxiv.org/pdf/1602.05629.pdf).
 
 
#### Communication module
We encapsulated the Socket and implemented functions such as send rev and broadcast.

##### initialization
  ```
 server=deliver(HOST,PORT,partyid=partyid,world_size=world_size)
  ```
