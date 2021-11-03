##                     **DIGFL use Guide**
Federated learning and participant contribution evaluation system based on pytorch and socket (can be deployed on servers and various edge nodes).
 
 ### Federatedlearning
 
The federated optimization algorithm is [FedSGD](https://arxiv.org/pdf/1602.05629.pdf).
 
 
#### Communication module
We encapsulated the Socket and implemented functions: send, rev, and broadcast. The details are in [deliver](https://github.com/qmkakaxi/DIG_FL/blob/master/clusterBeta/models/deliver.py).

##### initialization
  ```
  Host: ip of server
  port: available ports
  partyid: id of this machine, id 0 is the server
  world_size: num of participant
  ```
  ```
 server=deliver(HOST,PORT,partyid=0,world_size=world_size)
  ```
  ```
 client=deliver(HOST,PORT,partyid=partyid,world_size=world_size)
  ```
##### Example
Server send data A to client n. Server receive data B from client n.
  ```
server.send(A,id=n)        client_n.rec(id=0)
client_n.send(B,id=0)      server.rec(id=n)
  ```

