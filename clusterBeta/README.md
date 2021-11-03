#                     **clusterBeta use Guide**
Federated learning and participant contribution evaluation system based on pytorch and socket (can be deployed on servers and various edge nodes).
 
 ## 
 
The federated optimization algorithm is [FedSGD](https://arxiv.org/pdf/1602.05629.pdf).
 
 
### Communication module
We encapsulated the Socket and implemented functions: send, rev, and broadcast. The details are in [deliver](https://github.com/qmkakaxi/DIG_FL/blob/master/clusterBeta/models/deliver.py).

  ### 数据分割(DataSplit.py):
 
 对已有数据进行分割：
 
 ```
 python DataSplit.py
 ```
 本实验共有14个client，故将数据集划分为14份。
  ### attackdata分割(attackDataSplit):
 
  ```
 python attackDataSplit.py
 ```

 本实验提供两种attackdata方式（封装在models.attackdata.py)：
 1. generate_attack_data1:
    将一部分用户数据中的2与6的数据混合，形成标签为2的新数据
 2. generate_attack_data2:
    将一部分client的数据的标签置换为错误标签

#### initialization
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
#### Example
Server send data A to client n. Server receive data B from client n.
  ```
server.send(A,id=n)        client_n.rec(id=0)
client_n.send(B,id=0)      server.rec(id=n)
  ```
### Train model and calculate contribution
#### Participant's local training
We use PyTorch to complete the participant's local training.
  ```
 # train client model
 lf=lossfunction
 for i in range(iter):
     epoch_loss = 0.0
     for data, target in train_set:
         data, target = data.to(device), target.to(device)
         data, target = Variable(data), Variable(target)
         optimizer.zero_grad()
         output = net(data)
         loss = lf(output, target)
         epoch_loss += loss.item()
         loss.backward()
         optimizer.step()
  ```
#### Participants send local gradients to server
  ```
  client_net = copy.deepcopy(net.state_dict())

  # tensor to list
  client_net = dict(client_net)
  for key in client_net:
      client_net[key] = client_net[key].cpu().numpy().tolist()

  data = {}
  data["net"] = client_net
  data["partyid"] = partyid
  client.send(data)
  ```
#### Server calculate per-ecpoh contribution
  ```
  # calculate contribution

   #list to tensor
   for i in range(len(w_local)):
       for key in w_local[i]:
           w_local[i][key] = torch.tensor((new_net[key])).to(device)
   w_glob = net.state_dict()

   DIG_FL(w_local, w_glob, net,dataset,device)
  ```
#### Server performs aggregation
  ```
  # aggregation
  if len(recDatas) > 1:
      new_net = merge([data["net"] for data in recDatas])
  else:
      new_net = recDatas[0]["net"]
  ```
#### Server sends back model update
  ```
 # send model updates to all client
 for i in range(world_size):
     server.send(new_net,id=i+1)
 ```

## Run Example
