# Setup Once, Secure Always: A Single-Setup Secure Federated Learning Aggregation Protocol with Forward and Backward Secrecy for Dynamic Users

### Prerequisites

* python:3.7.11
* install required python packages from requirements.txt

### Code Structure

- flapp.py: the performance testing harness
- local_update.py: the implementation of users joining the federated learning
- fognode.py: the implementation of intermediate server and aggregation server
- lenet5.py: the model definition to train on MNIST and CIFAR-10 datasets.
- other files: utility functions used by the performance testing
  
### How to Run

in the 'src' folder, run
```bash
python flapp.py
```

### Configurations
The following configurations can be modified in the config.json file:

* "num_users": the number of users joining the federated learning
* "num_fognode": the number of intermediate servers
* "data_size": input vector size
* "dropout": the percentage of users dropping out after every iteration
* "epochs": the number of iterations to run the federated learning between the users, intermediate servers, and aggregators


