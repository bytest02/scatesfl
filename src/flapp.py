import random
import copy
import time
from datetime import datetime
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

import numpy as np

from local_update import LocalUpdate
from fognode import FogNode, AggregatorNode
from sign import generate_sign_key_pair
from utils import read_config, send_telegram_message
from lenet5 import create_LeNet5_model
from dataset import get_datasets, iid_distribute_dataset_to_users
from evaluate import evaluate
from perf_collector import PerfCollector
from gen_report import gen_report

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="{asctime} - {levelname}: {message}", 
                    style='{', datefmt="%Y-%m-%d %H:%M:%S")

# Assigning the seed value for the random function
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Checking if GPU is available or not
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print("GPU runtime selected, GPU device name:", torch.cuda.get_device_name())
else:
  print("No GPU runtime, running on CPU mode")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = 'CIFAR10' # 'CIFAR10' or 'MNIST'

# Number of clients available at a time to collaborate for federated learning
# If frac =1, total available clients will be engaged, if frac =0.5, only 50% of the total clients will be engaged, where clients' selection is random
frac = 1

# Learning rate for FL model training
lr = 0.004

dataset_train, dataset_test = get_datasets(dataset_name)
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")

### Server-side functions
channels = 1 if dataset_name == 'MNIST' else 3
net_test = create_LeNet5_model(device, channels)
print(net_test)      

### Main app
def start_learning(config, context):
    verified_results = None # the list of client-ids verified by every fog node and the aggregator
    num_users = config['num_users']
    num_fognode = config['num_fognode']
    epochs = config['epochs']

    # Distributed clients and their dataset in dict_user (for training) and dict_users_test (for inference)
    dict_users = iid_distribute_dataset_to_users(dataset_train, num_users)
    dict_users_test = iid_distribute_dataset_to_users(dataset_test, num_users)

    # set names for all the entities
    client_names = [f'client{i+1}' for i in range(num_users)]
    fog_names = [f'fog{i+1}' for i in range(num_fognode)]
    server_name = 'aggregator'
    collector_names = [*fog_names, server_name]

    # generate RSA key pairs for all entities
    private_keys = {}
    public_keys = {}
    for n in client_names:
        pub, priv = generate_sign_key_pair()
        private_keys[n] = priv
        public_keys[n] = pub.format()
    for n in collector_names:
        pub, priv = generate_sign_key_pair()
        private_keys[n] = priv
        public_keys[n] = pub.format()

    fog_map = {}

    for name in fog_names:
        fog_map[name] = FogNode(name, private_keys[name], public_keys)
    aggregator_node = AggregatorNode(server_name, private_keys[server_name], public_keys)
    fog_map[server_name] = aggregator_node 

    theta = None # the intermediate and precise value of global weights 

    verified_results = None # the list of client-ids verified by every fog node and the aggregator
    publishing_Si_Ri = {} # the verified Si and Ri published by fog nodes to the clients
    all_idxs = work_idxs = [i for i in range(num_users)]
    for iter in range(epochs):
        for n in collector_names:
            fog_map[n].clear_collected()    
        # The following for loop (sequential) implementation of the local training among the clients (we can also parallalize the process by using python threads)
        if config['dropout'] > 0.01:
            work_idxs = np.random.choice(all_idxs, int(num_users*(1-config['dropout'])), replace = False)
        for idx in work_idxs:
            name = client_names[idx]
            
            p_start = time.time_ns() 
            local = LocalUpdate(idx, name, lr, device, dataset_train = dataset_train,
                                    dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx],
                                    private_key = private_keys[name], config = config )
            context['perf'].collect({'iter': iter, 'rid': 'pu1', 'v': time.time_ns() - p_start})
            w = None
            # Training ------------------
            if iter == 0:         
                p_start = time.time_ns() 
                w, loss_train, acc_train = local.initial_train_and_mask( create_LeNet5_model(device, channels), iter=iter,
                                                                            collectors=collector_names )
                context['perf'].collect({'iter': iter, 'rid': 'pu2', 'v': time.time_ns() - p_start})
            else:
                # simulating this client received <R_i, S_i, A, I, F_j> from verified-by fog nodes
                if name in publishing_Si_Ri['intersected']:
                    p_start = time.time_ns() 
                    w, loss_train, acc_train = local.train_and_mask( theta, publishing_Si_Ri['msgs'], iter=iter, collectors=collector_names, 
                                                                        channels=channels, perfcollector= context['perf'])
                    context['perf'].collect({'iter': iter, 'rid': 'pu3', 'v': time.time_ns() - p_start})
            # send masked value to targeted collector
            if w:
                for m in w:
                    fog_map[m['pay_load']['target']].collect(m)
        # get the intersection of the clients with the verified signatures in uploaded masked weights
        p_start = time.time_ns() 
        verified_results = [fog_map[n].verify_collected_local_weights_sig() for n in fog_names]
        verified_results.append(aggregator_node.verify_collected_local_weights_sig())
        if iter > 0: # iteration 0 is used as warm-up
            context['perf'].collect({'iter': str(iter), 'rid': 'p1', 'v': time.time_ns() - p_start}) 

        # caculate the intersection of the verified clients by all fog nodes
        p_start = time.time_ns() 
        shared = verified_results[-1]['verified_clients']
        for i in range(num_fognode):
            shared = shared.intersection(verified_results[i]['verified_clients'])
        if iter > 0: # iteration 0 is used as warm-up
            context['perf'].collect({'iter': str(iter), 'rid': 'p2', 'v': time.time_ns() - p_start}) 
        # partial aggregating by all fog nodes
        p_start = time.time_ns() 
        aggregated_weights = []
        for n in fog_names:
            aggregated_weights.append(fog_map[n].partial_aggregate_and_sign(shared))
        if iter > 0: # iteration 0 is used as warm-up
            context['perf'].collect({'iter': str(iter), 'rid': 'p3', 'v': time.time_ns() - p_start}) 
             
        # Federation process: the server verifies and aggregates the partially aggregated model to generate one global model
        # This intermediate version of global model is sent back to all clients for their local training
        theta = aggregator_node.verify_and_aggregate_partial_sig( aggregated_weights, shared, context['perf'])

        # update global test model 
        #net_test.load_state_dict(intermediate_to_weights(theta))
        
        # test and print accuracy   
        # acc_test = evaluate(copy.deepcopy(net_test).to(device), test_loader, device)
        # context['perf'].collect({'iter': iter, 'rid': 'accuracy', 'v': acc_test}) 

        #print(f'Gloal Test Accuracy: {acc_test:.1f}, at epoch {iter}')
        # Vi: the <𝑅_𝑖, 𝑆_𝑖, 𝐴, 𝑆𝐼𝐺^(𝑅_𝑖, 𝑆_𝑖,𝐼, 𝐴)> created by the server nodes, passed to all fog nodes. 
        p_start = time.time_ns() 
        Vi = aggregator_node.calculate_Vi(theta, shared)
        if iter > 0: # iteration 0 is used as warm-up
            context['perf'].collect({'iter': str(iter), 'rid': 'p5', 'v': time.time_ns() - p_start}) 
        # fog nodes verify Vi and return Si and Ri to all the clients in I
        msgs = []
        for n in fog_names:
            new_ri_si = fog_map[n].verify_aggregator_Vi_sig(Vi, server_name, shared)
            if new_ri_si:
                msgs.append(new_ri_si)            
        publishing_Si_Ri = {'intersected': shared, 'msgs': msgs}        


def main():
    sys_config = read_config(logger)
    test_start = datetime.now()
    if torch.cuda.device_count() > 1:
        logger.info("We use " + torch.cuda.device_count() + " GPUs")
    
    logger.info(f'\tWorking on dataset: {dataset_name}')

    results = []
    for n_users in sys_config['num_users']:
        for data_size in sys_config['data_size']:
            for n_fognode in sys_config['num_fognode']:
                for dropout in sys_config['dropout']:
                    logger.info(f'\t   start test on {n_users} users, {n_fognode} fog nodes, {data_size} data_size, dropout = {dropout}')    
                    test_config = {'epochs' : sys_config['epochs'], 'num_users': n_users, 'data_size': data_size, 
                                'num_fognode' : n_fognode, 'test_accuracy': False, 'dropout': dropout}
                    context = {'perf' : PerfCollector(test_config), 'prefix' : dataset_name }
                    start_learning(test_config, context)
                    results.append(context)

    gen_report(results)
    #===================================================================================     
    logger.info(f'----start at {test_start}, end at {datetime.now()}  ------') 
    #===================================================================================     
    print("Training and Evaluation completed!")    
    
    complete_msg = 'performance test completed.'
    header = sys_config.get("completed_notice_header", "")
    if len(header) > 0:
        complete_msg = f'{header}:{complete_msg}'

if __name__ == "__main__":
    main()
