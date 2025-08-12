import pickle
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

from Crypto.Signature import pss
from Crypto.Hash import SHA256

from sign import get_msg_timestamp, sign_payload
from utils import intermediate_to_weights, gen_maskers, calculate_MAC, scaling
from lenet5 import create_LeNet5_model

### Client-side functions

# to calculate train/test accuray
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

class LocalUpdate(object):
    def __init__(self, idx, client_name, lr, device, dataset_train = None, dataset_test = None, idxs = None,
                 idxs_test = None, private_key = None, config = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 256*4, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 256*4, shuffle = True)
        self.name = client_name
        self.signer = private_key
        self.config = config

    # For training
    def train(self, net):
        net  = {'features.0.weight' : np.full((self.config['data_size']), 0.1, dtype="float64")}
        return net,0.1,0.1
    
    # For testing
    def evaluate(self, net):
        net.eval()
        epoch_acc = []
        epoch_loss = []
        with torch.no_grad():
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # calculate loss
                loss = self.loss_func(fx, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx, labels)
                
                                
                '''if batch_idx % 10 == 0:
                    print('Client{} Test => [{}/{} ({:.0f}%)]\tLoss: {:.4f} \tAcc: {:.3f}'.format(self.idx, batch_idx * len(images), len(self.ldr_test.dataset),
                               100. * batch_idx / len(self.ldr_test), loss.item(), acc.item()))'''
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            
            print('Client{} Test =>                     \tLoss: {:.4f} \tAcc: {:.3f}'.format(self.idx, loss.item(), acc.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
        return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

    def mask_for_fog_nodes(self, sw, iter, collectors):
        # generate random masking values 
        maskers = [gen_maskers(sw) for i in range(len(collectors))]
        # add the masking parameter for the server  
        maskers.append(maskers[0])

        masked = []        
        wkeys = list(sw.keys())
        for i in range(len(collectors)):
            mw = {}
            for key in wkeys:
                mw[key] =  sw[key] + maskers[i][key] - maskers[i+1][key]
            pay_load = {
                'iter': iter,
                'target': collectors[i],
                'result': mw,
                'from' : self.name,
                'timestamp': get_msg_timestamp()
            }
            masked.append({'signature':  sign_payload(self.signer, pay_load),
                        'pay_load' : pay_load})        
        return masked

    def initial_train_and_mask(self, net, iter, collectors):
        # train the model
        w, loss_train, acc_train = self.train(net)
        sw = {}
        wkeys = list(w.keys())
        collector_size = len(collectors)
        for key in wkeys:
            sw[key] = (w[key] * scaling / collector_size).astype(np.int64)
        # masking for fog nodes
        masked = self.mask_for_fog_nodes(sw, iter, collectors) 
        return masked, loss_train, acc_train           
    
    def train_and_mask(self, theta, msgs, iter, collectors, channels, perfcollector):
        # verify new messages and train the model
        try:
            p6_start = time.time_ns()
            verify_intersections(msgs)
            s_i = recover_si(msgs[0], theta)
            self.verify_MAC(s_i, theta, msgs[0]['S_i'])
            verify_S_from_fog_nodes(msgs)
            perfcollector.collect({'iter': str(iter), 'rid': 'p6', 'v': time.time_ns() - p6_start}) 
            #wt = intermediate_to_weights(theta)
            net = None # create_LeNet5_model(self.device, channels, wt)
            w, loss_train, acc_train = self.train(net)
            pmask_start = time.time_ns()
            sw = {}
            wkeys = list(w.keys())
            collector_size = len(collectors)
            for key in wkeys:
                sw[key] = (w[key] * scaling / collector_size).astype(np.int64)
            # masking for fog nodes
            masked = self.mask_for_fog_nodes(sw, iter, collectors) 
            perfcollector.collect({'iter': str(iter), 'rid': 'pmask', 'v': time.time_ns() - pmask_start}) 
            return masked, loss_train, acc_train           
        except Exception as e:
            print(f'Error: {e}')
            return None, None, None

    def verify_MAC(self, s_i, theta, S_i):
        S_0 = calculate_MAC(s_i, theta)
        if S_0 == S_i:
            return True
        else:
            raise Exception('MAC verification in client {self.name} failed!')

def verify_S_from_fog_nodes(msgs):
    for i in range(1, len(msgs)):
        if msgs[i]['S_i'] != msgs[0]['S_i']:
            raise Exception('S_i from different fog nodes are different!')
        
def recover_si(msg, theta):
    s_i = {}
    R_i = msg['R_i']
    for key in list(R_i.keys()):
        s_i[key] = R_i[key] - theta[key]
    return s_i

def verify_intersections(msgs):
    I0 = msgs[0]['A']
    for m in msgs:
        I0 = I0.intersection(m['A'])
    if I0 == msgs[0]['I']:
        return True
    else:
        raise Exception('calculated intersections is different from I!')
    

def verify_theta_consistency(theta, Ti_from_fog_nodes):
    for tf in Ti_from_fog_nodes:
        tf_name = tf['from']
        Ti = tf['Ti']
        S_i = Ti['S_i']
        R_i = Ti['R_i']
        s_i = {}
        for key in list(theta.keys()):
            s_i[key] = R_i[key] - theta[key]
        if calculate_MAC(s_i, theta) == S_i:
            # check all the S_i are the same from different Ti.
            for tmf in Ti_from_fog_nodes:
                if tmf['Ti']['S_i'] != S_i:
                    print(f'ERROR: NOT all Si are the same!')
                    return False
            return True
        else:
            print(f'ERROR: verify_theta_consistency failed from {tf_name}')       
    # None of the Ti from fog nodes can verify theta 
    return False

# DatasetSplit() will enable accessing the items (image and its label) from the Dataset
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
