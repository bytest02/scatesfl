import copy
import pickle
import json
import time

import torch
import numpy as np
from Crypto.Signature import pss
from Crypto.Hash import SHA256

from sign import get_msg_timestamp, verify_signature_in_message, sign_payload
from utils import gen_maskers, calculate_MAC

class FogNode(object):
    def __init__(self, name, private_key, public_keys):
        self.name = name
        self.collected = [] 
        self.signer = private_key
        self.public_keys = public_keys
        self.F = set()

    def collect(self, masked):
        self.collected.append(masked)

    def clear_collected(self):
        self.collected=[]
        self.F = set()

    def verify_aggregator_Vi_sig(self, Vi, server_name, shared):
        try:
            if server_name == Vi['pay_load']['from']:
                verify_signature_in_message(Vi, self.public_keys)
                pay_load = Vi['pay_load']
                msg = {
                    'S_i' : pay_load['S_i'],
                    'R_i' : pay_load['R_i'],
                    'A': pay_load['A'],
                    'I': pay_load['I'],
                    'F': self.F,
                    'from': self.name
                }
                return msg
        except (ValueError, TypeError):
            print(f'ERROR: Failed to verify signature from {server_name}!')
        return None

    def verify_collected_local_weights_sig(self):
        verified = set()
        for m in self.collected:
            try:
                verify_signature_in_message(m, self.public_keys)
                from_name = m['pay_load']['from']
                verified.add(m['pay_load']['from'])
            except (ValueError, TypeError):
                print(f'ERROR: Failed to verify signature from {from_name}!')
        self.F = verified
        return {'verified_clients': verified, 'verified_by': self.name}
    
    def partial_aggregate(self, intersected):
        """partial aggregate the masked weights collected by this node.
        Args:
            intersected: the list of client names which has been verified by all the fog nodes
            example: {'client1', 'client2', 'client3'}
        Returns:
            the partial aggregation result dictionary with weight name and numpy pairs
        """
        aggregated = None
        for c in self.collected:
            if c['pay_load']['from'] in intersected:
                if aggregated:
                    local_w =  c['pay_load']['result']
                    for k in local_w.keys():
                        if k in aggregated:
                            aggregated[k] = aggregated[k] + local_w[k]
                        else: 
                            aggregated[k] = local_w[k]
                else:
                    aggregated = c['pay_load']['result']
        return aggregated, self.collected[0]['pay_load']['iter']
    
    def partial_aggregate_and_sign(self, intersected):
        """partial aggregate the masked weights collected by this node, sign the result, with the iteration number.
        Args:
            intersected: the list of client names which has been verified by all the fog nodes
        Returns:
            an object with the partial aggregation result signed by this node
        """
        aggregated, iter = self.partial_aggregate(intersected)
        pay_load = {
            'iter': iter,
            'partial': aggregated,
            'from' : self.name,
            'timestamp': get_msg_timestamp()
        }
        return {'signature':  sign_payload(self.signer, pay_load),
                'pay_load' : pay_load}
    
class AggregatorNode(FogNode):
    
    def verify_and_aggregate_partial_sig(self, other_aggregated, intersected, perf_collector):
        server_aggregated, iter = self.partial_aggregate(intersected)
        p4_start = time.time_ns() 
        for paggr in other_aggregated:
            verify_signature_in_message(paggr, self.public_keys)
            local_w = paggr['pay_load']['partial']
            for k in local_w.keys():
                if k in server_aggregated:
                    server_aggregated[k] = server_aggregated[k] + local_w[k]
                else: 
                    server_aggregated[k] = local_w[k]
        # averaging the weights from the intersected clients
        for key in list(server_aggregated.keys()):
            server_aggregated[key] = server_aggregated[key] // len(intersected)  
        if iter > 0: # iteration 0 is used as warm-up
            perf_collector.collect({'iter': str(iter), 'rid': 'p4', 'v': time.time_ns() - p4_start})         
        return server_aggregated
    
    def calculate_Vi(self, theta, intersected):
        s_i = gen_maskers(theta)
        R_i = {}
        for key in list(theta.keys()):
            R_i[key] = s_i[key] + theta[key]
        pay_load = {
            'S_i' : calculate_MAC(s_i, theta),
            'R_i' : R_i,
            'A': self.F,
            'I': intersected,
            'from': self.name,
            'timestamp': get_msg_timestamp()
        }
        return {'signature':  sign_payload(self.signer, pay_load), 'pay_load': pay_load}
