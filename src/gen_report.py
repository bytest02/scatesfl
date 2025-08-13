'''
Author: Bo Yan bo.yan@csiro.au
Date: 2024-08-20 15:31:14
LastEditors: Bo Yan bo.yan@csiro.au
LastEditTime: 2024-08-29 11:53:40
FilePath: /scates/src/gen_report.py
'''
import csv
from datetime import datetime
from statistics import mean

def gen_report(context_list):
    gen_performace_report(context_list)
    gen_accuracy_report(context_list)

def gen_performace_report(context_list):
    filename = context_list[0]['prefix'] + datetime.now().strftime("-%M-%H-%d%m%Y") + '-latencies.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ['user_amount', 'fog_node_amount', 'round_amount', 'total-user-running',
                          'total-aggregator-running', 'total-fognodes-running'] )
        for context in context_list:
            perf = context['perf']
            config = perf.config
            data_row = [config['num_users'], config['num_fognode'], config['epochs']]  
            data_row.extend(collect_latency_data(perf))
            writer.writerow( data_row )

def sum_per_items(repo, prefix):
    ''' sum the values of items in repo with the same prefix
        return: the sum of the values in microsecond
    '''
    return sum(item['v'] for item in repo if item['rid'].startswith(prefix)) //1000

def collect_latency_data(perf):
    return [
        sum_per_items(perf.repo, 'pu'),  # collect user running time
        sum_per_items(perf.repo, 'pa'),  # collect aggregator running time
        sum_per_items(perf.repo, 'pf')   # collect fog nodes running time
    ]
def gen_accuracy_report(context_list):
    filename = context_list[0]['prefix'] + datetime.now().strftime("-%M-%H-%d%m%Y") + '-acc.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ['user_amount', 'fog_node_amount', 'iteration', 'accuracy'] )
        for context in context_list:
            perf = context['perf']
            config = perf.config
            for item in [r for r in perf.repo if r['rid'] == 'accuracy']:
                acc = item['v']
                iter = item['iter'] + 1
                if iter % 10 == 0:
                    data_row = [config['num_users'], config['num_fognode'], iter , f'{acc:.1f}']
                    writer.writerow( data_row )        
