import csv
from datetime import datetime
from statistics import mean

data_col_names = ['pukg', 'pugr', 'pakg', 'pumask', 'panaggr',
                  'psaggr', 'psxtsign', 'puverif']

def gen_report(context_list):
    filename = context_list[0]['prefix'] + datetime.now().strftime("%d%m%Y-%H-%M") + '-latencies.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ['user_amount', 'data_size', 'a-node amount'] + data_col_names)
        for context in context_list:
            perf = context['perf']
            config = perf.config
            user_node_data = [config['num_users'], config['data_size'], config['num_assisnode']]
            data_row = gen_data_row(perf)
            writer.writerow(user_node_data + data_row) 

def gen_data_row(perf):
    data_collect = {}
    for item in perf.repo:
        rid = item['rid']
        if rid not in data_collect:
            data_collect[rid] =  []
        data_collect[rid].append(item)
    return [(get_average(data_collect[cn])//1000) for cn in data_col_names]

def get_average(dlist):
    return int(mean([x['v'] for x in dlist]))