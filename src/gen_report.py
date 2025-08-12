import csv
from datetime import datetime
from statistics import mean

def gen_report(context_list):
    filename = context_list[0]['prefix'] + datetime.now().strftime("%d%m%Y-%H-%M") + '-latencies.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ['user_amount', 'data_size', 'dropout', 'fog_node_amount', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'pmask'] )
        for context in context_list:
            perf = context['perf']
            config = perf.config
            user_node_data = [config['num_users'], config['data_size'], config['dropout'], config['num_fognode']]
            data_row = gen_data_rows(perf, user_node_data)
            writer.writerow( data_row ) 

def gen_data_rows(perf, user_node_data):
    data_collect = {}
    for item in perf.repo:
        if item['iter'] != '0':
            rid = item['rid']
            if rid not in data_collect:
                data_collect[rid] =  []
            data_collect[rid].append(item)
    return user_node_data + [get_average(data_collect['p1'])//1000,
                             get_average(data_collect['p2'])//1000,
                             get_average(data_collect['p3'])//1000,
                             get_average(data_collect['p4'])//1000,
                             get_average(data_collect['p5'])//1000,
                             get_average(data_collect['p6'])//1000,
                             get_average(data_collect['pmask'])//1000
                             ]       

def get_average(dlist):
    return int(mean([x['v'] for x in dlist]))