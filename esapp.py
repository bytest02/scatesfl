import copy
import time
from datetime import datetime
import logging

from utils import read_config, send_telegram_message
from perf_collector import PerfCollector
from gen_report import gen_report
from ServerNode import Server
from AssistingNode import AssistingNode
from User import User

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="{asctime} - {levelname}: {message}", 
                    style='{', datefmt="%Y-%m-%d %H:%M:%S")

dataset_name = 'e-SeaFL'

### Main app
def start_learning(config, context):
    num_users = config['num_users']
    num_assisnode = config['num_assisnode']
    epochs = config['epochs']
    weight_size = config['data_size'] 
    enable_signature = config['enable_signature']
    perf_collector = context['perf']
    ## create aggregator
    logger.info(f'creating aggregating Server')
    server = Server(enable_signature, weight_size, perf_collector)
    ## create assistant nodes
    an_name_list = [f'an{i+1}' for i in range(num_assisnode)]
    an_map = {}
    user_list = []
    logger.info(f'creating {num_assisnode} assisting nodes')
    for an_name in an_name_list:
        anode = AssistingNode(an_name, num_assisnode, weight_size, num_users, enable_signature, perf_collector)
        an_map[an_name] = anode

    ## create clients
    logger.info(f'creating {num_users} Users')
    logger.info(f'Run Algo 1 - Phase 1 - Step 1 & 7')
    for i in range(num_users):
        user = User(i, num_users, num_assisnode, enable_signature, weight_size, perf_collector)
        user_list.append(user)
        # user setup phase
        #Algorithm 1 - Setup (Phase 1) - Step 1 & Step 7
        user.key_gen()
        user.server_public_key = server.exchange_user_public_key(user.name, user.public_key_sign.format())
        u_2pks_format = user.get_twp_pub_keys_format()
        for an_name in an_name_list:
            as_node = an_map[an_name]
            as_node.receive_user_public_keys(user.name, u_2pks_format)    
    
    #Algorithm 1 - Setup (Phase 1) - Step 2
    logger.info(f'Run Algo 1 - Phase 1 - Step 2 ')
    for an_name in an_name_list:
        as_node = an_map[an_name]
        as_node.generate_keys()
        server.receive_an_public_key(an_name, as_node.public_key_sign.format())
        a_2pks_format = as_node.get_twp_pub_keys_format()
        for user in user_list:
            user.receive_an_public_keys(an_name, a_2pks_format)
    
    for iter in range(epochs):
        logger.info(f'start federated learning iteration {iter+1}')

        #Algorithm 1 - Setup (Phase 1) - Step 3
        logger.info(f'Run Algo 1 - Phase 1 - Step 3 ')
        for an_name in an_name_list:
            an_map[an_name].compute_xpa()

        #Algorithm 1 - Setup (Phase 1) - Step 4
        logger.info(f'Run Algo 1 - Phase 1 - Step 4 ')
        for user in user_list:
            user.compute_xpa()

        #Algorithm 1 - Setup (Phase 1) - Step 5 & 6
        logger.info(f'Run Algo 1 - Phase 1 - Step 5 & 6 ')
        s5_as_node = an_map[an_name_list[0]]
        s5_as_node.cal_cpa() 
        for user in user_list:
            user.receive_cpa(an_name_list[0], s5_as_node.client_dict_for_cpa[user.name])

        # algo 2 Phase 1 - training & masking
        message_to_AN_list = []
        u_message_to_server_list = []
        an_message_to_server_list = []
        logger.info(f'Run Algo 2 - Phase 1 - Step 1 & 2 & 3')
        for user in user_list:
            sigServer, sigAssistingNodes, messageMPrime, messageM, cm, finalMaskedWeightList = user.train_and_masking(iter+1)
            u_message_to_server_list.append( create_message_to_server(user.name, messageM, cm, finalMaskedWeightList, sigServer) )
            message_to_AN_list.append( {'user_name': user.name,
                             'sigAssistingNodes': sigAssistingNodes, 
                             'messageMPrime': messageMPrime } )
        """
            Verify the signatures of users.
            Algorithm 2 - Aggregation (Phase 2) 
        """    
        for an_name in an_name_list:
            as_node = an_map[an_name]
            p_start = time.time_ns() 
            # Algorithm 2 - Aggregation (Phase 2) - Step 1
            list_of_verified_users, iterationNumber = as_node.verify_signatures_from_all_users(message_to_AN_list)
            # Algorithm 2 - Aggregation (Phase 2) - Step 2
            messageMdoublePrime, messageMPrime_I_L, sigServer, finalMaskedValue_byte = as_node.check_threshold(list_of_verified_users, iterationNumber)
            perf_collector.collect({'iter': iter, 'rid': 'panaggr', 'v': time.time_ns() - p_start})
            an_message_to_server_list.append({'an_name': an_name,
                                              'messageMdoublePrime': messageMdoublePrime,
                                              'messageMPrime_I_L': messageMPrime_I_L,
                                              'signature': sigServer,
                                              "finalMaskedValue_byte": finalMaskedValue_byte })
            
        # Algorithm 2 - Aggregation (Phase 2) - Step 3, 4, 5
        finalMaskedWeightList_byte, x_t, signature = server.aggregate_weights(u_message_to_server_list, an_message_to_server_list, iter)
        logger.info(f'Run Algo 2 - Phase 2 - Step 6 - User verify received weights from server')
        for i in range(3):
            user = user_list[i]
            # Algorithm 2 - Aggregation (Phase 2) - Step 6
            p_start = time.time_ns() 
            user.receive_and_verify_new_weights(finalMaskedWeightList_byte, x_t, signature)
            perf_collector.collect({'iter': iter, 'rid': 'puverif', 'v': time.time_ns() - p_start})


def create_message_to_server(user_name, messageM, cm, finalMaskedWeightList, signature):
    commitmentX = str(cm.x())
    commitmentY = str(cm.y())
    commitmentXbyte = commitmentX.encode('utf-8')
    commitmentYbyte = commitmentY.encode('utf-8')
    return {    'user_name':  user_name,
                'commit_X': commitmentXbyte,
                'commit_Y': commitmentYbyte,
                'signature': signature,
                'finalMaskedWeightList' : finalMaskedWeightList }

def main():
    sys_config = read_config(logger)
    test_start = datetime.now()
    results = []
    for n_users in sys_config['num_users']:
        for data_size in sys_config['data_size']:
            for n_assisnode in sys_config['num_assisnode']:
                logger.info(f'\t   start test on {n_users} users, {n_assisnode} fog nodes, {data_size} data_size, enable_signature is {sys_config["enable_signature"]}')    
                test_config = {'epochs' : sys_config['epochs'], 'num_users': n_users, 'data_size': data_size, 
                                'num_assisnode' : n_assisnode, 'enable_signature': sys_config['enable_signature']}
                context = {'perf' : PerfCollector(test_config), 'prefix' : dataset_name }
                start_learning(test_config, context)
                results.append(context)

    gen_report(results)
    #===================================================================================     
    logger.info(f'----start at {test_start}, end at {datetime.now()}  ------') 
    #===================================================================================     
    print("Training and Evaluation completed!")    

if __name__ == "__main__":
    main()
