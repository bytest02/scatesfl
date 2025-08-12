import pickle
import json
import os

import requests

import numpy as np
import torch

from Crypto.Hash import SHA256

scaling = 0x000000ffffffffff # 2^40-1

max_random_number = 0xfffffff

# for masking and precisely hashing purpose

def intermediate_to_weights(ir):
    weights = {}
    for key in list(ir.keys()):
        uw = ir[key].astype(np.float64) / scaling   
        weights[key] = torch.from_numpy(uw)     
    return weights

def gen_maskers(np_weights):
    rnm = {}
    for key in list(np_weights.keys()):
        rnm[key] = np.random.randint(0, max_random_number, np_weights[key].shape)
    return rnm

def calculate_MAC(s_i, theta):
    sha256 = SHA256.new(pickle.dumps(s_i))
    sha256.update(pickle.dumps(theta))
    return sha256.digest().hex()

def load_json(filename, encoding='utf-8'):
    try:
        f = open(filename, encoding=encoding)
        return json.load(f)
    except Exception as e:
        print(e)
        return None
        
def read_config(logger):
    config_file = os.environ.get('CONFIG_FILE')
    if not config_file:
        config_file = 'config.json'
    logger.info(f'Loading config file: {config_file}')
    return load_json(config_file)

def send_telegram_message(bot_token, chat_id, message):
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print('Message sent successfully')
    else:
        print(f'Failed to send message: {response.status_code}')