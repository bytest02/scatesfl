import json
import os
from ctypes import cdll

import requests

parent_dir = os.path.dirname(os.path.abspath(__file__))
cpp_executable = os.path.join(parent_dir,".", "AesModeCTR")
file_path2 = os.path.join(parent_dir, "aggregation.so")
aggregation_lib = cdll.LoadLibrary(file_path2) # Load the external C code

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
