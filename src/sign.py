from datetime import datetime
import pickle
import hashlib

from coincurve.keys import PrivateKey
from coincurve.utils import verify_signature

def generate_sign_key_pair():
    private_key_sign = PrivateKey()
    return private_key_sign.public_key, private_key_sign

def get_msg_timestamp():
    return datetime.now().isoformat()

def verify_payload_signature(public_key, pay_load, signature):
    if not verify_signature(signature, hash_payload(pay_load), public_key):
        raise ValueError("Signature verification failed")

def verify_signature_in_message(message, public_keys):
    pay_load = message['pay_load']
    from_name = pay_load['from']
    verify_payload_signature(public_keys[from_name], pay_load, message['signature'])

def hash_payload(pay_load):
    return hashlib.sha256(pickle.dumps(pay_load)).digest()

def sign_payload(private_key, pay_load):
    bhash = hash_payload(pay_load)
    return private_key.sign(bhash)