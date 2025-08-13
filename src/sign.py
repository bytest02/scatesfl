from datetime import datetime
import pickle

from Crypto.Hash import SHA256
from Crypto.Signature import pss
from Crypto.PublicKey import RSA

def generate_RSA_key_pair():
    private_key = RSA.generate(2048)
    return private_key.publickey(), private_key

def get_msg_timestamp():
    return datetime.now().isoformat()

def verify_signature(public_key, pay_load, signature):
    verifier = pss.new(public_key)
    sha256 = SHA256.new(pickle.dumps(pay_load))
    verifier.verify(sha256, signature)

def verify_signature_in_message(message, public_keys):
    pay_load = message['pay_load']
    from_name = pay_load['from']
    verify_signature(public_keys[from_name], pay_load, message['signature'])

def sign_payload(signer, pay_load):
    return signer.sign(SHA256.new(pickle.dumps(pay_load)))