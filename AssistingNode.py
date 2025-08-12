import socket
import struct
import threading
import time
import random
import sys
import nacl.secret
import nacl.utils
import hashlib
import base64
import subprocess
from ctypes import cdll, c_long, POINTER
from coincurve.keys import PrivateKey
from coincurve.utils import verify_signature
import os

from utils import cpp_executable, aggregation_lib
sigmaValue = 0

def callCcode(key_base64):
    """
    Call C++ code to perform AES encryption in CTR mode.
    """    
    input_data = key_base64.encode()
    process = subprocess.Popen(cpp_executable, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate(input_data)
    output_str = output.decode('utf-8')
    lines = output_str.strip().splitlines()
    maskAlfaTime = lines[-1]
    lines.pop()
    hex_values = []
    binary_mask_values = []
    
    # Convert the output from hex to binary
    for line in lines:
        hex_values.extend(line.split())
    
    for hex_value in hex_values:
        hex_int = int(str(hex_value), 16)
        binary_str = format(hex_int, '032b')
        binary_mask_values.append(binary_str)
    
    if error:
        error_str = error.decode('utf-8')
        print("Error occurred:")
        print(error_str)
    
    return maskAlfaTime, binary_mask_values    

def rhoComputation():
    """
    Generate a random value rho for commitment.
    Algorithm 1 - Setup (Phase 1) - Step 5
    """    
    rho = random.randint(0, 2**256-1)
    return rho

def commitmentMode(commitmentUse, assistantNode_ID, numberOfANs):
    if commitmentUse == 1 and int(assistantNode_ID) == numberOfANs:
        rho = rhoComputation()
    else:
        rho = 0
    return rho

def agree(private_key,clientPublicKey,AssistantNode_ID,NumberOfANs,type):
    if int(AssistantNode_ID) == NumberOfANs and type == 0:
        start_setup_phase2 = time.time()
    
    Xpa = private_key.ecdh(clientPublicKey) # Computing shared secret seed

    return Xpa

def ciphertextComputation(Xpa, rho):
    """
    Encrypt the value rho using the shared secret Xpa.
    Algorithm 1 - Setup (Phase 1) - Step 5    
    """    
    box = nacl.secret.SecretBox(Xpa)
    message = str(rho).encode()
    nonce = nacl.utils.random(nacl.secret.SecretBox.NONCE_SIZE)
    ciphertext = box.encrypt(message, nonce) # Line 5 in setup
    return ciphertext

def aggOfLists(aVector, NumOfAN, type, WEIGHTLISTSIZE):    
    """
    Perform aggregation of lists using C.
    """    
    lib = aggregation_lib # Load the external C code
    lib.add_one.argtypes = [POINTER(POINTER(c_long)), c_long, c_long]
    lib.add_one.restype = POINTER(c_long) # Define the return type for the C function
    
    rows = NumOfAN
    num_rows = len(aVector)
    num_cols = len(aVector[0])
    templist = [[0 for j in range(num_cols)] for i in range(num_rows)]

    # Check format
    for i in range(len(aVector)):
        for j in range(len(aVector[0])):
            if type == 0:
                templist[i][j] = int(aVector[i][j],2)
            else:
                templist[i][j] = aVector[i][j]

    # Convert the Python list to a C-compatible array
    arr_ptr = (POINTER(c_long) * rows)()
    for i in range(rows):
        arr_ptr[i] = (c_long * WEIGHTLISTSIZE)(*templist[i])

    # Perform the aggregation using the C code
    startAggTime = time.time()  
    new_arr_ptr = lib.add_one(arr_ptr, rows, WEIGHTLISTSIZE)
    endAggTime = time.time()  

    # Convert the result back to a Python list
    result = [new_arr_ptr[i] for i in range(WEIGHTLISTSIZE)]    

    aggTime = endAggTime - startAggTime

    return result, aggTime

class AssistingNode:
    def __init__(self, an_name, total_num_ans, weight_list_size, total_num_users, enable_signature, perf_collector):
        self.weight_size = weight_list_size
        self.user_list = []
        self.sigma_value = 0
        self.ver_time = []
        self.client_dict_conn_addr = {}
        self.client_dict = {}
        self.client_dict_information = {}
        self.client_dict_for_xpa = {}
        self.client_dict_for_cpa = {}
        self.list_of_client_address = []
        self.message_m_double_prime = []
        self.total_num_users = total_num_users
        self.an_id = '0'
        self.an_name = an_name
        self.total_num_ans = total_num_ans
        self.user_dict = {}
        self.rho = 0
        self.enable_signature = enable_signature
        self.perf_collector = perf_collector

    def generate_keys(self):
        """
        Generate the private and public keys for the assisting node.
        """
        p_start = time.time_ns() 
        self.private_key_sign = PrivateKey()
        self.public_key_sign = self.private_key_sign.public_key
        self.private_key = PrivateKey()
        self.public_key = self.private_key.public_key
        self.perf_collector.collect({'iter': 999, 'rid': 'pakg', 'v': time.time_ns() - p_start})        

    def compute_xpa(self):
        """
        Compute the shared secrets between the assistant node and all clients.
        Algorithm 1 - Setup (Phase 1) - Step 3
        """    
        for i, (user_name, pk) in enumerate(self.user_dict.items()):
            xpa = self.private_key.ecdh(pk[1]) 
            self.client_dict_for_xpa [user_name] = xpa

    def rhoComputation(self):
        """
        Generate a random value rho for commitment.
        Algorithm 1 - Setup (Phase 1) - Step 5
        """    
        self.rho = random.randint(0, 2**256-1)

    def cal_cpa(self):
        """
        Compute the commitment value for the assistant node.
        Algorithm 1 - Setup (Phase 1) - Step 5
        """    
        self.rhoComputation()
        for i, (user_name, pk) in enumerate(self.user_dict.items()):
            xpa = self.client_dict_for_xpa[user_name]
            self.client_dict_for_cpa [user_name] = ciphertextComputation(xpa, self.rho)
    
    def receive_user_public_keys(self, user_name, client_PKs):
        twoKeysFromClient = struct.unpack(('33s 33s'), client_PKs)
        clientPublicKeySign = twoKeysFromClient[0]
        clientPublicKey = twoKeysFromClient[1]
        self.user_dict[user_name] = clientPublicKeySign, clientPublicKey

    def get_twp_pub_keys_format(self):
        return struct.pack('33s 33s', self.public_key_sign.format(), self.public_key.format())

    def verify_signatures_from_all_users(self, message_to_AN_list):
        """
        Verify the signatures from all users.
        """
        list_of_verified_users = []
        for message in message_to_AN_list:
            user_name = message['user_name']
            messageMPrime = message['messageMPrime']
            # print(f'AN verify_user_signature: msg = {messageMPrime}, signature = {sigAssistingNodes}')
            if self.enable_signature:
                sigAssistingNodes = message['sigAssistingNodes']
                u_pub_key_sign = self.user_dict[user_name][0]
                if verify_signature(sigAssistingNodes, messageMPrime, u_pub_key_sign):
                    list_of_verified_users.append(user_name)
                else:
                    raise Exception(f'AN.verify_signatures_from_all_users: Signature verification failed for {user_name}')
            else:
                list_of_verified_users.append(user_name)
            iterationNumberList = struct.unpack(('l'), messageMPrime)
            iterationNumber = iterationNumberList[0]
        return list_of_verified_users, iterationNumber
    
    def check_threshold(self, list_of_verified_users, iterationNumber):
        """
        Algorithm 2 - Aggregation (Phase 2) - Step 2
        Check the threshold of the number of verified users.
        """
        totalList = []
        finalMaskedValue = []

        total, maskAlfaTime, aggTimePRF = self.computeMaskValue(list_of_verified_users)

        total_0 = total[0]

        for y in total:
            binMaskedWeight = bin(y)[2:]
            if len(binMaskedWeight) > 32:
                binMaskedWeight = binMaskedWeight[-32:]
            totalList.append(binMaskedWeight)

        finalMaskedValue = [element.rjust(32, '0') for element in totalList]
        finalMaskedValue_byte = ','.join(finalMaskedValue).encode('utf-8')

        messageMdoublePrime = struct.pack(f'!L{len(finalMaskedValue_byte)}s', len(total), finalMaskedValue_byte)

        sigServer = None
        if self.enable_signature:
            strForSig = str(finalMaskedValue) + str(iterationNumber) + str(len(list_of_verified_users))
            mystr = strForSig.encode('utf-8')
            messageMdoyblePrimeForSigHash = hashlib.sha256(mystr).hexdigest()
            sigServer = self.private_key_sign.sign(messageMdoyblePrimeForSigHash.encode('utf-8'))

        messageMPrime_I_L = struct.pack('l l l',iterationNumber, len(list_of_verified_users), total_0)

        return messageMdoublePrime, messageMPrime_I_L, sigServer, finalMaskedValue_byte

    def computeMaskValue(self, userlist):
        """
        Compute the mask values for secure aggregation using PRF and AES.
        Algorithm 2 - Aggregation (Phase 2) - Step 2    
        """    
        total = 0
        bVector = []

        if len(userlist) > sigmaValue:        
            for u_name in userlist:
                byte_key = self.client_dict_for_xpa[u_name] #clientDictForXpa[int(userlist[i])]
                key_base64 = base64.b64encode(byte_key).decode('utf-8')
                maskAlfaTime, binary_mask_values = callCcode(key_base64)
                bVector.append(binary_mask_values)
                
            total, aggTimePRF = aggOfLists(bVector, len(userlist),0, self.weight_size) #callToCompute_a

        return total, maskAlfaTime, aggTimePRF
