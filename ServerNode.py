
import os
import time
import struct
import hashlib
from ctypes import cdll, c_long, POINTER
from ecdsa.ellipticcurve import CurveFp, Point

from coincurve.keys import PrivateKey
from coincurve.utils import verify_signature

from utils import aggregation_lib

def aggOfLists(aVector, NumOfAN, type, WEIGHTLISTSIZE):
    """
    Perform aggregation.
    """

    lib = aggregation_lib # Load the external C code
    lib.add_one.argtypes = [POINTER(POINTER(c_long)), c_long, c_long]
    lib.add_one.restype = POINTER(c_long) # Define the return type for the C function

    rows = NumOfAN
    num_rows = len(aVector)
    num_cols = len(aVector[0])
    templist = [[0 for j in range(num_cols)] for i in range(num_rows)]

    # Check format for aggregation
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

def curveInfo():
    """
    Return the elliptic curve parameters for secp256k1.
    """
    p = 115792089237316195423570985008687907853269984665640564039457584007908834671663 # Curve bsae field
    a = 0 # Curve coefficient a
    b = 7 # Curve coefficient b
    curve = CurveFp(p, a, b) # Define the elliptic curve
    h_x = 56193167961224325557053041404644322304275828303249957102234782382884055918593
    h_y = 19073509862472175270077542735739351864502962599188443395223956996042974952935
    h = Point(curve, h_x, h_y) # Generator
    return h, curve

def computeX(num_assisnode, cmDict, total_0):
    """
    Compute the final value x using commitments and masked updates from assistant nodes.
    Algorithm 2 - Aggregation (Phase 2) - Step 5
    """    
    h, curve = curveInfo() # Get the curve and a generator
    cmList = []  
    
    # Compute the second part of x
    for i in range(num_assisnode):
        if i == 0:
            secondPart = -total_0[i] * h
        else:
            secondPart_temp = -total_0[i] * h
            secondPart += secondPart_temp

    # Compute the first part of x
    for key in cmDict:
        x = cmDict[key][0]
        y = cmDict[key][1]
        cm = Point(curve, x, y)
        cmList.append(cm)

    for i in range(len(cmList)):
        if i == 0:
            firstPart = cmList[i]
        else:
            firstPart += cmList[i]

    x = firstPart + secondPart # Combine both parts to get x

    # Pack the coordinates of x into binary format
    x_Xbyte = str(x.x()).encode('utf-8')
    x_Ybyte = str(x.y()).encode('utf-8')
    x_X = struct.pack(f"!{len(x_Xbyte)}s", x_Xbyte)
    x_Y = struct.pack(f"!{len(x_Ybyte)}s", x_Ybyte)
    return x_X, x_Y

class Server:

    def __init__(self, enable_signature, weight_size, perf_collector) -> None:
        self.an_dict = {}
        self.user_dict = {}
        self.enable_signature = enable_signature
        self.weight_size = weight_size
        self.perf_collector = perf_collector
        self.key_gen()

    def key_gen(self):
        self.private_key_sign = PrivateKey()
        self.public_key_sign = self.private_key_sign.public_key
        
    def receive_an_public_key(self, an_name, pks_from_AN):
        self.an_dict[an_name] = pks_from_AN

    def exchange_user_public_key(self, user_name, public_key_sign):
        self.user_dict[user_name] = public_key_sign
        return self.public_key_sign.format()
        
    def receive_weights_from_users(self, u_message_list):
        """
        Algorithm 2 - Aggregation (Phase 2) - Step 3
        """
        self.verified_user_weights = []
        for u_message in u_message_list:
            commitment_x = u_message['commit_X'].decode('utf-8')
            commitment_y = u_message['commit_Y'].decode('utf-8')
            finalMaskedWeightList = u_message['finalMaskedWeightList'] 
            strForSig = str(finalMaskedWeightList) + commitment_x + commitment_y
            user_name = u_message['user_name']
            if self.verify_user_signature(u_message, strForSig):
                u_message['commit'] = int(commitment_x), int(commitment_y)
                self.verified_user_weights.append( {'commit': (int(commitment_x), int(commitment_y)),
                                        'user_name' : user_name,
                                        'finalMaskedWeightList': finalMaskedWeightList} )
                #print(f" ------- Server: Signature verification successful for {user_name}")
            else:
                print(f" ------- Server: Signature verification failed for {user_name}")

    def receive_AN_delta(self, an_message_list):
        """
        Algorithm 2 - Aggregation (Phase 2) - Step 4
        """
        #print(f" ------- Server: Received all delta from ANs")
        self.verified_AN_delta = []
        for an_message in an_message_list:
            an_name = an_message['an_name']
            mDoublePrimeUnpack = struct.unpack(('l l l'),an_message['messageMPrime_I_L'])
            #print(f'----- mDoublePrimeUnpack[1] = {mDoublePrimeUnpack[1]}') # listOfUserInAssistantNodes[0]
            a_from_AN = an_message['finalMaskedValue_byte'].decode('utf-8').split(',') # listOfmaskedUpdatesInAssistantNodes[0]
            if self.verify_AN_signature(an_message, mDoublePrimeUnpack, a_from_AN):
                self.verified_AN_delta.append( {'an_name': an_name,
                                            'mDoublePrimeUnpack': mDoublePrimeUnpack,
                                            'total_0': mDoublePrimeUnpack[2],
                                            'a_from_AN': a_from_AN} )
                #print(f" ------- Server: Signature verification successful for {an_name}")
            else:
                raise Exception(f" ------- Server: Signature verification failed for {an_name}")
        return self.check_userList_condition(self.verified_AN_delta) # copy from Server.py: line 262 - checkUserListCondition

    def aggregate_weights(self, u_message_list, an_message_list, iter):
        p_start = time.time_ns() 
        #Algorithm 2 - Aggregation (Phase 2) - Step 3
        self.receive_weights_from_users(u_message_list)
        #Algorithm 2 - Aggregation (Phase 2) - Step 4
        if not self.receive_AN_delta(an_message_list):
            raise Exception("Server: ABORT - verifying messages from ANs and users are failed!")
        """
        Compute the final aggregated update from the masked updates and user-provided values.
        Algorithm 2 - Aggregation (Phase 2) - Step 4 & Step 5
        """
        finalWeightList = self.compute_final_update()
        self.perf_collector.collect({'iter': iter, 'rid': 'psaggr', 'v': time.time_ns() - p_start})

        # copy from Server.py: line 277 - checkUserListCondition
        p_start = time.time_ns() 
        cmDict = {}
        total_0 = [va['total_0'] for va in self.verified_AN_delta]
        for vu in self.verified_user_weights:
            uname = vu['user_name']
            cmDict[uname] = vu['commit']    
        num_verified_AN = len(self.verified_AN_delta)
        # copy from Server.py: line 77 - send_x_w
        x_X, x_Y = computeX(num_verified_AN, cmDict, total_0)
        FinalWeightListbin = []

        # Convert each element of FinalWeightList to binary string
        for element in finalWeightList:
            binary = bin(element)[2:]
            FinalWeightListbin.append(binary)
        
        # Ensure each binary string is 32 bits long
        FinalWeightListbin256 = [element.rjust(32, '0') for element in FinalWeightListbin]
        finalMaskedWeightList_byte = ','.join(FinalWeightListbin256).encode('utf-8')
        
        signature = self.sign_final_weight_list(finalWeightList)
        self.perf_collector.collect({'iter': iter, 'rid': 'psxtsign', 'v': time.time_ns() - p_start})

        return finalMaskedWeightList_byte, (x_X, x_Y), signature

    def sign_final_weight_list(self, finalWeightList):
        signature = None
        if self.enable_signature:
            data_size = struct.pack(f'{len(finalWeightList)}i', *finalWeightList)
            messageMForSigHash = hashlib.sha256(data_size).hexdigest()
            signature = self.private_key_sign.sign(messageMForSigHash.encode('utf-8'))

        return signature        

    def compute_final_update(self):
        """
        Compute the final aggregated update from the masked updates and user-provided values.
        Algorithm 2 - Aggregation (Phase 2) - Step 4 
        copy from Server.py: line 214 - computeFinalUpdate
        """
        finalWeightList = []
        listOfmaskedUpdatesInAssistantNodes = [va['a_from_AN'] for va in self.verified_AN_delta]
        # Aggregate masked updates from assistant nodes
        a_t_A, _ = aggOfLists(listOfmaskedUpdatesInAssistantNodes, len(self.verified_AN_delta),0, self.weight_size)
        # Aggregate masked values from users
        listOfyInclient = [vu['finalMaskedWeightList'] for vu in self.verified_user_weights]
        y_t_p, _ = aggOfLists(listOfyInclient, len(self.verified_user_weights),0, self.weight_size)

        # Check format of the values
        a_t_A_list = [(bin(y & 0xFFFFFFFF)[2:]) for y in a_t_A]
        y_t_p_list = [(bin(y & 0xFFFFFFFF)[2:]) for y in y_t_p]

        # Compute the final weights by subtracting aggregated values
        for i in range(0, self.weight_size):
            FinalWeightValue = int(y_t_p_list[i], 2) - int(a_t_A_list[i], 2)
            if FinalWeightValue < 0:
                y_t_p_list[i] = "1" + y_t_p_list[i]
                FinalWeightValue = int(y_t_p_list[i], 2) - int(a_t_A_list[i], 2)

            finalWeightList.append(FinalWeightValue)
        return finalWeightList        

    def check_userList_condition(self, verified_AN_delta):
        verified_user_amount = len(self.verified_user_weights)
        for an_delta in verified_AN_delta:
            if an_delta['mDoublePrimeUnpack'][1] != verified_user_amount:
                raise Exception(f"Server: ABORT - Number of users verified by AN {an_delta['an_name']} is not equal to number of users by the server!")
        return True

    def verify_AN_signature(self, an_message, mDoublePrimeUnpack, a_from_AN):
        if self.enable_signature:
            an_name = an_message['an_name']
            signature = an_message['signature']
            strForSig = str(a_from_AN) + str(mDoublePrimeUnpack[0]) + str(mDoublePrimeUnpack[1])
            mystr = strForSig.encode('utf-8')
            messageMForSigHash = hashlib.sha256(mystr).hexdigest().encode('utf-8')
            return verify_signature(signature, messageMForSigHash, self.an_dict[an_name])
        return True
    
    def verify_user_signature(self, u_message, strForSig):
        if self.enable_signature:
            user_name = u_message['user_name']
            mystr = strForSig.encode('utf-8')
            messageMForSigHash = hashlib.sha256(mystr).hexdigest().encode('utf-8')
            return verify_signature(u_message['signature'], messageMForSigHash, self.user_dict[user_name])
        return True