import os
import struct
import time
import hashlib
import nacl.secret
import nacl.utils
import ecdsa
from ecdsa.ellipticcurve import CurveFp, Point
from ctypes import cdll, c_long, POINTER
import subprocess
import base64
from coincurve.keys import PrivateKey
from coincurve.utils import verify_signature

from utils import cpp_executable, aggregation_lib
            
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
        raise Exception("Error occurred: " + error.decode('utf-8'))

    return maskAlfaTime, binary_mask_values


def getWeightList(WEIGHTLISTSIZE):
    """
    Get the trained model weight list.
    """    
    weightList = [2] * WEIGHTLISTSIZE
    return weightList

def curveInfo():
    """
    Return the elliptic curve parameters for secp256k1.
    """    
    p = 115792089237316195423570985008687907853269984665640564039457584007908834671663 # Curve bsae field
    a = 0 # Curve coefficient a
    b = 7 # Curve coefficient b
    curve_1 = CurveFp(p, a, b) # Define the elliptic curve
    h_x = 56193167961224325557053041404644322304275828303249957102234782382884055918593
    h_y = 19073509862472175270077542735739351864502962599188443395223956996042974952935
    h = Point(curve_1, h_x, h_y) # Generator
    return h

def KeyGen():
    """
    Generate a key pair.
    Algorithm 1 - Setup (Phase 1) - Step 1
    """    
    start_setup_phase_malicous_setting = time.time()
    private_key_sign = PrivateKey()
    public_key_sign = private_key_sign.public_key
    end_setup_phase_malicous_setting = time.time()
    timeMaliciousSetting = end_setup_phase_malicous_setting - start_setup_phase_malicous_setting

    start_setup_phase1 = time.time()
    private_key = PrivateKey()
    public_key = private_key.public_key
    end_setup_phase1 = time.time()
    keyGenTime = end_setup_phase1-start_setup_phase1

    return private_key_sign, public_key_sign, private_key, public_key, keyGenTime, timeMaliciousSetting

def aggOfLists(aVector, NumOfAN, type, WEIGHTLISTSIZE):
    """
    Perform aggregation of lists.
    """    
    lib = aggregation_lib # Load the external C code
    lib.add_one.argtypes = [POINTER(POINTER(c_long)), c_long, c_long]
    lib.add_one.restype = POINTER(c_long)  # Define the return type for the C function

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

def computeCommitment(weightList, total_0, rho, WEIGHTLISTSIZE):
    """
    Compute the commitment for the given weight list.
    Algorithm 2 - Aggregation (Phase 1) - Step 2
    """
    listOfGenerators = []
    curve = ecdsa.SECP256k1
    generator = curve.generator
    x = int(generator.x())
    y = int(generator.y())

    secondpart = Point(None, None, None)
    listofTimes = []
    
    # Compute the second part of the commitment
    for i in range(WEIGHTLISTSIZE):
        rho_weight = rho * weightList[i]           
        newGenerator = generator * (135351+i)
        listOfGenerators.append(newGenerator)

        start100 = time.time()
        secondpart += (newGenerator * rho_weight)        
        end100 = time.time()
        listofTimes.append(end100-start100)
    
    ecMultTime = sum(listofTimes)
    h = curveInfo()

    # Compute the first part of the commitment and combine both parts
    start10 = time.time()
    FirstPart = h * total_0
    cm = FirstPart + secondpart # line 2 - Aggregation phase (Round 1)
    end10 = time.time()
    timeForCM = end10 - start10

    return ecMultTime, timeForCM, cm, listOfGenerators

def nComputeMaskedWeight(weightList, iterationNumber, total, WEIGHTLISTSIZE):
    """
    Compute the masked weight list for secure aggregation.
    Algorithm 2 - Aggregation (Phase 1) - Step 1
    """
    maskedWeightList = []
    cVector = []
    cVector.append(weightList)
    cVector.append(total)

    # Aggregate the masked weight list using C
    yList, aggTimeY = aggOfLists(cVector, len(cVector), 1, WEIGHTLISTSIZE)

    for y in yList:
        binMaskedWeight = bin(y)[2:]
        if len(binMaskedWeight) > 32:
            binMaskedWeight = binMaskedWeight[-32:]
        maskedWeightList.append(binMaskedWeight)

    finalMaskedWeightList = []
    finalMaskedWeightList = [element.rjust(32, '0') for element in maskedWeightList]
    finalMaskedWeightList_byte = ','.join(finalMaskedWeightList).encode('utf-8')

    messageM = struct.pack(f'!L{len(finalMaskedWeightList_byte)}s', len(weightList), finalMaskedWeightList_byte)
    messageMPrime = struct.pack('l',iterationNumber)
    return finalMaskedWeightList, messageMPrime, messageM, aggTimeY

def generatingSignature(finalMaskedWeightList, private_key_sign, cm, messageMPrime):
    """
    Generate the signature for the masked weight list and commitment.
    Algorithm 2 - Aggregation (Phase 1) - Step 3
    """
    strForSig = str(finalMaskedWeightList) + str(cm.x()) + str(cm.y())
    mystr = strForSig.encode('utf-8')
    
    messageMForSigHash = hashlib.sha256(mystr).hexdigest()
    sigServer = private_key_sign.sign(messageMForSigHash.encode('utf-8'))
    sigAssistingNodes = private_key_sign.sign(messageMPrime)

    return sigServer, sigAssistingNodes

class User:
    def __init__(self, user_id, num_users, num_ans, enable_signature, weight_size, perf_collector):
        self.user_id = str(user_id)
        self.name = f'usr{user_id}' 
        self.num_users = int(num_users)
        self.num_ans = int(num_ans)
        self.bandwidth_print = 0
        self.input_argv = 1
        self.outbound_bandwidth = []
        self.client = None
        self.list_assistantNode_connections = []
        self.list_encrypted_value = [] # List to store encrypted rho
        self._assistantNodeSharedSecretDict = {}
        self._assistantNodePublicKeyDict = {}
        self._assistantNodeIDList = []
        self._Xvalue = [] # List to store x elliptic curve points
        self.server_public_key = None
        self.an_pks = {}
        self.enable_signature = enable_signature
        self.weight_size = weight_size
        self.perf_collector = perf_collector
    
    def key_gen(self):
        # Generate the key        
        p_start = time.time_ns() 
        self.private_key_sign, self.public_key_sign, self.private_key, self.public_key, self.key_gen_time, self.time_malicious_setting = KeyGen()
        self.perf_collector.collect({'iter': 999, 'rid': 'pukg', 'v': time.time_ns() - p_start})

    def get_twp_pub_keys_format(self):
        return struct.pack('33s 33s', self.public_key_sign.format(), self.public_key.format())
    
    def compute_xpa(self):
        # Compute shared secrets
        """
        Compute the shared secrets between the user and all assistant nodes.
        Algorithm 1 - Setup (Phase 1) - Step 4 & 6
        """
        for _, (an_name, an_pub_key) in enumerate(self.an_pks.items()):
            Xpa = self.private_key.ecdh(an_pub_key) # Computing shared secret seed
            self._assistantNodeSharedSecretDict[an_name] = Xpa


    def setup_phase(self):
        self.key_gen()
    
    def receive_an_public_keys(self, name, an_PKs):
        twoKeysFromAssistantNode = struct.unpack(('33s 33s'), an_PKs)
        AssistantNodePublicKeySign = twoKeysFromAssistantNode[0]
        AssistantNodePublicKey = twoKeysFromAssistantNode[1]
        self.an_pks[name] = AssistantNodePublicKey

    def receive_cpa(self, an_name, cpa):
        p_start = time.time_ns() 
        self.cpa = cpa
        xpa = self._assistantNodeSharedSecretDict[an_name] 
        box = nacl.secret.SecretBox(xpa)
        self.rho = int(box.decrypt(cpa))
        self.perf_collector.collect({'iter': 999, 'rid': 'pugr', 'v': time.time_ns() - p_start})

        #print(f'User-{self.name}.receive_cpa: Decrypted rho = {self.rho}')

    def train_and_masking(self, iterationNumber):
        """
        Get the trained model weights and compute masked weights, generate signatures, and sending data to the server.
        Algorithm 2 - Aggregation (Phase 1) - Step 1 & 2 & 3
        """    
        # Step 1
        weightList = getWeightList(self.weight_size)
        p_start = time.time_ns() 
        total, total_0, maskAlfaTime, aggTimePRF = self.computeMaskValue() #a
        # step 2
        ecMultTime, timeForCM, cm, listOfGenerators = computeCommitment(weightList, total_0, self.rho, self.weight_size)
        self._listOfGenerators = listOfGenerators # Store the list of generators for validate the aggregated model from the server

        finalMaskedWeightList, messageMPrime, messageM, aggTime2 = nComputeMaskedWeight(weightList, iterationNumber, total, self.weight_size)
            
        # step 3
        sigServer = sigAssistingNodes= None
        if self.enable_signature:
            sigServer, sigAssistingNodes= generatingSignature(finalMaskedWeightList, self.private_key_sign, cm, messageMPrime)
        self.perf_collector.collect({'iter': iterationNumber, 'rid': 'pumask', 'v': time.time_ns() - p_start})
                        
        return sigServer, sigAssistingNodes, messageMPrime, messageM, cm, finalMaskedWeightList

    def computeMaskValue(self):
        """
        Compute the mask values for secure aggregation using PRF and AES.
        Algorithm 2 - Aggregation (Phase 1) - Step 1
        """    
        total = 0
        bVector = []
        an_name_list = []
        for _, (an_name, byte_key) in enumerate(self._assistantNodeSharedSecretDict.items()):
            key_base64 = base64.b64encode(byte_key).decode('utf-8')
            maskAlfaTime, binary_mask_values = callCcode(key_base64)
            bVector.append(binary_mask_values)
            an_name_list.append(an_name)

        # Aggregate the mask values using C
        total, aggTimePRF = aggOfLists(bVector, self.num_ans,0, self.weight_size)
        total_0 = total[0]

        return total, total_0, maskAlfaTime, aggTimePRF
    
    def receive_and_verify_new_weights(self, finalMaskedWeightList_byte, x_t, signature):
        """
        Retrieve the aggregated weight list and x values from the server.
        Algorithm 2 - Aggregation (Phase 2) - Step 6
        """        
        finalAggregatedWeightList = finalMaskedWeightList_byte.decode('utf-8').split(',')
        finalWeightListAggregated = [int(x, 2) for x in finalAggregatedWeightList]
        x_X = x_t[0]
        x_Y = x_t[1]
        x1 = struct.unpack(f"!{len(x_X)}s", x_X)
        x_xValue = x1[0].decode('utf-8')
        x2 = struct.unpack(f"!{len(x_Y)}s", x_Y)
        x_yValue = x2[0].decode('utf-8')
        Xvalue = [x_xValue, x_yValue]
        
        self.validate_aggregated_model(finalWeightListAggregated, Xvalue)
        self.verify_server_signature(finalWeightListAggregated, signature)

    def verify_server_signature(self, finalWeightList, signature):
        if self.enable_signature:
            mystr = struct.pack(f'{len(finalWeightList)}i', *finalWeightList)
            sigHash = hashlib.sha256(mystr).hexdigest().encode('utf-8')
            if not verify_signature(signature, sigHash, self.server_public_key):
                raise Exception("signature verification failed") 

    def validate_aggregated_model(self, finalWeightListAggregated, Xvalue):
        """
        Validate the aggregated model using the aggregated weight list and a list of generators.
        Algorithm 2 - Aggregation (Phase 2) - Step 6
        """    
        for i in range(len(finalWeightListAggregated)):
            rho_FinalWeight = self.rho * finalWeightListAggregated[i]
            if i == 0:
                tempCompare = rho_FinalWeight * self._listOfGenerators[i]
            else:
                tempCompare += rho_FinalWeight * self._listOfGenerators[i]

        if tempCompare.x() != int(Xvalue[0]) or tempCompare.y() != int(Xvalue[1]):
            raise Exception("Client: Aborts, not a valid aggregated model") 

