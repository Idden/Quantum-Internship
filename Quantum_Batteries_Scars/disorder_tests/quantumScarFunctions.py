import math
import numpy as np
from scipy.sparse import csr_matrix
import qutip as qt

# task 1: make function that turns binary to decimal
def binToDeci(num):
    
    startBase = math.pow(2, len(num)-1)
    deciNum = 0

    for i in range(len(num)):
        deciNum += int(num[i]) * startBase
        startBase /= 2
        
    return int(deciNum)

# task 2: no consecutive ones in binary sequence
def binNoConsecOnesEfficient(N):

    def recursiveBin(n, prevNum, currNum):

        #print(currNum, 'b')
        
        if n == 0:
            listNoConsecOnes.append(currNum)
            return
        
        recursiveBin(n - 1, '0', currNum + '0')

        #print(currNum, 'a')
        
        if prevNum != '1':
            recursiveBin(n - 1, '1', currNum + '1')

    listNoConsecOnes = []
    recursiveBin(N, None, '')
    
    return listNoConsecOnes

# creates z2 state
def z2_initial(N):
    return ''.join('1' if i % 2 == 0 else '0' for i in range(N))

# drive functions
def coeff(t, A, omega):
    return A * np.sin(omega * t)
def const(t, A):
    return A * t
def timed_drive(t, A, omega, limit):
    return (A if t < limit else 0) * np.sin(omega * t)
def timed_const(t, A, limit):
    return (A if t < limit else 0) * t
def make_coeff(r):
    return lambda t, args: args["A"] * np.cos(args[f"wd{r}"] * t)


def get_scar_ham(N, disorder=False, detuning=0.0, random_seed=False, ds_detuning=0.0, random_drive_strength=False, random_drive_freq=False):
    assert (N % 2 == 0), "N must be a multiple of 2"
    assert (random_drive_strength == False or random_drive_freq == False), "Random drive strength and frequency cannot both be True"

    if not random_seed:
        np.random.seed(0)

    basisList = binNoConsecOnesEfficient(N)
    for basis in basisList:
        if basis[0] == '1' and basis[-1] == '1':
            basisList.remove(basis)
        
    basisMap = {bitStr: i for i, bitStr in enumerate(basisList)}
    basisLen = len(basisList)
    flippedList = []
    ohms = 1.0

    rowBare = []
    columnBare = []

    rowFactor = []
    columnFactor = []

    # flip bit hashmap
    flipMap = {'0': '1', '1': '0'}

    # sigma z op hashmap
    sigzMap = {'0': '-1', '1': '1'}

    # list of ints for Hamiltonian
    numList = []

    # -------------------------------
    #
    # create the bare PXP hamiltonian
    #
    # -------------------------------
    for i in range(basisLen):

        # add padding so that search doesnt go out of range
        paddedBitStr = basisList[i][-1] + basisList[i] + basisList[i][0]
        copyBit = list(paddedBitStr)

        # apply the sum of r P_r-1 * sigma_x * P_r+1 operator
        for j in range(1, N+1):
            
            if paddedBitStr[j-1] == '0' and paddedBitStr[j+1] == '0':
                copyBit[j] = flipMap[paddedBitStr[j]]
                flippedList.append(''.join(copyBit)[1:-1])
                copyBit = list(paddedBitStr)
            
        # adds row and column values for the sparse matrix
        for k in range(len(flippedList)):
            rowBare.append(basisMap[flippedList[k]])
            columnBare.append(i)
            
        flippedList.clear()

    # -------------------------------
    #
    # create the sigma Z PXP hamiltonian perturbation
    #
    # -------------------------------
    for i in range(basisLen):

        # add padding so that search doesnt go out of range
        paddedBitStr = basisList[i][-2] + basisList[i][-1] + basisList[i] + basisList[i][0] + basisList[i][1]
        copyBit = list(paddedBitStr)
        factor = 1

        # apply the PXP operator
        for j in range(2, N+2):
            
            if (paddedBitStr[j-1] == '0') and (paddedBitStr[j+1] == '0'):
                copyBit[j] = flipMap[paddedBitStr[j]]

                # apply sigmaZ_r-2 + sigmaZ_r+2
                r_neg2 = int(sigzMap[paddedBitStr[j-2]])
                r_pos2 = int(sigzMap[paddedBitStr[j+2]])
                factor = r_neg2 + r_pos2
                numList.append(factor)

                flippedList.append(''.join(copyBit)[2:-2])
                copyBit = list(paddedBitStr)
            
        # adds row and column values for the sparse matrix
        for k in range(len(flippedList)):
            rowFactor.append(basisMap[flippedList[k]])
            columnFactor.append(i)
            
        flippedList.clear()

    # list of ones for the sparse matrix
    onesList = np.ones(len(rowBare), dtype=int)

    # create the sparse matrix and turn it into a Qobj
    sparseBareHamiltonian = csr_matrix((onesList, (rowBare, columnBare)), shape=[basisLen, basisLen])
    sparseFactoredHamiltonian = csr_matrix((numList, (rowFactor, columnFactor)), shape=[basisLen, basisLen])
    H0 = (ohms / 2 * sparseBareHamiltonian) + (-0.026 * ohms * sparseFactoredHamiltonian)

    # -------------------------------
    #
    # create anharmonic term
    #
    # -------------------------------

    # disorder strength
    if disorder:
        d = detuning
        detune_list = []

        # random diagonal with zero mean
        delta_w = np.random.uniform(-d, d, N)
        delta_w -= np.mean(delta_w)

        intBasisList = []
        for i in range(basisLen):
            intBasisList.append(2 * np.array([int(k) for k in basisList[i]]) - 1)

        for i in range(basisLen):
            detune_list.append(np.dot(intBasisList[i], delta_w))

        pert_location = list(range(basisLen))
        deltaH = csr_matrix((detune_list, (pert_location, pert_location)), shape=[basisLen, basisLen])
        
        H0 = H0 + deltaH

    H0 = qt.Qobj(H0)

    # -------------------------------
    #
    # states and evolutions set ups
    #
    # -------------------------------

    # diagonalize the sparse matrix
    eigenvalues, eigenstates = H0.eigenstates()

    # initial state
    z2_str = z2_initial(N)
    z2_index = basisMap[z2_str]
    psi0 = qt.basis(basisLen, z2_index)

    # -------------------------------
    #
    # create the driving hamiltonian
    #
    # -------------------------------

    if not random_drive_freq:
        # create H1 operator for QobjEvo!
        copyBasis = basisList.copy()
        diagH1 = []

        if random_drive_strength:
            driveWeights = np.random.uniform(-ds_detuning, ds_detuning, N)
            driveWeights = 1.0 + driveWeights

        # switches 0s to -1s and keeps 1s the same for the copyBasis
        # appends to diagH1 the dot product between each bit string and the 0 -> -1 Z2 state
        z2bitString = list(z2_initial(N))
        z2bitString = 2 * np.array([int(i) for i in z2bitString]) - 1
        
        for i in range(basisLen):

            bitString = list(copyBasis[i])
            bitString = [int(i) for i in bitString]

            if random_drive_strength:
                diagH1.append(np.dot(driveWeights * (2 * np.array(bitString) - 1), z2bitString))
            else:
                diagH1.append(np.dot(2 * np.array(bitString) - 1, z2bitString))

        # rows and columns lists for diagonal positions in H1
        diagLocationH1 = [i for i in range(basisLen)]

        # creates sparse matrix with diagonals as diagH1 list
        H1 = csr_matrix((diagH1, (diagLocationH1, diagLocationH1)), shape=[basisLen, basisLen])
        H1 = qt.Qobj(H1)
    else:
        # create H1_list operator
        H1_list = []
        z2bitString = 2 * np.array([int(b) for b in z2_initial(N)]) - 1
        for r in range(N):
            diagHr = []
            for i in range(basisLen):
                bitString = 2 * np.array([int(b) for b in basisList[i]]) - 1
                diagHr.append(bitString[r] * z2bitString[r])

            Hr = csr_matrix((diagHr, (range(basisLen), range(basisLen))), shape=(basisLen, basisLen))
            H1_list.append(qt.Qobj(Hr))

    if not random_drive_freq:
        return H0, H1, eigenvalues, eigenstates, psi0, basisList
    else:
        return H0, H1_list, eigenvalues, eigenstates, psi0, basisList

def get_random_freq_scar_ham(N):
    assert (N % 2 == 0), "N must be a multiple of 2"

    basisList = binNoConsecOnesEfficient(N)
    for basis in basisList:
        if basis[0] == '1' and basis[-1] == '1':
            basisList.remove(basis)
        
    basisMap = {bitStr: i for i, bitStr in enumerate(basisList)}
    basisLen = len(basisList)
    flippedList = []
    ohms = 1.0

    rowBare = []
    columnBare = []

    rowFactor = []
    columnFactor = []

    # flip bit hashmap
    flipMap = {'0': '1', '1': '0'}

    # sigma z op hashmap
    sigzMap = {'0': '-1', '1': '1'}

    # list of ints for Hamiltonian
    numList = []

    # -------------------------------
    #
    # create the bare PXP hamiltonian
    #
    # -------------------------------
    for i in range(basisLen):

        # add padding so that search doesnt go out of range
        paddedBitStr = basisList[i][-1] + basisList[i] + basisList[i][0]
        copyBit = list(paddedBitStr)

        # apply the sum of r P_r-1 * sigma_x * P_r+1 operator
        for j in range(1, N+1):
            
            if paddedBitStr[j-1] == '0' and paddedBitStr[j+1] == '0':
                copyBit[j] = flipMap[paddedBitStr[j]]
                flippedList.append(''.join(copyBit)[1:-1])
                copyBit = list(paddedBitStr)
            
        # adds row and column values for the sparse matrix
        for k in range(len(flippedList)):
            rowBare.append(basisMap[flippedList[k]])
            columnBare.append(i)
            
        flippedList.clear()

    # -------------------------------
    #
    # create the sigma Z PXP hamiltonian perturbation
    #
    # -------------------------------
    for i in range(basisLen):

        # add padding so that search doesnt go out of range
        paddedBitStr = basisList[i][-2] + basisList[i][-1] + basisList[i] + basisList[i][0] + basisList[i][1]
        copyBit = list(paddedBitStr)
        factor = 1

        # apply the PXP operator
        for j in range(2, N+2):
            
            if (paddedBitStr[j-1] == '0') and (paddedBitStr[j+1] == '0'):
                copyBit[j] = flipMap[paddedBitStr[j]]

                # apply sigmaZ_r-2 + sigmaZ_r+2
                r_neg2 = int(sigzMap[paddedBitStr[j-2]])
                r_pos2 = int(sigzMap[paddedBitStr[j+2]])
                factor = r_neg2 + r_pos2
                numList.append(factor)

                flippedList.append(''.join(copyBit)[2:-2])
                copyBit = list(paddedBitStr)
            
        # adds row and column values for the sparse matrix
        for k in range(len(flippedList)):
            rowFactor.append(basisMap[flippedList[k]])
            columnFactor.append(i)
            
        flippedList.clear()

    # list of ones for the sparse matrix
    onesList = np.ones(len(rowBare), dtype=int)

    # create the sparse matrix and turn it into a Qobj
    sparseBareHamiltonian = csr_matrix((onesList, (rowBare, columnBare)), shape=[basisLen, basisLen])
    sparseFactoredHamiltonian = csr_matrix((numList, (rowFactor, columnFactor)), shape=[basisLen, basisLen])
    H0 = (ohms / 2 * sparseBareHamiltonian) + (-0.026 * ohms * sparseFactoredHamiltonian)
    H0 = qt.Qobj(H0)

    # diagonalize the sparse matrix
    eigenvalues, eigenstates = H0.eigenstates()

    # initial state
    z2_str = z2_initial(N)
    z2_index = basisMap[z2_str]
    psi0 = qt.basis(basisLen, z2_index)

    # -------------------------------
    #
    # create the driving hamiltonian
    #
    # -------------------------------

    # create H1 operator for QobjEvo!
    H1_list = []
    z2bitString = 2 * np.array([int(b) for b in z2_initial(N)]) - 1
    for r in range(N):
        diagHr = []
        for i in range(basisLen):
            bitString = 2 * np.array([int(b) for b in basisList[i]]) - 1
            diagHr.append(bitString[r] * z2bitString[r])

        Hr = csr_matrix((diagHr, (range(basisLen), range(basisLen))), shape=(basisLen, basisLen))
        H1_list.append(qt.Qobj(Hr))

    return H0, H1_list, eigenvalues, eigenstates, psi0, basisList

def get_qubit_ham(N, wm=1.0, disorder=False, detuning=0.0, random_seed=False):
    if not random_seed:
        np.random.seed(0)
    if disorder:
        wlist = np.random.uniform(-detuning, detuning, N)
        wlist -= np.mean(wlist)

    sigz = qt.sigmaz()
    sigx = qt.sigmax()
    eye = qt.qeye(2)

    eyeList = [eye] * N

    qH0 = 0
    qH1 = 0

    for i in range(N):
        ops0 = eyeList.copy()
        ops1 = eyeList.copy()

        ops0[i] = -0.5 * wm * sigz
        if disorder:
            ops0[i] += wlist[i] * sigz
        ops1[i] = sigx

        qH0 += qt.tensor(ops0)
        qH1 += qt.tensor(ops1)

    return qH0, qH1

def get_qubit_ham_x(N, detuning=0.0, wm=1.0):
    np.random.seed(0)
    d = detuning
    diag_detune = np.random.uniform(-d, d, N)
    diag_detune -= np.mean(diag_detune)

    sigz = qt.sigmaz()
    sigx = qt.sigmax()
    eye = qt.qeye(2)

    eyeList = [eye] * N

    qH0 = 0
    qH1 = 0

    for i in range(N):
        ops0 = eyeList.copy()
        ops1 = eyeList.copy()

        ops0[i] = -0.5 * wm * sigz + diag_detune[i] * sigx
        ops1[i] = sigx

        qH0 += qt.tensor(ops0)
        qH1 += qt.tensor(ops1)

    return qH0, qH1, wm

def get_random_freq_qubit_ham(N, wm=1.0):
    sigz = qt.sigmaz()
    sigx = qt.sigmax()
    eye = qt.qeye(2)

    qH0 = 0
    qH1_list = []

    for i in range(N):
        ops0 = [eye] * N
        ops1 = [eye] * N

        ops0[i] = -0.5 * wm * sigz
        ops1[i] = sigx

        qH0 += qt.tensor(ops0)
        qH1_list.append(qt.tensor(ops1))

    return qH0, qH1_list, wm