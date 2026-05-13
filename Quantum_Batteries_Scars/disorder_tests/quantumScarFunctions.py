import math
import numpy as np
from scipy.sparse import csr_matrix
import qutip as qt
import matplotlib.pyplot as plt

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

def embed_scar_state_to_full(state, basisList, N):
    vec_constrained = state.full().flatten()
    vec_full = np.zeros(2**N, dtype=complex)

    for i, bitstr in enumerate(basisList):
        full_index = int(bitstr, 2)
        vec_full[full_index] = vec_constrained[i]

    return qt.Qobj(vec_full, dims=[[2]*N, [1]*N])

def giveMeScarVonNeumannEntrop(N, wd, tlist, disorder=[0, 0, 0], reals=50):
    scarEntangle = []
    for _ in range(reals):
        H0_clean, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
        H0, eigenvalues, eigenstates = get_dis_scar_ham(
            H0_clean,
            N,
            basisList,
            ham_disorder=disorder,
            fixed_seed=False
        )
        H1, driveWeights = get_scar_H1(N, basisList)

        args = {"A": 0.1, "omega": wd}
        H = qt.QobjEvo([H0, [H1, coeff]], args=args)
        psi_t = qt.sesolve(H, eigenstates[0], tlist)

        temp = []
        for state in psi_t.states:
            psi_full = embed_scar_state_to_full(state, basisList, N)
            rho_A = psi_full.ptrace(list(range(N//2)))
            temp.append(qt.entropy_vn(rho_A))
        scarEntangle.append(temp)

    scarEntangle = np.array(scarEntangle)
    plotScar = np.mean(scarEntangle, axis=0)

    plt.plot(tlist, plotScar)
    plt.title(f"Avged Thingamabob w/ {N} Qubits and {disorder} Disorder")
    plt.ylabel("Von Neumann Entropy")
    plt.xlabel("Time")
    plt.show()

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
    return lambda t, args: args["A"] * np.sin(args[f"wd{r}"] * t)


def get_scar_ham(N, fixed_seed=False, ohms=1.0):
    assert (N % 2 == 0), "N must be a multiple of 2"

    if fixed_seed:
        np.random.seed(0)

    basisList = binNoConsecOnesEfficient(N)
    basisList = [
        basis for basis in basisList
        if not (basis[0] == '1' and basis[-1] == '1')
    ] # rydberg blockade

    basisMap = {bitStr: i for i, bitStr in enumerate(basisList)}
    basisLen = len(basisList)
    flippedList = []

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

    return H0, eigenvalues, eigenstates, psi0, basisList
    
def get_qubit_ham(N, wm=1.0, fixed_seed=False, indv_qubit=False, ds_dis=0.0, sigz_ham=False):
    if fixed_seed:
        np.random.seed(0)

    ds = np.random.uniform(-ds_dis, ds_dis, N)
    ds += 1.0

    sigz = qt.sigmaz()
    sigy = qt.sigmay()
    sigx = qt.sigmax()
    eye = qt.qeye(2)

    eyeList = [eye] * N

    qH0 = 0
    qH1 = 0
    qH1_list = []

    for i in range(N):
        ops0 = eyeList.copy()
        ops1 = eyeList.copy()

        if sigz_ham:
            ops0[i] = -0.5 * wm * sigz
            ops1[i] = ds[i] * sigx
        else:
            ops0[i] = -0.5 * wm * sigx
            ops1[i] = ds[i] * sigz

        qH0 += qt.tensor(ops0)

        if not indv_qubit:
            qH1 += qt.tensor(ops1)
        else:
            qH1_list.append(qt.tensor(ops1))

    eigenvalues, eigenstates = qH0.eigenstates()

    if not indv_qubit:
        return qH0, qH1, eigenvalues, eigenstates
    else:
        return qH0, qH1_list, eigenvalues, eigenstates


def get_dis_scar_ham(H0_dis, N, basisList, N_dis=None, ham_disorder=[0, 0, 0], fixed_seed=False):
    if fixed_seed:
        np.random.seed(0)

    if N_dis == None:
        N_dis = N

    basisLen = len(basisList)
    basisMap = {bitStr: i for i, bitStr in enumerate(basisList)}

    if ham_disorder[0] != 0.0:
        zd = ham_disorder[0]
        dataZ = []

        hz = np.zeros(N)
        dis_sites = np.random.choice(N, size=N_dis, replace=False)
        hz[dis_sites] = np.random.uniform(-zd, zd, N_dis)
        print(hz)

        intBasisList = []
        for i in range(basisLen):
            intBasisList.append(2 * np.array([int(k) for k in basisList[i]]) - 1)

        for i in range(basisLen):
            dataZ.append(np.dot(intBasisList[i], hz))

        pert_location = list(range(basisLen))
        Hz = qt.Qobj(csr_matrix((dataZ, (pert_location, pert_location)), shape=[basisLen, basisLen]))
        H0_dis = H0_dis + Hz
    
    if ham_disorder[1] != 0.0:
        yd = ham_disorder[1]
        hy = np.zeros(N)
        dis_sites = np.random.choice(N, size=N_dis, replace=False)
        hy[dis_sites] = np.random.uniform(-yd, yd, N_dis)

        rowY, colY, dataY = [], [], []

        for i, s in enumerate(basisList):
            s_list = list(s)
            for r in range(N):
                flipped = s_list.copy()
                flipped[r] = '1' if s[r] == '0' else '0'
                flipped_str = ''.join(flipped)

                if flipped_str in basisMap:
                    j = basisMap[flipped_str]

                    phase = 1j if s[r] == '0' else -1j
                    rowY.append(j)
                    colY.append(i)
                    dataY.append(hy[r] * phase)

        Hy = qt.Qobj(csr_matrix((dataY, (rowY, colY)), shape=(basisLen, basisLen)))
        H0_dis = H0_dis + Hy

    if ham_disorder[2] != 0.0:
        xd = ham_disorder[2]
        hx = np.zeros(N)
        dis_sites = np.random.choice(N, size=N_dis, replace=False)
        hx[dis_sites] = np.random.uniform(-xd, xd, N_dis)

        rowX, colX, dataX = [], [], []

        for i, s in enumerate(basisList):
            s_list = list(s)
            for r in range(N):
                flipped = s_list.copy()
                flipped[r] = '1' if s[r] == '0' else '0'
                flipped_str = ''.join(flipped)

                if flipped_str in basisMap:
                    j = basisMap[flipped_str]
                    rowX.append(j)
                    colX.append(i)
                    dataX.append(hx[r])

        Hx = qt.Qobj(csr_matrix((dataX, (rowX, colX)), shape=(basisLen, basisLen)))
        H0_dis = H0_dis + Hx

    H0_dis = qt.Qobj(H0_dis)
    eigenvalues, eigenstates = H0_dis.eigenstates()

    return H0_dis, eigenvalues, eigenstates

def get_dis_qubit_ham(qH0_dis, N, N_dis=None, ham_disorder=[0, 0, 0], fixed_seed=False):
    if N_dis == None:
        N_dis = N

    if fixed_seed:
        np.random.seed(0)

    if ham_disorder[0] != 0.0:
        zd = ham_disorder[0]
        hz = np.zeros(N)
        dis_sites = np.random.choice(N, size=N_dis, replace=False)
        hz[dis_sites] = np.random.uniform(-zd, zd, N_dis)

    if ham_disorder[1] != 0.0:
        yd = ham_disorder[1]
        hy = np.zeros(N)
        dis_sites = np.random.choice(N, size=N_dis, replace=False)
        hy[dis_sites] = np.random.uniform(-yd, yd, N_dis)

    if ham_disorder[2] != 0.0:
        xd = ham_disorder[2]
        hx = np.zeros(N)
        dis_sites = np.random.choice(N, size=N_dis, replace=False)
        hx[dis_sites] = np.random.uniform(-xd, xd, N_dis)

    sigz = qt.sigmaz()
    sigy = qt.sigmay()
    sigx = qt.sigmax()
    eye = qt.qeye(2)

    eyeList = [eye] * N

    ham_dis = qt.Qobj(np.zeros((2**N, 2**N)), dims=[[2]*N, [2]*N])

    for i in range(N):
        ops0 = eyeList.copy()

        if ham_disorder[0] != 0.0:
            ops0[i] = hz[i] * sigz

        if ham_disorder[1] != 0.0:
            ops0[i] = hy[i] * sigy

        if ham_disorder[2] != 0.0:
            ops0[i] = hx[i] * sigx
        
        ham_dis += qt.tensor(ops0)

    qH0_dis += ham_dis

    qeigenvalues, qeigenstates = qH0_dis.eigenstates()

    return qH0_dis, qeigenvalues, qeigenstates

def get_scar_H1(N, basisList, ds_dis=0.0, N_dis=None, fixed_seed=False, indv_qubit=False):
    if fixed_seed:
        np.random.seed(0)

    if N_dis is None:
        N_dis = N

    basisLen = len(basisList)

    # default no-disorder drive weights
    driveWeights = np.ones(N)

    # choose which sites get drive-strength disorder
    if ds_dis != 0.0:
        dis_sites = np.random.choice(N, size=N_dis, replace=False)
        driveWeights[dis_sites] += np.random.uniform(-ds_dis, ds_dis, N_dis)

    # Z2 staggered sign pattern: 1010... -> +1, -1, +1, -1, ...
    z2bitString = 2 * np.array([int(b) for b in z2_initial(N)]) - 1

    diagLocationH1 = list(range(basisLen))

    if not indv_qubit:
        diagH1 = []

        for i in range(basisLen):
            bitString = 2 * np.array([int(b) for b in basisList[i]]) - 1
            diagH1.append(np.dot(driveWeights * bitString, z2bitString))

        H1 = csr_matrix(
            (diagH1, (diagLocationH1, diagLocationH1)),
            shape=(basisLen, basisLen)
        )

        return qt.Qobj(H1), driveWeights

    else:
        H1_list = []

        for r in range(N):
            diagHr = []

            for i in range(basisLen):
                bitString = 2 * np.array([int(b) for b in basisList[i]]) - 1
                diagHr.append(driveWeights[r] * bitString[r] * z2bitString[r])

            Hr = csr_matrix(
                (diagHr, (diagLocationH1, diagLocationH1)),
                shape=(basisLen, basisLen)
            )

            H1_list.append(qt.Qobj(Hr))

        return H1_list, driveWeights