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

def giveMeVonNeumannEntrop(N, wd, tlist, disorder=[0, 0, 0], reals=50):
    scarEntangle = []
    for _ in range(reals):
        H0, H1, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N, ham_disorder=disorder, random_seed=True)
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

def get_qubit_ham(N, wm=1.0, ham_disorder=[0, 0, 0], random_seed=False, indv_qubits=False, ds_dis=0.0, N_dis=None):
    assert len(ham_disorder) == 3, "ham_disorder must have 3 values [dz, dy, dx]"

    if N_dis == None:
        N_dis = N

    if not random_seed:
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

    ds = np.random.uniform(-ds_dis, ds_dis, N)
    ds -= np.mean(ds)
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

        ops0[i] = -0.5 * wm * sigz
        ops1[i] = ds[i] * sigx

        if ham_disorder[0] != 0.0:
            ops0[i] += hz[i] * sigz

        if ham_disorder[1] != 0.0:
            ops0[i] += hy[i] * sigy

        if ham_disorder[2] != 0.0:
            ops0[i] += hx[i] * sigx

        qH0 += qt.tensor(ops0)

        if not indv_qubits:
            qH1 += qt.tensor(ops1)
        else:
            qH1_list.append(qt.tensor(ops1))

    if not indv_qubits:
        return qH0, qH1
    else:
        return qH0, qH1_list

def get_scar_ham(N, ham_disorder=[0, 0, 0],
                 random_seed=False, indv_qubit=False,
                 ohms=1.0, ds_dis=0, N_dis=None):
    assert (N % 2 == 0), "N must be a multiple of 2"
    assert (len(ham_disorder) == 3), "ham_disorder must have 3 values [dz, dy, dx]"

    if N_dis == None:
        N_dis = N

    if not random_seed:
        np.random.seed(0)

    basisList = binNoConsecOnesEfficient(N)
    for basis in basisList:
        if basis[0] == '1' and basis[-1] == '1':
            basisList.remove(basis)
        
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

    # -------------------------------
    #
    # create disorder term
    #
    # -------------------------------
    if ham_disorder[0] != 0.0:
        zd = ham_disorder[0]
        dataZ = []

        hz = np.zeros(N)
        dis_sites = np.random.choice(N, size=N_dis, replace=False)
        hz[dis_sites] = np.random.uniform(-zd, zd, N_dis)

        intBasisList = []
        for i in range(basisLen):
            intBasisList.append(2 * np.array([int(k) for k in basisList[i]]) - 1)

        for i in range(basisLen):
            dataZ.append(np.dot(intBasisList[i], hz))

        pert_location = list(range(basisLen))
        Hz = csr_matrix((dataZ, (pert_location, pert_location)), shape=[basisLen, basisLen])
        H0 = H0 + Hz
    
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

        Hy = csr_matrix((dataY, (rowY, colY)), shape=(basisLen, basisLen))
        H0 = H0 + Hy

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

        Hx = csr_matrix((dataX, (rowX, colX)), shape=(basisLen, basisLen))
        H0 = H0 + Hx

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

    # drive strength disorder
    driveWeights = np.random.uniform(-ds_dis, ds_dis, N)
    driveWeights = 1.0 + driveWeights

    if not indv_qubit:
        # create H1 operator for QobjEvo!
        diagH1 = []

        # switches 0s to -1s and keeps 1s the same for the copyBasis
        # appends to diagH1 the dot product between each bit string and the 0 -> -1 Z2 state
        z2bitString = 2 * np.array([int(i) for i in z2_initial(N)]) - 1
        
        for i in range(basisLen):
            bitString = [int(i) for i in basisList[i]]
            diagH1.append(np.dot(driveWeights * (2 * np.array(bitString) - 1), z2bitString))

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
                diagHr.append(driveWeights[r] * bitString[r] * z2bitString[r])

            Hr = csr_matrix((diagHr, (range(basisLen), range(basisLen))), shape=(basisLen, basisLen))
            H1_list.append(qt.Qobj(Hr))

    if not indv_qubit:
        return H0, H1, eigenvalues, eigenstates, psi0, basisList
    else:
        return H0, H1_list, eigenvalues, eigenstates, psi0, basisList
    
def giveMeScarOverlap(N, psi0, tlist, disorder=[0, 0, 0], plot_scars=False, reals=10, args=None):

    H0, H1, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N, ham_disorder=[0,0,0], random_seed=False)

    # find scar indices using overlaps
    sections = np.linspace(eigenvalues[0] - 0.5, eigenvalues[-1] + 0.5, N+2)
    scarIndices = []

    for i in range(len(sections) - 1):

        eigenSection = []

        for k in range(len(eigenvalues)):
            if (eigenvalues[k] > sections[i]) and (eigenvalues[k] < sections[i+1]):
                eigenSection.append(k)

        highestOverlap = np.abs(psi0.dag() * eigenstates[eigenSection[0]]) ** 2
        highestOverlapIndex = eigenSection[0]

        if len(eigenSection) == 1:
            scarIndices.append(eigenSection[0])
            continue
            
        for m in range(1, len(eigenSection)):
            if np.abs(psi0.dag() * eigenstates[eigenSection[m]]) ** 2 > highestOverlap:
                highestOverlap = np.abs(psi0.dag() * eigenstates[eigenSection[m]]) ** 2
                highestOverlapIndex = eigenSection[m]
        
        scarIndices.append(highestOverlapIndex)

    amplitudes = []
    eigenvalueIndices = []

    for i in scarIndices:
        amplitudes.append(psi0.dag() * eigenstates[i])
        eigenvalueIndices.append(eigenvalues[i])

    if plot_scars:
        print(scarIndices)
        plt.plot(eigenvalueIndices, np.abs(amplitudes) ** 2, ".")
        plt.yscale("log")
        plt.ylim(10**-5, 1)
        plt.xlabel("Eigenvalues")
        plt.ylabel("Probability")
        plt.title(f"Overlap of Z2 State and Scar States w/ {disorder} Disorder")
        plt.show()

    totalScarProbs = np.zeros(len(tlist))
    for _ in range(reals):
        H0, H1, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N, ham_disorder=disorder, random_seed=True)
        H = qt.QobjEvo([H0, [H1, coeff]], args=args)
        psi_t = qt.sesolve(H, eigenstates[0], tlist)

        scarProbs = []
        for states in psi_t.states:
            temp = 0
            for scars in scarIndices:
                temp += np.abs(eigenstates[scars].dag() * states)**2
            scarProbs.append(temp)
        totalScarProbs += scarProbs
    totalScarProbs = totalScarProbs / reals

    plt.plot(tlist, totalScarProbs)
    plt.ylim(0, 1.05)
    plt.xlabel("Time")
    plt.ylabel("Total Scar Probability")
    plt.title(f"Overlap of Psi_t and Scar States w/ {disorder} Disorder")
    plt.show()

    return scarIndices