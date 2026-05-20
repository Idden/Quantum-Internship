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

def get_C_AB_matrix(state, basisList, N):

    NA = N // 2
    NB = N - NA

    C_AB = np.zeros((2**NA, 2**NB), dtype=complex)

    vec = state.full().flatten()

    for k, bitstr in enumerate(basisList):
        A_bits = bitstr[:NA]
        B_bits = bitstr[NA:]

        i = int(A_bits, 2)
        j = int(B_bits, 2)

        C_AB[i, j] = vec[k]

    return C_AB

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
    
def get_Hy(N, basisList):

    basisLen = len(basisList)
    basisMap = {bitStr: i for i, bitStr in enumerate(basisList)}

    hy = [(-1)**i for i in range(N)]

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

    return Hy


def get_qubit_ham(N, wm=1.0, ham_disorder=[0, 0, 0], N_dis=None, fixed_seed=False, ds_dis=0.0, sigz_ham=False):
    if fixed_seed:
        np.random.seed(0)

    if N_dis == None:
        N_dis = N

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
    ds += 1.0

    sigz = qt.sigmaz()
    sigy = qt.sigmay()
    sigx = qt.sigmax()

    qH0_list = []
    qH1_list = []

    for i in range(N):

        if sigz_ham:
            ops0 = -0.5 * wm * sigz
            ops1 = sigx
        else:
            ops0 = -0.5 * wm * sigx
            ops1 = sigz

        if ham_disorder[0] != 0.0:
            dz = hz[i] * sigz
            ops0 += dz
        if ham_disorder[1] != 0.0:
            dy = hy[i] * sigy
            ops0 += dy
        if ham_disorder[2] != 0.0:
            dx = hx[i] * sigx
            ops0 += dx
        
        qH0_list.append(ops0)
        qH1_list.append(ops1)
            
    return qH0_list, qH1_list


def get_zero_scar(N):

    N2 = N // 2

    Hx, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
    Hy = get_Hy(N, basisList)
    Hz, _ = get_scar_H1(N, basisList)

    xeigvals = Hx.eigenenergies()
    yeigvals = Hy.eigenenergies()
    zeigvals = Hz.eigenenergies()

    Hx = Hx / np.max(xeigvals) * N2
    Hy = Hy / np.max(yeigvals) * N2
    Hz = Hz / np.max(zeigvals) * N2

    xeigvals, xeigstates = Hx.eigenstates()
    yeigvals, yeigstates = Hy.eigenstates()
    zeigvals, zeigstates = Hz.eigenstates()

    # ----------------------------
    # Find zero-energy subspace of Hx
    # ----------------------------

    threshold = 1e-14

    zeros_eigenstates = []

    for i, energy in enumerate(xeigvals):
        if abs(energy) < threshold:
            zeros_eigenstates.append(xeigstates[i])

    if len(zeros_eigenstates) == 0:
        raise ValueError("No zero-energy states found. Try increasing threshold.")

    # P has rows = zero-energy basis vectors
    P = []

    for state in zeros_eigenstates:
        P.append(state.full().flatten())

    P = np.array(P)

    # ----------------------------
    # Build projected angular momentum S^2
    #
    # Important:
    # Use P (Sx^2 + Sy^2 + Sz^2) P^\dagger
    # NOT (P Sx P^\dagger)^2 + ...
    # ----------------------------

    S2_full = (
        Hx.full() @ Hx.full()
        + Hy.full() @ Hy.full()
        + Hz.full() @ Hz.full()
    )

    S2_zeroes = np.conj(P) @ S2_full @ P.T

    S2 = qt.Qobj(S2_zeroes)

    seigvals, seigstates = S2.eigenstates()

    # ----------------------------
    # Take the maximum-S2 subspace
    # ----------------------------

    s_tol = 1e-10
    max_s_val = seigvals[-1]

    max_s_states = []

    for i, val in enumerate(seigvals):
        if abs(val - max_s_val) < s_tol:
            max_s_states.append(seigstates[i])

    # ----------------------------
    # Reconstruct max-S2 states back into full constrained Hilbert space
    # ----------------------------

    candidates = []

    for s_state in max_s_states:
        candidate_np = s_state.full().flatten() @ P
        candidate = qt.Qobj(candidate_np)
        candidate = candidate / candidate.norm()
        candidates.append(candidate)

    # ----------------------------
    # Pick the Z2-visible state inside the max-S2 scar manifold
    #
    # This is NOT projecting Z2 into the full zero-energy subspace.
    # This projects Z2 only into the angular-momentum-selected max-S2 subspace.
    # ----------------------------

    scar = 0 * candidates[0]

    for candidate in candidates:
        coeff = candidate.dag() * psi0
        scar += coeff * candidate

    if scar.norm() < 1e-14:
        print("WARNING: Z2 has almost zero overlap with the max-S2 zero-energy subspace.")
        print("Try checking operator definitions or degeneracies.")
    else:
        scar = scar / scar.norm()

    z2_overlap = np.abs(psi0.dag() * scar) ** 2

    return scar, z2_overlap