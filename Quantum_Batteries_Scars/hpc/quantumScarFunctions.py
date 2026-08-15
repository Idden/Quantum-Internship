import math
import numpy as np
import scipy.sparse as sp
import qutip as qt
from scipy.sparse.linalg import eigsh, splu, LinearOperator

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


def get_scar_ham(N, fixed_seed=False, ohms=1.0, diagonalize=True):
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
    sparseBareHamiltonian = sp.csr_matrix((onesList, (rowBare, columnBare)), shape=[basisLen, basisLen])
    sparseFactoredHamiltonian = sp.csr_matrix((numList, (rowFactor, columnFactor)), shape=[basisLen, basisLen])
    H0 = (ohms / 2 * sparseBareHamiltonian) + (-0.026 * ohms * sparseFactoredHamiltonian)
    H0 = qt.Qobj(H0)

    # -------------------------------
    #
    # states and evolutions set ups
    #
    # -------------------------------

    # diagonalize the sparse matrix
    if diagonalize:
        eigenvalues, eigenstates = H0.eigenstates()
    else:
        eigenvalues, eigenstates = None, None

    # initial state
    z2_str = z2_initial(N)
    z2_index = basisMap[z2_str]
    psi0 = qt.basis(basisLen, z2_index)

    return H0, eigenvalues, eigenstates, psi0, basisList


def get_dis_scar_ham(H0_dis, N, basisList, N_dis=None, ham_disorder=[0, 0, 0], fixed_seed=False, diagonalize=True):
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
        Hz = qt.Qobj(sp.csr_matrix((dataZ, (pert_location, pert_location)), shape=[basisLen, basisLen]))
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

        Hy = qt.Qobj(sp.csr_matrix((dataY, (rowY, colY)), shape=(basisLen, basisLen)))
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

        Hx = qt.Qobj(sp.csr_matrix((dataX, (rowX, colX)), shape=(basisLen, basisLen)))
        H0_dis = H0_dis + Hx

    H0_dis = qt.Qobj(H0_dis)

    if diagonalize:
        eigenvalues, eigenstates = H0_dis.eigenstates()
    else:
        eigenvalues, eigenstates = None, None

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

        H1 = sp.csr_matrix(
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

            Hr = sp.csr_matrix(
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

    Hy = qt.Qobj(sp.csr_matrix((dataY, (rowY, colY)), shape=(basisLen, basisLen)))

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

    driveWeights = np.ones(N)
    if ds_dis != 0.0:
        dis_sites = np.random.choice(N, size=N_dis, replace=False)
        driveWeights[dis_sites] += np.random.uniform(-ds_dis, ds_dis, N_dis)

    sigz = qt.sigmaz()
    sigy = qt.sigmay()
    sigx = qt.sigmax()

    qH0_list = []
    qH1_list = []

    for i in range(N):

        if sigz_ham:
            ops0 = -0.5 * wm * sigz
            ops1 = driveWeights[i] * sigx
        else:
            ops0 = -0.5 * wm * sigx
            ops1 = driveWeights[i] * sigz

        if ham_disorder[0] != 0.0:
            ops0 += hz[i] * sigz
        if ham_disorder[1] != 0.0:
            ops0 += hy[i] * sigy
        if ham_disorder[2] != 0.0:
            ops0 += hx[i] * sigx

        qH0_list.append(ops0)
        qH1_list.append(ops1)

    return qH0_list, qH1_list, driveWeights

def get_zero_scar(N, k0=None):

    def to_scipy(H):
        # qutip 4 stores scipy sparse directly, qutip 5 wraps it
        data = getattr(H, "data", None)
        if sp.issparse(data):
            return data.tocsr()
        if hasattr(data, "as_scipy"):
            return data.as_scipy().tocsr()
        return sp.csr_matrix(H.full())

    def max_eig(H):
        return float(eigsh(H, k=1, which="LA", return_eigenvectors=False, tol=0)[0].real)

    N2 = N // 2

    H0_clean, _, _, psi0, basisList = get_scar_ham(N, diagonalize=False)
    D = len(basisList)

    Hx = to_scipy(H0_clean).astype(complex)
    Hy = to_scipy(get_Hy(N, basisList)).astype(complex)
    Hz = to_scipy(get_scar_H1(N, basisList)[0]).astype(complex)

    Hx = Hx * (N2 / max_eig(Hx))
    Hy = Hy * (N2 / max_eig(Hy))
    Hz = Hz * (N2 / max_eig(Hz))

    # ---- null space of Hx --------------------------------------------
    H2 = (Hx @ Hx).tocsc()

    # one LU, reused by every ARPACK restart. this is the expensive step.
    lu = splu((H2 + 1e-9 * sp.eye(D, format="csc", dtype=complex)).tocsc())
    OPinv = LinearOperator((D, D), matvec=lu.solve, dtype=complex)

    # start generously: the zero-mode count is ~1-2% of D and each restart
    # costs a full ARPACK solve
    K = k0 if k0 is not None else max(16, int(0.02 * D))

    while True:
        w, v = eigsh(H2, k=K, sigma=-1e-9, which="LM", OPinv=OPinv)
        nz = np.linalg.norm(Hx @ v, axis=0) < 1e-8   # residual, not the squared eigenvalue
        if nz.sum() < K:
            break
        K *= 2

    V, _ = np.linalg.qr(v[:, nz])                    # ARPACK basis is not orthonormal here

    # ---- S^2 inside the null space, without a dense DxD --------------
    S2 = sum((M @ V).conj().T @ (M @ V) for M in (Hx, Hy, Hz))

    sv, ss = np.linalg.eigh(S2)
    keep = np.abs(sv - sv[-1]) < 1e-10
    cand = V @ ss[:, keep]                           # D x n_max, orthonormal

    z2 = psi0.full().ravel().astype(complex)
    scar = cand @ (cand.conj().T @ z2)               # project Z2 into the max-S^2 manifold

    norm = np.linalg.norm(scar)
    if norm < 1e-14:
        raise ValueError("Z2 has essentially no overlap with the max-S2 zero-energy subspace")

    scar = scar / norm
    z2_overlap = float(np.abs(np.vdot(z2, scar)) ** 2)

    return qt.Qobj(scar.reshape(-1, 1)), z2_overlap