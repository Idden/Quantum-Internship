import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
from disorder_tests.quantumScarFunctions import *

mpl.rcParams["font.size"] = 12

def plotEigEnergies(H, N):

    eigenvalues = H.eigenenergies()

    plt.figure()
    plt.plot(eigenvalues, ".", ms=12)
    plt.grid(True)
    plt.xlabel("Eigenvalue")
    plt.ylabel("Energy")
    plt.grid(True, alpha=0.4)
    plt.title(f"Energies of Eigenvalues N={N}")
    plt.show()

def plotAmpEigenstatesZ2Log(H, z2Ket, N):

    amplitudes = []

    eigenvalues, eigenstates = H.eigenstates()

    for states in eigenstates:
        amplitudes.append(z2Ket.dag() * states)

    plt.figure()
    plt.plot(eigenvalues, np.abs(amplitudes) ** 2, ".", ms=12)
    plt.yscale("log")
    plt.xlabel("Eigenvalues")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.4)
    plt.ylim(10**-5, 1)
    plt.title(f"Overlap of Z2 State and Eigenstates N={N}")
    plt.show()

def plotProbZ2Time(H, N, z2Ket, t=20):

    amplitudes = []

    tlist = np.linspace(0, t, t*25)
    psi_t = qt.sesolve(H, z2Ket, tlist)

    for states in psi_t.states:
        amplitudes.append(z2Ket.dag() * states)

    plt.figure()
    plt.plot(tlist, np.abs(amplitudes)**2)
    plt.grid(True, alpha=0.4)
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.title(f"Overlap of Z2 State with Itself Over Time N={N}")
    plt.show()

def Rtau_plot(H0, H1, N, w=None, indv_qubit=False, freq_dis=0.0, args=None, t=100):
    assert (freq_dis == 0 or indv_qubit == True), "freq_dis will do NOTHING when indv_qubit is false"
    assert (len(args) == 2 or indv_qubit == True), "args must be len = 2 if indv_qubit = False"
    assert (len(args) == 1 or indv_qubit == False), "args must be len = 1 if indv_qubit = True"

    tlist = np.linspace(0, t, t * 2)
    eigenvalues, eigenstates = H0.eigenstates()
    bandwidth = eigenvalues[-1] - eigenvalues[0]

    if not indv_qubit:
        H = qt.QobjEvo([H0, [H1, coeff]], args=args)
        psi_t = qt.sesolve(H, eigenstates[0], tlist, e_ops=[H0])
        Rtau_scar = np.array(np.real(psi_t.expect[0] - psi_t.expect[0][0]) / bandwidth)
    else:
        dw = np.random.uniform(-freq_dis, freq_dis, N)
        omega_list = w + dw
        H = [H0]
        for r in range(N):
            args[f"wd{r}"] = omega_list[r]
            H.append([H1[r], make_coeff(r)])

        H = qt.QobjEvo(H, args=args)
        psi_t = qt.sesolve(H, eigenstates[0], tlist, e_ops=[H0])
        Rtau_scar = np.array(np.real(psi_t.expect[0] - psi_t.expect[0][0]) / bandwidth)

    plt.plot(tlist, Rtau_scar)
    plt.xlabel("Time")
    plt.ylabel("Rtau")
    plt.title("Rtau vs. Time")
    plt.ylim(0, 1)
    plt.show()

def giveMeScarOverlap(N, psi0, tlist, disorder=[0, 0, 0], plot_scars=False, reals=50, args=None):

    H0, H1, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
    H0, eigenvalues, eigenstates = get_dis_scar_ham(H0, N, basisList, ham_disorder=disorder)

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
        H0, H1, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
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