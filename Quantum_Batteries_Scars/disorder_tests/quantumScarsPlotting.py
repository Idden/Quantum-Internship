import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
from qutip import Qobj
from quantumScarFunctions import *

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
        dw = np.random.uniform(-freq_dis, freq_dis, N)
        omega_list = w + dw
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