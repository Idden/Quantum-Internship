import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
from qutip import Qobj

mpl.rcParams["font.size"] = 12

def plotEigEnergies(H, N):
    
    # if not isinstance(H, Qobj):
    #     print("Matrix must be Qobj")
    #     return 1

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

    # if not isinstance(H, Qobj):
    #     print("Matrix must be Qobj")
    #     return 1

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

    # if not isinstance(H, Qobj):
    #     print("Matrix must be Qobj")
    #     return 1

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