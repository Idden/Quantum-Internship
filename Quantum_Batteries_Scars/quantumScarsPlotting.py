import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip import Qobj

def plotEigEnergies(H):
    
    # if not isinstance(H, Qobj):
    #     print("Matrix must be Qobj")
    #     return 1

    eigenvalues = H.eigenenergies()

    plt.figure()
    plt.plot(eigenvalues, ".")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Energy")
    plt.title("Energies of Eigenvalues")
    plt.savefig("plots//eigEnergies.pdf")
    plt.show()

def plotAmpEigenstatesZ2Lin(H, z2Ket):

    # if not isinstance(H, Qobj):
    #     print("Matrix must be Qobj")
    #     return 1

    amplitudes = []

    eigenvalues, eigenstates = H.eigenstates()

    for states in eigenstates:
        amplitudes.append(z2Ket.dag() * states)

    plt.figure()
    plt.plot(eigenvalues, np.abs(amplitudes) ** 2, ".")
    plt.xlabel("Eigenvalues")
    plt.ylabel("Probability")
    plt.title("Overlap of Z2 State and Eigenstates")
    plt.savefig("plots//ampEigenstateZ2.pdf")
    plt.show()

def plotAmpEigenstatesZ2Log(H, z2Ket):

    # if not isinstance(H, Qobj):
    #     print("Matrix must be Qobj")
    #     return 1

    amplitudes = []

    eigenvalues, eigenstates = H.eigenstates()

    for states in eigenstates:
        amplitudes.append(z2Ket.dag() * states)

    plt.figure()
    plt.plot(eigenvalues, np.abs(amplitudes) ** 2, ".")
    plt.yscale("log")
    plt.ylim(10**-5, 1)
    plt.xlabel("Eigenvalues")
    plt.ylabel("Probability")
    plt.title("Overlap of Z2 State and Eigenstates")
    plt.savefig("plots//ampEigenstateZ2.pdf")
    plt.show()

def plotProbZ2Time(H, z2Ket):

    # if not isinstance(H, Qobj):
    #     print("Matrix must be Qobj")
    #     return 1

    amplitudes = []

    tlist = np.linspace(0, 20, 250)
    psi_t = qt.sesolve(H, z2Ket, tlist)

    for states in psi_t.states:
        amplitudes.append(z2Ket.dag() * states)

    plt.figure()
    plt.plot(tlist, np.abs(amplitudes)**2)
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.title("Overlap of Z2 State with Itself Over Time")
    plt.savefig("plots//ampTimeZ2.pdf")
    plt.show()