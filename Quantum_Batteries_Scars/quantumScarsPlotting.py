import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip import Qobj

def plotEigEnergies(H):
    
    if not isinstance(H, Qobj):
        print("Matrix must be Qobj")
        return 1

    eigenvalues = H.eigenenergies()

    plt.figure()
    plt.scatter(eigenvalues, [0 for i in range(len(eigenvalues))])
    plt.show()

def plotAmpEigenstatesZ2(H, z2Ket):

    if not isinstance(H, Qobj):
        print("Matrix must be Qobj")
        return 1

    amplitudes = []

    eigenvalues, eigenstates = H.eigenstates()

    for states in eigenstates:
        amplitudes.append(z2Ket.dag() * states)

    plt.figure()
    plt.plot(eigenvalues, np.abs(amplitudes)**2, ".")
    plt.yscale("log")
    plt.show()