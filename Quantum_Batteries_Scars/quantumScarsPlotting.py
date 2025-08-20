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

def plotAmpEigenstatesZ2(H):

    if not isinstance(H, Qobj):
        print("Matrix must be Qobj")
        return 1

    amplitudes = []

    eigenvalues, eigenstates = H.eigenstates()

    z2State = ''.join('1' if i % 2 == 0 else '0' for i in range(len(eigenvalues)))

    for states in eigenstates:
        amplitudes.append( * states)

    plt.figure()
    plt.plot(eigenvalues, np.abs(amplitudes)**2, ".")
    plt.yscale("log")
    plt.show()