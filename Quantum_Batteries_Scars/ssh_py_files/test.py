
import numpy as np
import qutip as qt
from multiprocessing import Pool
from GitHub_QM.Quantum_Batteries_Scars.better_ssh_files.quantumScarFunctions import *


xlist = np.logspace(-3, 0, 4)
ylist = np.logspace(-3, 0, 4)
zlist = np.logspace(-3, 0, 4)
dslist = np.linspace(0.01, 5.0, 3) # talk about the resonance
ddlist = np.linspace(0.01, 5.0, 3)
nlist = np.linspace(4, 18, 8)

print(xlist)
