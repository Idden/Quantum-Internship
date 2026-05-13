
import numpy as np
import qutip as qt
from multiprocessing import Pool
from quantumScarFunctions import *


xlist = np.linspace(0, 0.5, 11)
ylist = np.linspace(0, 0.5, 11)
zlist = np.linspace(0, 0.5, 11)
dslist = np.linspace(0, 2.0, 21)
ddlist = np.linspace(0, 0.5, 11)
nlist = np.linspace(4, 18, 8)

parameter_sweep = []

for N in nlist:
    for x in xlist:
        for y in ylist:
            for z in zlist:
                for ds in dslist:
                    for dd in ddlist:
                        parameter_sweep.append((int(N), x, y, z, ds, dd))

print(parameter_sweep)
