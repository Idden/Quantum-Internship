import numpy as np
import qutip as qt
from qutip import Qobj

def isHermitian(M):

    if isinstance(M, np.ndarray):
        return np.allclose(M, M.conj().T)
    
    elif isinstance(M, Qobj):
        return M == M.dag()
    
    else:
        print("Input must be a numpy array or Qobj")

