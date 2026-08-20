import os
import numpy as np
import qutip as qt
from GitHub_QM.important.hpc.quantumScarFunctions import *

# run this once per N. it is the only place a dense diagonalization happens.
# it saves the clean scar states so xyz_parallel never has to build them again.
N = 12

H0_clean, _, _, psi0, basisList = get_scar_ham(N, diagonalize=False)

# dense eigh on the real array. qutip's .eigenstates() casts to complex AND
# wraps every column in its own Qobj, so it pays the D^2 block ~3x
# (~12 GB at N=20). the clean PXP hamiltonian is real symmetric, so numpy
# on float64 is ~4 GB peak and gives the same vectors.
Hd = np.asarray(H0_clean.full()).real.astype(np.float64)
eigenvalues, V = np.linalg.eigh(Hd)          # V[:, k] is eigenvector k
del Hd

z2 = psi0.full().ravel().real

# one scar per energy section, picked by highest Z2 overlap (same as giveMeScarOverlap)
sections = np.linspace(eigenvalues[0] - 0.5, eigenvalues[-1] + 0.5, N + 2)
scarIndices = []

for i in range(len(sections) - 1):
    eigenSection = [k for k in range(len(eigenvalues))
                    if sections[i] < eigenvalues[k] < sections[i + 1]]

    if len(eigenSection) == 0:
        continue

    overlaps = [np.abs(z2 @ V[:, k])**2 for k in eigenSection]
    scarIndices.append(eigenSection[int(np.argmax(overlaps))])

scarMat = np.ascontiguousarray(V[:, scarIndices].T).astype(complex)
scarEnergies = eigenvalues[np.array(scarIndices)].astype(float)
del V

# replace the E=0 scar with the max-S^2 zero mode.
# argmin|E| instead of len//2: len//2 is only the middle if every section
# was non-empty, which is true for N=8..20 but is not guaranteed.
mid = int(np.argmin(np.abs(scarEnergies)))

zero_scar, z2_overlap = get_zero_scar(N)
scarMat[mid] = zero_scar.full().ravel()
scarEnergies[mid] = 0.0
print(f"scar {mid} of {len(scarIndices)} from get_zero_scar, "
      f"|<Z2|scar>|^2 = {float(z2_overlap):.6f}")

# the scar states must be orthonormal or the summed probability isn't a projector
gram = scarMat.conj() @ scarMat.T
gram_err = np.abs(gram - np.eye(len(gram))).max()
print(f"max |Gram - I| = {gram_err:.2e}")
assert gram_err < 1e-6, "scar states are not orthonormal, the projector is invalid"

z2_weight = float(np.sum(np.abs(scarMat.conj() @ z2.astype(complex))**2))
print(f"{len(scarIndices)} scar states, total Z2 weight in the tower = {z2_weight:.6f}")

os.makedirs("xyz_data", exist_ok=True)
np.savez(f"xyz_data/scar_states_N{N}.npz",
         scarMat=scarMat,
         scarMatC=scarMat.conj(),   # what the overlap in xyz_parallel actually needs
         scarIndices=np.array(scarIndices),
         scarEnergies=scarEnergies)

print(f"saved xyz_data/scar_states_N{N}.npz  (D={scarMat.shape[1]})")