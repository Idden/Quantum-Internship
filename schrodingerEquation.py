import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags

# constants
hbar = 1
omega = 1
mass = 1
x_0 = 3

# sigma
sig = np.sqrt(hbar / (omega * mass))

# discretization
N = 250
x_max = 10
x_list = np.linspace(-x_max, x_max, N)
dx = x_list[1] - x_list[0]

# constants factored out
kineticConstant = -hbar**2 / (2 * mass * dx**2)
potentialConstant = 0.5 * mass * omega**2

# finite difference coefficient into kinetic matrix
main_diag = np.full(N, -2)
off_diag = np.full(N - 1, 1)
kineticMatrix = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1])

# potential matrix
potentialMatrix = diags([x_list**2], [0])

# hamiltonian -> H = K + P
H = kineticConstant * kineticMatrix + potentialConstant * potentialMatrix
H = qt.Qobj(H)

# initial state
psi0 = (1 / (np.pi**0.25 * np.sqrt(sig)) * np.exp(-(x_list - x_0)**2 / (2 * sig**2)))
psi0 = qt.Qobj(psi0)
psi0 = psi0.unit()

# tlist
tlist = np.linspace(0, 5, 100)

# time evolution
evol = qt.sesolve(H, psi0, tlist)

# eigenvalues and vectors
eigenvalues, eigenstates = H.eigenstates()

# plot time evolution
plt.figure()
psi_t = evol.states[0]
plt.plot(x_list, np.abs(psi_t.full().flatten())**2) # full turns Qobj into 2Darray and flatten turns column into row
plt.xlabel("x")
plt.ylabel("Probability")
plt.show()

# plot eigenstates
plt.figure()
energy = 6
psi0_eigen = eigenstates[energy]
waveFunc = psi0_eigen.full().flatten()
plt.plot(x_list, np.abs(waveFunc)**2)
plt.xlabel("x")
plt.ylabel(f"Probability at Energy {energy}")
plt.show()
