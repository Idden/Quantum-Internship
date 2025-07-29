import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

style.use("dark_background")

# constants
hbar = 1
omega = 1
mass = 1

# discretization
N = 100
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

# eigenvalues and vectors
numStates = 5
eigenvalues, eigenstates = eigsh(H, k=numStates, which='SM')

# plot eigenvalues
plt.scatter(eigenvalues.real, eigenvalues.imag)
plt.title("Complex Plane")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid(True)
plt.show()

# plot eigenstates 0 and 1
plt.plot(x_list, eigenstates[:,0]**2)
plt.plot(x_list, eigenstates[:,1]**2)
plt.show()
