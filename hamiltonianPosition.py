import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.sparse import diags

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
T = -hbar^2 / (2 * mass * dx^2)
U = 0.5 * mass * omega^2

# finite difference coefficient into kinetic matrix
main_diag = np.full(N, -2)
off_diag = np.full(N - 1, 1)
kineticMatrix = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1])

# potential matrix
potentialMatrix = diags(x_list)

# hamiltonian -> H = K + P
H = T * kineticMatrix + U * potentialMatrix


