import scqubits as scq

zp_yaml = """# zero-pi
branches:
- [JJ, 1,2, EJ = 10, 20]
- [JJ, 3,4, EJ, 20]
- [L, 2,3, 0.008]
- [L, 4,1, 0.008]
- [C, 1,3, 0.02]
- [C, 2,4, 0.02]
"""

zero_pi = scq.Circuit(zp_yaml, from_file=False)
symHamiltonian = zero_pi.sym_hamiltonian()

