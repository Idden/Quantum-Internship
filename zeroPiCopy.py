import scqubits as scq
import numpy as np
import matplotlib.pyplot as plt

zp_yaml = """# zero-pi
branches:
- [JJ, 1,2, EJ = 50, 2]
- [JJ, 3,4, EJ, 20]
- [L, 2,3, 0.008]
- [L, 4,1, 0.008]
- [C, 1,3, 0.02]
- [C, 2,4, 0.02]
"""
zero_pi = scq.Circuit(zp_yaml, from_file=False)

# eigenvalues and eigenvectors of zero_pi circuit
evals, evecs = zero_pi.eigensys(evals_count=3)
E01 = evals[1] - evals[0]
E12 = evals[2] - evals[1]
anharm = E12 - E01
print(f"E01={E01:.4f} GHz, anharm={anharm:.4f} GHz")

# plot eigen energies
# plt.figure()
# plt.plot(range(len(evals)), evals, "o-")
# plt.xlabel("Energy level index")
# plt.ylabel("Energy (GHz)")
# plt.show()

