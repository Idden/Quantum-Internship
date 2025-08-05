import scqubits as scq
import numpy as np
import matplotlib.pyplot as plt

transmon2 = scq.Transmon(EJ=2.0, EC=1.0, ng=0.3, ncut=31)
transmon10 = scq.Transmon(EJ=10.0, EC=1.0, ng=0.3, ncut=31)
transmon50 = scq.Transmon(EJ=50.0, EC=1.0, ng=0.3, ncut=31)

list = np.linspace(-1, 1, 100)

transmon2.plot_evals_vs_paramvals('ng', list, evals_count=3, subtract_ground=False)
transmon10.plot_evals_vs_paramvals('ng', list, evals_count=3, subtract_ground=False)
transmon50.plot_evals_vs_paramvals('ng', list, evals_count=3, subtract_ground=False)

plt.show()
