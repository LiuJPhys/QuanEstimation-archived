import numpy as np
from qutip import *
from ParameterEstimation.control import grape_CramerRao
#import matplotlib.pyplot as plt


rho_initial = Qobj(np.array([[0.5, 0.5], [0.5, 0.5]]))

M0 = Qobj(0.5 * np.array([[1.0, 1.], [1., 1.]]))
M1 = Qobj(0.5 * np.array([[1.0, -1.], [-1., 1.]]))
M = [M0, M1]

times = np.linspace(0, 10, 100)

epsilon = 0.1

w = 1.
H0 = 0.5 * w * sigmaz()
dH = [0.5 * sigmaz()]

Lvec = [sigmam()]
gamma = [0.1]

Hc = [sigmax(), sigmay(), sigmaz()]

Hc_coeff = [np.array([0. for i in range(0, len(times))]) for k in range(0, len(Hc))]

ctrlgrape = grape_CramerRao.control(H0, rho_initial, times, Lvec, gamma, dH, Hc, Hc_coeff, epsilon)

#ctrlgrape.Run('classical', M)
#ctrlgrape.Run('quantum')