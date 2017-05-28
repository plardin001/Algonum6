import numpy as np
import method as met

#####GLOBAL VALUES#####

g = 9.81
l = 10
t0 = 0
tf = 100
eps = 10**(-1)

#####PENDULUM EQUATION#####
def pendulum_equation():
    return (lambda Y, t: np.array(Y[1], (-g/l) * np.sin(Y[0])))

####FREQUENCY#########

def pendulum_frequency(theta):
    Y0 = np.array([0, theta])
    f = pendulum_equation()
    solution = met.meth_epsilon(Y0, t0, tf, eps, f, met.step_rk4)
    print (solution)
    return (1)

pendulum_frequency(10)
