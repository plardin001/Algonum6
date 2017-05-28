from method import *
import math as m
import matplotlib.pyplot as plt
from afficher import *

#####TEST_METHOD####


def f1(y,t):
    return(y/(1+t**2))

#solution of y'(t)=f1(y,t) and y(0)=1
def y1(t):
    return m.exp(m.atan(t))

def f2(Y,t):
    return([[-Y[1]],[Y[0]]])

#solution of y'(t)=f2[Y,t] and y(0)
    

def test_meth_n_step():
    y0=[1]
    t0=0
    tf = 10
    eps = 0.0001


    h=0.001
    N = (tf - t0) // h
    N = int(N)

    X = np.arange(t0, t0+N*h, h)

    EULER = meth_n_step(y0, t0, N, h, f1, step_euler)
    MID = meth_n_step(y0, t0, N, h, f1, step_mid_point)
    HEUN = meth_n_step(y0, t0, N, h, f1, step_heun)
    RK4 = meth_n_step(y0, t0, N, h, f1, step_rk4)
    Y = np.zeros(N)
    for i in range(N):
        Y[i]=y1(X[i])
    plt.plot(X,Y, label='Solution')
    plt.plot(X,EULER, label='Euler')
    plt.plot(X,MID, label='Middle Point')
    plt.plot(X,HEUN, label='Heun')
    plt.plot(X,RK4, label='Rk4')
    plt.title("test_meth_n_step sur f1 avec "+str(N)+" pas")
    plt.legend()
    plt.show()
    return

def test_meth_epsilon():
    y0=[1]
    t0=0
    tf = 10
    eps = 0.1

    (XE,EULER) = meth_epsilon(y0, t0, tf, eps, f1, step_euler)
    (XM,MID) = meth_epsilon(y0, t0, tf, eps, f1, step_mid_point)
    (XH,HEUN) = meth_epsilon(y0, t0, tf, eps, f1, step_heun)
    (XR,RK4) = meth_epsilon(y0, t0, tf, eps, f1, step_rk4)

    plt.plot(XE,EULER, label='Euler')
    plt.plot(XM,MID, label='Middle Point')
    plt.plot(XH,HEUN, label='Heun')
    plt.plot(XR,RK4, label='Rk4')
    plt.title("test_meth_epsilon sur f1 avec epsilon = "+str(eps))
    plt.legend()
    plt.show()
    return

def test_print_field():
    print_field(f1,[1],0,"Tangentes à f1",step_euler)


test_meth_n_step() 
test_meth_epsilon()
test_print_field()

