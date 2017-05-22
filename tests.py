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
    

def test_meth_epsilon():
    y0=[1]
    t0=0
    tf = 10
    eps = 0.0001
    
    #print(meth_epsilon(y0, t0, tf, eps, f1, step_euler))
    #print(meth_epsilon(y0, t0, tf, eps, f1, step_mid_point))
    #print(meth_epsilon(y0, t0, tf, eps, f1, step_heun))
    #print(meth_epsilon(y0, t0, tf, eps, f1, step_rk4))

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
    plt.legend()
    plt.show()
    return

def test_print_field():
    print_field(f1,[1],0,"Champ",step_euler)

test_print_field()
#test_meth_epsilon()
#show_vector_field(f1,0,1,10,10,0.01,0.01)

