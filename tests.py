from method import *
from lotka_volterra import *
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
    return(np.array([-Y[1],Y[0]]))

#solution of y'(t)=f2[Y,t] and y(0)
def y2(t):
    return np.array([np.cos(t),np.sin(t)])

def test_meth_n_step():
    y0=[1]
    t0=0
    tf = 10
    eps = 0.0001


    h=0.1
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

def test_meth_epsilon2():
    y0 = np.array([1,0])
    t0 = 0
    tf = 10
    eps = 0.1
    
    (XE,EULER) = meth_epsilon(y0, t0, tf, eps, f2, step_euler)
    (XM,MID) = meth_epsilon(y0, t0, tf, eps, f2, step_mid_point)
    (XH,HEUN) = meth_epsilon(y0, t0, tf, eps, f2, step_heun)
    (XR,RK4) = meth_epsilon(y0, t0, tf, eps, f2, step_rk4)

    EULER1,EULER2 = [],[]
    MID1,MID2 = [],[]
    HEUN1,HEUN2 = [],[]
    RK41,RK42 = [],[]
    
    for i in range(len(XE)):
        EULER1.append(EULER[i][0])
        EULER2.append(EULER[i][1])
    for i in range(len(XM)):
        MID1.append(MID[i][0])
        MID2.append(MID[i][1])
    for i in range(len(XH)):
        HEUN1.append(HEUN[i][0])
        HEUN2.append(HEUN[i][1])
    for i in range(len(XR)):
        RK41.append(RK4[i][0])
        RK42.append(RK4[i][1])

        
    plt.plot(XE,EULER1, label='y1')
    plt.plot(XM,MID1)
    plt.plot(XH,HEUN1)
    plt.plot(XR,RK41)
    
    plt.plot(XE,EULER2,label='y2')
    plt.plot(XM,MID2)
    plt.plot(XH,HEUN2)
    plt.plot(XR,RK42)

    plt.title("test_meth_epsilon sur f2 avec epsilon = "+str(eps))
    plt.legend()
    plt.show()
    return

def test_print_field():
    print_field(f1,[1],0,"Tangentes à f1",step_euler)


test_meth_n_step() 
test_meth_epsilon()
test_meth_epsilon2()

test_print_field()

############ TEST LOTKA VOLTERRA ##################

affiche_malthus(np.array([100]),0,150,0.1,step_rk4,0.40)
affiche_verhulst(np.array([100]),0,200,0.1,step_rk4,0.40,400)
affiche_Lotka_Volterra(np.array([100,30]),0,300,0.1,step_rk4,10,0.4,0.1,8)

