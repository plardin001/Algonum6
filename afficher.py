import matplotlib.pyplot as plt
import numpy as np
from method import *

#print a vector field for the cauchy problem : y'(t)= f(y(t),t) ( f : R2 -> R) )
#with y(t0) = y0 as a initial condition
def print_field(f, y0, t0, field_name,method, h=0.01, xmin=0, xmax=10, ymin=0, ymax=10):
    X = np.arange(xmin, xmax, h)
    Y = np.arange(ymin, ymax, h) #maillage
    N = len(X)
    
    plt.title(field_name)
    plt.axis('equal')#repere othonorme
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    #plt.figure()

    F = meth_n_step(y0, t0, N, h, f, method) #calcul de la fonction
    t = t0
    DF = []
    for i in range(N): #calcul de la derivee
        DF.append(f(F[i],t))#calcul de la derivee
        t = t + h
        Y[i]= F[i]

        
    step = N//25
    plt.quiver(X[::step],
               Y[::step],
               F[::step],
               DF[::step])
    plt.show()
    return

def tangents_field(y0, t0, h, f, meth, N):
    #tan0 = lambda x : f(y0, t0).dot((x - t0)) + y0
    t0_ = t0
    X = [0.]*N
    Y = [0.]*N
    DY = [0.]*N
    X[0] = t0
    Y[0] = y0
    DY[0] = f(y0, t0)
    for i in range(1,N):
        y0 = meth(y0, t0, h, f)
        t0 += h
        X[i] = t0
        Y[i] = y0
        DY[i] = f(y0, t0)
    print(t0_,t0)
    U, V = np.meshgrid(np.arange(0.,0.5,0.1), np.arange(0.,0.5,0.1))
    plt.quiver(U, V, Y, DY)
    plt.show()
    return
