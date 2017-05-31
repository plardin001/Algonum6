import matplotlib.pyplot as plt
import numpy as np
from method import *

#print a vector field for the cauchy problem : y'(t)= f(y(t),t) ( f : R2 -> R) )
#with y(t0) = y0 as a initial condition
def hs_print_field(f, y0, t0, field_name,method, h=0.01, xmin=0, xmax=10, ymin=0, ymax=10):
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
"""

def print_field(f, y0, t0, field_name,method, h=0.01, xmin=0, xmax=10, ymin=0, ymax=10):
    X = np.arange(xmin, xmax, h)
    Y = np.arange(ymin, ymax, h) #maillage
    N = len(X)
    dX,dY=VdPol([X1 ,Y1],0) # generer  les  vecteurs  tangents
    M=hypot(dX,dY)# normalisation
    M[M==0]=1.
    dX /= M # remplaces  par 1 avant  division
    dY /= M
    quiver(Y1 ,X1 ,dY ,dX,M)
    # generation du  champs  de vecteurs
    show ()
"""
