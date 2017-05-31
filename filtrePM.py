import math as m
import numpy as np
from method import*
import matplotlib.pyplot as plt

def malthus_modele(gamma):
    return lambda N,t: gamma*N[0]

def verhulst_modele(gamma, k):
    return lambda N,t : gamma*N[0]*(1-(N[0]/k))

def affiche_malthus(N0,t0,tf,eps,meth,gamma):
    f = malthus_modele(gamma)
    X,Y=meth_epsilon(N0,t0,tf,eps, f,meth)
    print(f([10], 1))
    print(X,Y)
    plt.plot(X,Y)
    plt.title('Modele de Malthus en prenant gamma=')
    plt.xlabel('Temps')
    plt.ylabel('Population')
    plt.show()


def affiche_verhulst(N0,t0,tf,eps,meth,gamma):
    X,Y=meth_epsilon(N0,t0,tf,eps,verhulst_modele(gamma),meth)
    print(X,Y)
    plt.plot(X,Y)
    plt.title('Modele de Verhulst en prenant gamma=')
    plt.xlabel('Temps')
    plt.ylabel('Population')
    plt.show()

affiche_malthus(np.array([30]),0,100,1,step_rk4,0.40)
