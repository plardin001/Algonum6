import math as m
import numpy as np
from method import*
import matplotlib.pyplot as plt

def malthus_modele(gamma):
    return lambda N,t: gamma*N[0]

def verhulst_modele(gamma, k):
    return lambda N,t : gamma*N[0]*(1-(N[0]/k))

def affiche_malthus(N0,t0,nb_points,h,meth,gamma):
    f = malthus_modele(gamma)
    Y=meth_n_step(N0,t0,nb_points,h, f,meth)
    X=[t0+k*h for k in range(nb_points)]
    plt.plot(X,Y)
    plt.title('Modele de Malthus en prenant gamma=0.40')
    plt.xlabel('Temps')
    plt.ylabel('Population')
    plt.show()


def affiche_verhulst(N0,t0,nb_points,h,meth,gamma, k):
    Y=meth_n_step(N0,t0,nb_points,h,verhulst_modele(gamma,k),meth)
    X=[t0+k*h for k in range(nb_points)]
    plt.plot(X,Y)
    plt.title('Modele de Verhulst en prenant gamma=0.40 et k=400')
    plt.xlabel('Temps')
    plt.ylabel('Population')
    plt.show()

def Lotka_Volterra_modele(a,b,c,d):
   return lambda y,t : np.array([y[0]*(a-b*y[1]), y[1]*(c*y[0]-d)])

def affiche_Lotka_Volterra(y0,t0,nb_points,h,meth,a,b,c,d):
    Y=meth_n_step(y0,t0,nb_points,h,Lotka_Volterra_modele(a,b,c,d),meth)
    X=[t0+k*h for k in range(nb_points)]
    Y1=[Y[i][0] for i in range(nb_points)]
    Y2=[Y[i][1] for i in range(nb_points)]
    plt.plot(X, Y1, label="proies")
    plt.plot(X, Y2, label="predateurs")
    plt.title("Methode de Lotka Volterra")
    plt.ylabel("Population")
    plt.xlabel("Temps")
    plt.legend()
    plt.show()

    plt.plot(Y1,Y2)
    plt.title("Proies/Predateurs avec modele Lotka")
    plt.ylabel("Predateurs")
    plt.xlabel("Proies")
    plt.show()



affiche_malthus(np.array([100]),0,150,0.1,step_rk4,0.40)
affiche_verhulst(np.array([100]),0,200,0.1,step_rk4,0.40,400)
affiche_Lotka_Volterra(np.array([100,30]),0,300,0.1,step_rk4,10,0.4,0.1,8)
