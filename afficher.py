import matplotlib.pyplot as plt
import numpy as np



def print_field(Champ,field_name,xmin=0,xmax=10,ymin=0,ymax=10,h=0.1):
    h = .01
    X = np.arange(xmin, xmax, h)
    Y = np.arange(ymin, ymax, h) #maillage

    #XX, YY = np.meshgrid(X, Y)# XX[i, j] == X[j] et YY[i, j] == Y[i]

    plt.title(field_name)
    plt.axis('equal')#repere othonorme
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    step = 100#
    plt.quiver(X[::step],
               Y[::step],
               h,
               Champ[::step])
    plt.show()
