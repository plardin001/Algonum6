import matplotlib.pyplot as plt


def print_field(Champ,field_name,xmin=0,xmax=10,ymin=0,ymax=10,h=0.1):
    h = .01
    X = plt.arange(xmin, xmax, h)
    Y = plt.arange(ymin, ymax, h)#maillage

    XX, YY = np.meshgrid(X, Y)# XX[i, j] == X[j] et YY[i, j] == Y[i]

    plt.title(u'filed_name')
    plt.axis('equal')#repere othonorme
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    step = 50#
    plt.quiver(XX[::step, ::step],
               YY[::step, ::step],
               Champ[0, ::step, ::step],
               Champ[1, ::step, ::step])

