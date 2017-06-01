import numpy as np
import math as m
import method as met
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

#######MATHEMATIC OPERATORS##########
def grad(U):
    N, M = U.shape
    grad = np.empty((N, M), dtype=object)
    vec = np.zeros(2, dtype=np.float)
    for k in range (1, N + 1):
        for l in range (1, M + 1):
            
            grad[k - 1][l - 1] = [0., 0.]
            if (k == N):
                vec[0] = 0.
            else:
                vec[0] = float(U[k + 1 - 1, l - 1]) - float(U[k - 1, l - 1])
            if (l == M):
                vec[1] = 0.
            else:
                vec[1] = float(U[k - 1, l + 1 - 1]) - float(U[k - 1, l - 1])
            grad[k - 1, l - 1][0] = float(vec[0])
            grad[k - 1, l - 1][1] = float(vec[1])
            #if k > 458 :
            #    print("grad k l = ",k,l,grad[k-1][l-1])
    print(grad)
    return (grad)

def div_operator(P):
    N, M = P.shape
    div = np.empty((N, M))
    for k in range (1, N + 1):
        for l in range (1, M + 1):
            if (k == 1):
                div[k - 1][l - 1] += P[k - 1, l - 1][0]
            elif (k == N):
                div[k - 1][l - 1] += -P[k - 1 - 1, l - 1][0]
            else:
                div[k - 1][l - 1] += (P[k - 1, l - 1][0] - P[k - 1 - 1, l - 1][0])
            if (l == 1):
                div[k - 1][l - 1] += P[k - 1, l - 1][1]
            elif (l == M):
                div[k - 1][l - 1] += -P[k - 1, l - 1 - 1][1]
            else:
                div[k - 1][l - 1] += (P[k - 1, l - 1][1] - P[k - 1, l - 1 - 1][1])
    return (div)

def laplace_operator(U):
    return (div_operator(grad(U)))


#########################FILTERS####################

def filter_euler(name_image, N, step):
    laplace_function = lambda U, t: laplace_operator(U)
    im = Image.open(name_image)
    pic = np.array(im, dtype='int64')
    solution_picture = met.meth_n_stepx(pic, 0, N, step, laplace_function, met.step_euler)[-1]
    solution_im = Image.fromarray(solution_picture)
    solution_im.show()
    solution_im.convert('RGB').save("filter_euler.png")
    return (solution_picture)

def kernel_gaussian(sigma, size):
    K = np.zeros((size, size))
    gauss_function = lambda x, y: (1./(m.sqrt(2*m.pi*(sigma**2)))*m.exp(-(1./(2*(sigma**2))* ((x - size//2)** 2) + (y - size//2)**2)))
    for x in range(size):
        for y in range(size):
            K[x][y] = gauss_function(x, y)
    return (K)

def create_f_matrix(pic, sigma):
    K = kernel_gaussian(sigma, 5)
    F = signal.convolve2d(pic, K, 'full', 'symm')
    grad_of_F = grad(F)
    f_matrix = np.empty(grad_of_F.shape, dtype=object)
    n, m = grad_of_F.shape
    print(n,m)
    for i in range(n):
        for j in range(m):
            f_matrix[i][j] = np.exp(-(np.linalg.norm(grad_of_F[i][j]))**2)
    return(f_matrix)

def plot_f(name_image, sigma):
    im = Image.open(name_image)
    pic = np.array(im, dtype='int64')
    f_matrix = create_f_matrix(pic, sigma)
    print(f_matrix)
    solution_im = Image.fromarray(f_matrix)
    solution_im.show()
    #print(f_matrix[300][300])
    #print(f_matrix[1])
    #print(f_matrix[2])
    #print(f_matrix)
"""
    X=[]
    Y=[]
    for i in range(len(f_matrix)):
        for j in range(len(f_matrix[0])):
            X.append(i)
            Y.append(f_matrix[i][j])
    plt.plot(X,Y)
    plt.show()
"""


def filter_perona(name_image, N, step, sigma):
    im = Image.open(name_image)
    pic = np.array(im, dtype='int64')
    f_matrix = create_f_matrix(pic, sigma)
    grad_pic = grad(pic)
    n, m = grad_pic.shape
    matrix_in_div = np.copy(grad_pic)
    for i in range (n):
        for j in range(m):
                for k in range(2):
                    matrix_in_div[i][j][k] = f_matrix[i][j]*grad_pic[i][j][k]
    perona_filter_function = lambda U, t: div_operator(matrix_in_div)
    print(matrix_in_div)
    solution_picture = met.meth_n_stepx(pic, 0, N, step, perona_filter_function, met.step_euler)[-1]
    solution_im = Image.fromarray(solution_picture)
    solution_im.show()
    solution_im.convert('RGB').save("filter_perona.png")
    return (solution_picture)


####################TESTS###########
##print(filter_euler("picture.png", 10, 0.1))
#im = Image.open("picture.png")
#pic = np.array(im, dtype='int64')
##print(create_f_matrix(pic, 1.4))
plot_f("picture.png",1.2)
#print(kernel_gaussian(1.4, 5))
#print(filter_perona("picture.png", 100, 0.2, 1.2))

