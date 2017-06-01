import numpy as np
import math as m
import method as met
from PIL import Image

#######MATHEMATIC OPERATORS##########
def grad(U):
    N, M = U.shape
    grad = np.empty((N, M), dtype=object)
    vec = np.zeros(2, dtype=np.float)
    for k in range (1, N + 1):
        for l in range (1, M + 1):
            grad[k - 1][l - 1] = [0., 0.]
            if (k == N):
                vec[0] = 0
            else:
                vec[0] = U[k + 1 - 1, l - 1] - U[k - 1, l - 1]
            if (l == M):
                vec[1] = 0
            else:
                vec[1] = U[k - 1, l + 1 - 1] - U[k - 1, l - 1]
            grad[k - 1, l - 1][0] = vec[0]
            grad[k - 1, l - 1][1] = vec[1]
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
    solution_picture = met.meth_n_step(pic, 0, N, step, laplace_function, met.step_euler)[-1]
    solution_im = Image.fromarray(solution_picture)
    solution_im.show()
    solution_im.convert('RGB').save("filter_euler.png")
    return (solution_picture)

def kernel_gaussian(sigma, size):
    K = np.zeros((size, size))
    gauss = lambda x, y: (1./(m.sqrt(2*m.pi*(sigma**2)))*m.exp(-(1./(2*(sigma**2))* ((x - size/2)** 2) + (y - size/2)**2))
    print (K)
    return (K)


####################TESTS###########
#print(filter_euler("picture.png", 10, 0.1))
print(kernel_gaussian(1.4, 3))
