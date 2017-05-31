import numpy as np
import method as met
from PIL import Image

A = np.eye(2)
print(A)
A[1][1] = 12
A[0][1] = 42

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
    pic = np.array(im)
    print(pic)
    
    solution_picture = met.meth_n_step(pic, 0, N, step, laplace_function, met.step_euler)
    return (1)

print(laplace_operator(A))
filter_euler("picture.png", 10, 0.1)
