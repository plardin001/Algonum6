import numpy as np

#######METHOD#######

def step_euler(y, t, h, f):
	return (y + h*f(y, t))

def step_mid_point(y, t, h, f):
	return (y + h * f(y + h * f(y,t)/2, t + h/2))

def step_heun(y, t, h, f):
	return (y + (h/2.0)*(f(y, t) + f(y + h*f(y,t),t + h)))

def step_rk4(y, t, h, f):
	k1 = f(y, t)
	k2 = f(y + (h/2.)*k1, t + (h/2.))
	k3 = f(y + (h/2.)*k2, t + (h/2.))
	k4 = f(y + (h)*k3, t + h)
	return (y + (h/6.)*(k1 + 2*k2 + 2*k3 + k4))

def meth_n_step(y0, t0, N, h, f, meth):
        Y = np.zeros((N, len(y0)))
        Y[0] = y0
        i = 1
        t = t0
        while (i < N):
                Y[i] = meth(Y[i - 1], t, h, f)
                t = t + h
                i += 1
        return (Y)

def meth_epsilon(y0,t0,tf,eps,f,meth):
	N = 10
	h = (tf - t0) / N
	YN = meth_n_step(y0, t0, N, h, f, meth)
	YNN = meth_n_step(y0, t0, N//2, 2*h, f, meth)
	while((np.linalg.norm(YN[::2] - YNN) > eps) and (N < 10**6)):
		YNN=YN
		N = 2*N
		h = (tf - t0) / N
		YN = meth_n_step(y0, t0, N, h, f, meth)
	x=t0
	XN = np.zeros((N, 1))
	for i in range(N):
                XN[i]=x
                x += h
	return (XN,YN)
