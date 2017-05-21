import numpy as np

#######METHOD#######

def step_euler(y, t, h, f):
	return (y + h*f(y, t))

def step_mid_point(y, t, h, f):
	return (y + (h/2.)*(f(y,t)))

def step_heun(y, t, h, f):
	return (y + (h/2.)*(f(y, t) + f(t + h, y + h*f(y,t))))

def step_rk4(y, t, h, f):
	k1 = f(y, t)
	k2 = f(y + (h/2.)*k1, t + (h/2.))
	k3 = f(y + (h/2.)*k2, t + (h/2.))
	k4 = f(y + (h)*k3, t + (h))
	return (y + (h/6.)*(k1 + 2*k2 + 2*k3 + k4))

def meth_n_step(y0, t0, N, h, f, meth):
	Y = np.zeros((N, 1))
	Y[0][0] = y0
	i = 1
	t = t0
	while (i < N):
		Y[0][i] = meth(Y[0][i - 1], t, f)
		t = t + h
		i += 1
	return (Y)

def meth_epsilon(y0,t0,tf,eps,f,meth):
        N = (tf-t0)/eps
        return (meth_n_step(y0,t0,N,eps,f,meth))
