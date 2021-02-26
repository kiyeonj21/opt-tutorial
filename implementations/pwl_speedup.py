# ##########################################################
# Solves piecewise linear minimization problem
#      minimize   max_i=1..m (a_i'x + b_i)
#  using subgradient method with constant step lengths.
# ##########################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

# generate a problem instance
n = 20  # number of variables
m = 100  # number of terms
A = np.random.randn(m, n)
b = np.random.randn(m)

# compute optimal value by solving a linear program (use CVX)
x1 = np.zeros(n)
x = cp.Variable(n)
objective = cp.Minimize(cp.max(A @ x + b))
prob = cp.Problem(objective)
result = prob.solve()
f_min = max(A.dot(x.value) + b)
Rtrue = np.linalg.norm(x1 - x.value)
R = 10

def sgm_pwl_nonsum_dimin(A, b, x1, a, MAX_ITERS):
    # ********************************************************************
    # subgradient method for linear piecewise minimization
    # uses nonsummable diminishing step size rule, alpha_k = a/sqrt(k)
    # ********************************************************************
    f, fbest = [+np.Inf], [+np.Inf]

    k = 1
    x = x1

    while k < MAX_ITERS:
        # subgradient calculation
        ind = np.argmax(A.dot(x) + b)
        fval = np.max(A.dot(x) + b)
        g = A[ind, :].T

        # step size selection
        alpha = a/np.sqrt(k)

        # store objective values
        f.append(fval)
        fbest.append(min(fval, fbest[-1]))

        # subgradient update
        x = x - alpha*g
        k = k + 1

    # collect history information
    hist ={'f':f, 'fbest':fbest}
    return x, hist

def sgm_pwl_sqrsum_nonsum(A,b,x1,a,MAX_ITERS):
    #********************************************************************
    # subgradient method for linear piecewise minimization
    # uses square summable, but nonsummable step size rule, alpha_k = a/k
    #********************************************************************
    f, fbest = [+np.Inf], [+np.Inf]

    k = 1
    x = x1

    while k < MAX_ITERS:

        # subgradient calculation
        ind = np.argmax(A.dot(x) + b)
        fval = np.max(A.dot(x) + b)
        g = A[ind, :].T

        # step size selection
        alpha = a/k

        # objective values
        f.append(fval)
        fbest.append(min(fval, fbest[-1]))

        # subgradient update
        x = x - alpha * g
        k = k + 1

    # collect history information
    hist = {'f': f, 'fbest': fbest}
    return x, hist

def sgm_pwl_cfm(A,b,x1,fmin,MAX_ITERS):
    #********************************************************************
    # subgradient method for linear piecewise minimization
    # uses step size rule as proposed by Camerini, Fratta, and Maffioli
    #********************************************************************
    f, fbest = [+np.Inf], [+np.Inf]

    k = 1
    x = x1
    sprev = np.zeros(x.shape[0])

    while k < MAX_ITERS:

        # subgradient calculation
        ind = np.argmax(A.dot(x) + b)
        fval = np.max(A.dot(x) + b)
        g = A[ind, :].T

        # step size selection
        if sum(sprev**2)==0:
            beta = 0
        else:
            beta = max(0, -(1.5)*(sprev.dot(g))/sum(sprev**2))
        s = g + beta*sprev
        alpha = (fval - fmin)/sum(s**2)

        # objective values
        f.append(fval)
        fbest.append(min(fval, fbest[-1]))

        # subgradient update
        x = x - alpha * s
        k = k + 1
        sprev = s

    # collect history information
    hist = {'f': f, 'fbest': fbest}
    return x, hist

def sgm_pwl_optimal_step(A,b,x1,fmin,MAX_ITERS):
    #********************************************************************
    # subgradient method for linear piecewise minimization
    # uses Polyak's optimal step size based on knowledge of optimal value
    #********************************************************************
    f, fbest = [+np.Inf], [+np.Inf]

    k = 1
    x = x1

    while k < MAX_ITERS:
        # subgradient calculation
        ind = np.argmax(A.dot(x) + b)
        fval = np.max(A.dot(x) + b)
        g = A[ind, :].T

        # step size selection
        alpha = (fval-fmin)/sum(g**2)

        # objective values
        f.append(fval)
        fbest.append(min(fval, fbest[-1]))

        # subgradient update
        x = x - alpha*g
        k = k + 1


    # collect history information
    hist = {'f': f, 'fbest': fbest}
    return x, hist

def sgm_pwl_filt_opt_step(A,b,x1,fmin,beta,MAX_ITERS):
    #********************************************************************
    # subgradient method for linear piecewise minimization
    # uses `filtered' Polyak's optimal step size
    #********************************************************************
    f, fbest = [+np.Inf], [+np.Inf]

    k = 1
    x = x1
    sprev = np.zeros(x.shape[0])

    while k < MAX_ITERS:

        # subgradient calculation
        ind = np.argmax(A.dot(x) + b)
        fval = np.max(A.dot(x) + b)
        g = A[ind, :].T

        # step size selection
        s = (1-beta)*g + beta*sprev
        alpha = (fval - fmin)/sum(s**2)

        # objective values
        f.append(fval)
        fbest.append(min(fval, fbest[-1]))

        # subgradient update
        x = x - alpha*s
        k = k + 1
        sprev = s

    # collect history information
    hist = {'f': f, 'fbest': fbest}
    return x, hist

def sgm_pwl_momentum(A,b,x1,beta,MAX_ITERS):
    #********************************************************************
    # subgradient method for linear piecewise minimization
    # uses `heavy-ball' or `momentum' step size
    #********************************************************************
    f, fbest = [+np.Inf], [+np.Inf]

    k = 1
    x = x1
    xprev = np.zeros(x.shape[0])

    while k < MAX_ITERS:
        # subgradient calculation
        ind = np.argmax(A.dot(x) + b)
        fval = np.max(A.dot(x) + b)
        g = A[ind, :].T

        # step size selection
        alpha = 1/k

        # objective values
        f.append(fval)
        fbest.append(min( fval, fbest[-1]))

        # subgradient update
        xnew = x - (1-beta)*alpha*g + beta*(x - xprev)
        k = k + 1
        xprev = x
        x = xnew

    # collect history information
    hist = {'f': f, 'fbest': fbest}
    return x, hist


# constant step length examples
MAX_ITERS = 100000

# run subgradient method with diminishing step sizes
# _,hist1 = sgm_pwl_nonsum_dimin(A,b,x1,0.1,MAX_ITERS)
# _,hist2 = sgm_pwl_sqrsum_nonsum(A,b,x1,1,MAX_ITERS)
# _,hist4 = sgm_pwl_momentum(A,b,x_1,.75,MAX_ITERS)

# CFM speedup
_,hist3 = sgm_pwl_cfm(A,b,x1,f_min,MAX_ITERS)

# filtered optimal Polyak's step
# _,hist2 = sgm_pwl_filt_opt_step(A,b,x1,f_min,0.275,MAX_ITERS)
_,hist4 = sgm_pwl_filt_opt_step(A,b,x1,f_min,0.25,MAX_ITERS)

# run subgradient method with Polyak's optimal step
_,histo = sgm_pwl_optimal_step(A,b,x1,f_min,MAX_ITERS)


# generate plots
# setup plot data

f3, f4, fo = hist3['f'],hist4['f'], histo['f']
fbest3, fbest4, fbesto = hist3['fbest'],hist4['fbest'], histo['fbest']

# plots
plt.figure()
plt.semilogy(fbesto[1:] - f_min, label='optimal')
plt.semilogy(fbest4[1:] - f_min, label='with filter beta val')
plt.semilogy(fbest3[1:] - f_min, label='CFM')
plt.legend()
plt.xlabel('k')
plt.ylabel('fbest - fmin')
plt.show()
print()
