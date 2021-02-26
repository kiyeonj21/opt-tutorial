# ##########################################################
# Solves piecewise linear minimization problem
#      minimize   max_i=1..m (a_i'x + b_i)
#  using subgradient method with decreasing step lengths.
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


# constant step length examples
MAX_ITERS = 50000

# run subgradient method with diminishing step sizes
_,hist1 = sgm_pwl_nonsum_dimin(A,b,x1,0.1,MAX_ITERS)
_,hist2 = sgm_pwl_sqrsum_nonsum(A,b,x1,1,MAX_ITERS)

# run subgradient method with Polyak's optimal step
_,histo = sgm_pwl_optimal_step(A,b,x1,f_min,MAX_ITERS)


# generate plots
# setup plot data
f1, f2, fo = hist1['f'], hist2['f'], histo['f']
fbest1, fbest2, fbesto = hist1['fbest'], hist2['fbest'], histo['fbest']

# plots
plt.figure()
plt.semilogy(fbesto[1:] - f_min, label='optimal')
plt.semilogy(fbest1[1:] - f_min, label='a/sqrt(k) valuelab')
plt.semilogy(fbest2[1:] - f_min, label='a/k value')
plt.legend()
plt.xlabel('k')
plt.ylabel('fbest - fmin')
plt.show()
print()
