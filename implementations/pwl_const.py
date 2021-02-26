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


def sgm_pwl_const_step_length(A, b, x1, R, gamma, TOL, MAX_ITERS):
    # ********************************************************************
    #  subgradient method for piecewise linear minimization
    #  - uses constant step length rule, alpha_k = gamma/norm(subgrad_k);
    #  - keeps track of function and best function values
    #  - also computes the best lower bound (which is a very loose bound)
    #
    #  EE364b Convex Optimization II, S. Boyd
    #  Written by Almir Mutapcic, 01/19/07
    # ********************************************************************
    f, fbest, lbest = [+np.inf], [+np.inf], [-np.inf]
    sum_alpha, sum_alpha_f, sum_alpha_subg = 0, 0, 0
    k = 1
    x = x1

    while k < MAX_ITERS:
        # subgradient calculation
        ind = np.argmax(A.dot(x) + b)
        fval = np.max(A.dot(x) + b)
        g = A[ind, :].T

        # step size selection
        alpha = gamma / np.linalg.norm(g)

        # objective values
        f.append(fval)
        fbest.append(min(fval, fbest[-1]))

        # lower bound computation
        sum_alpha = sum_alpha + alpha
        sum_alpha_f = sum_alpha_f + alpha * fval
        sum_alpha_subg = sum_alpha_subg + alpha ** 2 * sum(g ** 2)

        lval = (2 * sum_alpha_f - R ** 2 - sum_alpha_subg) / (2 * sum_alpha)
        lbest.append(max(lval, lbest[-1]))

        # stopping criteria
        if fbest[-1] - lbest[-1] < TOL:
            break

        # subgradient update
        x = x - alpha * g
        k = k + 1

    # collect history information
    hist = {'f': f, 'fbest': fbest, 'lbest': lbest}
    return x, hist


# constant step length examples
TOL = 1e-3
MAX_ITERS = 3000
gammas = [.05, .01, .005]

# run subgradient method with constant step length for different gammas
_, hist0 = sgm_pwl_const_step_length(A, b, x1, R, gammas[0], TOL, MAX_ITERS)
_, hist1 = sgm_pwl_const_step_length(A, b, x1, R, gammas[1], TOL, MAX_ITERS)
_, hist2 = sgm_pwl_const_step_length(A, b, x1, R, gammas[2], TOL, MAX_ITERS)

# generate plots
# setup plot data
f0, f1, f2 = hist0['f'], hist1['f'], hist2['f']
fbest0, fbest1, fbest2 = hist0['fbest'], hist1['fbest'], hist2['fbest']
lbest0, lbest1, lbest2 = hist0['lbest'], hist1['lbest'], hist2['lbest']

# plots
iter_sm = 100
plt.figure()
plt.semilogy(f0[1:iter_sm] - f_min, label='gamma1val')
plt.semilogy(f1[1:iter_sm] - f_min, label='gamma2val')
plt.semilogy(f2[1:iter_sm] - f_min, label='gamma3val')
plt.axis([1, 100, 1e-1, 2e0])
plt.xlabel('k')
plt.ylabel('f - fmin')
plt.legend()
plt.show()

plt.figure()
plt.semilogy(fbest0[1:] - f_min, label='gamma1val')
plt.semilogy(fbest1[1:] - f_min, label='gamma2val')
plt.semilogy(fbest2[1:] - f_min, label='gamma3val')
plt.axis([1, 3000, 1e-3, 2e0])
plt.legend()
plt.xlabel('k')
plt.ylabel('fbest - fmin')
plt.show()
print()
