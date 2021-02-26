# Solves least l1-norm minimization problem
#     minimize    ||x||_1
#     subject to  Ax = b
# using projected subgradient method
#
# EE364b Convex Optimization II, S. Boyd
# Written by Almir Mutapcic, 01/19/07

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# generate a problem instance
n = 1000    # number of variables
m = 50  # number of equality constraints
# randn('state',1); # set state so problem is reproducable
A = np.random.randn(m,n)
b = np.random.randn(m)

# threshold value below which we consider an element to be zero
DELTA = 1e-8


# compute an optimal point by solving an LP (use CVX)
x = cp.Variable(n)
objective = cp.Minimize(cp.norm1(x))
constraint = [A @ x == b]
prob = cp.Problem(objective,constraint)
result = prob.solve()
f_min = sum(np.abs(x.value))
print(f_min)
print(f'Optimal value is {f_min:0.4f}')

nnz = len(np.where(abs(x.value)>DELTA))
print(f'Found a feasible x in R^{n} that hs {nnz}')

# initial point needs to satisfy A*x1 = b (can use least-norm solution)
x1 = np.linalg.pinv(A).dot(b)

# ********************************************************************
# subgradient method computation
# ********************************************************************
f = [+np.inf]
fbest = [+np.inf]

k = 1
x = x1
MAX_ITERS = 3000;

while k < MAX_ITERS:

    # subgradient calculation
    fval = sum(np.abs(x))
    g = np.zeros(x.shape[0])
    g[np.where(x> DELTA)] = 1
    g[np.where(x < -DELTA)] = -1
    # g = (x > DELTA) - (x < -DELTA) # sign(x) with DELTA tolerance

    # step size selection
    alpha = 0.1/k

    # keep objective values
    f.append(fval)
    fbest.append(min(fval, fbest[-1]))

    # subgradient update
    x = x - alpha * (g - A.T.dot(np.linalg.inv(A.dot(A.T)).dot(A.dot(g))))
    # x = x - alpha*(g - A'*(A'\g))
    k = k + 1

    if (k % 500) == 0:
        print(f'iter: {k}')
#
# ********************************************************************
# plot results
# ********************************************************************
plt.figure()
plt.semilogy(fbest[1:]-f_min)
plt.xlabel('k')
plt.ylabel('fbest- fmin')
plt.show()
# figure(1), clf
# set(gca, 'FontSize',18);
# semilogy( [1:MAX_ITERS], fbest-f_min,'LineWidth',1.5 )
# xlabel('k');
# ylabel('fbest - fmin');
# %print -depsc least_l1_norm_fbest