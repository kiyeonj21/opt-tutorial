# Subgradient method for the dual problem.
#
# Primal QCQP problem:
#   minimize    (1/2)x'Px - q'x
#   subject to  x^2 <= 1
#
# where P is a positive definite matrix (objective is strictly convex)
#
# Dual problem is:
#   maximize    -(1/2)q'(P + diag(2*lambda))^{-1}q - sum(lambda)
#   subject to  lambda => 0
#
# where lambda are dual variables
#
# EE364b Convex Optimization II, S. Boyd
# Written by Almir Mutapcic, 01/19/07
#

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# generate a problem instance
n = 50  # number of variables n
# generate problem data (P should be positive definite)
# randn('state',1) #set state so problem is reproducable
A = np.random.randn(n, n)
P = A.T.dot(A)
q = np.random.randn(n)
# eigsP = np.linalg.eig(P)
w, v = np.linalg.eig(P)
print(f'Constructed mtx with min eig = {min(w):3.4f} and max eig = {max(w):3.4f}')

# optimal solution of the primal QCQP problem
x = cp.Variable(n)
objective = cp.Minimize((1 / 2) * cp.quad_form(x, P) - q.T @ x)
constraint = [sum(x ** 2) <= 1]
prob = cp.Problem(objective, constraint)
result = prob.solve()
x_opt = x.value
f_min = 1 / 2 * x_opt.T.dot(P.dot(x_opt)) - q.T.dot(x_opt)

print(f'QCQP optimal value is {f_min:0.4f}')

# projected subgradient method applied to the primal problem
fp = [+np.inf]
fpbest = [+np.inf];
print('Starting projected subgradient algorithm for the primal problem...')

# initial point
x = np.zeros(n)

k = 1
MAX_ITERS = 100

while k < MAX_ITERS:
    # subgradient calculation
    g = P.dot(x) - q

    # primal objective values
    fval = (1 / 2) * x.T.dot(P.dot(x)) - q.T.dot(x)
    fp.append(fval)
    fpbest.append(min(fval, fpbest[-1]))

    # step size selection
    alpha = (fval - f_min) / sum(g**2)

    # projected subgradient update
    x = x - alpha * g
    k = k + 1

    # projection onto the feasible set (saturation function)
    x = np.maximum(np.minimum(x, 1), -1)

# subgradient method applied to the dual problem

f = [+np.inf]
fbest = [+np.inf]
g = [-np.inf]
gbest = [-np.inf]
print('Starting the subgradient algorithm applied to the dual problem...')

# initial point
lamb_1 = np.ones(n)
lamb = lamb_1

k = 1

while k < MAX_ITERS:

    # subgradient calculation
    x_star = np.linalg.pinv(P + np.diag(2*lamb)).dot(q)
    h = x_star**2 - 1

    # dual objective values
    gval = -(1/2)*q.dot(x_star) - sum(lamb)
    g.append(gval)
    gbest.append(max(gval, gbest[-1]))

    # primal objective values
    x_star = np.maximum( np.minimum( x_star, 1 ), -1 ) # find nearby feasible point
    fval = (1/2)*x_star.T.dot(P.dot(x_star)) - q.T.dot(x_star)
    f.append(fval)
    fbest.append(min( fval, fbest[-1] ))

    # step size selection
    alpha = 0.1

    # projected subgradient update
    lamb = np.maximum(0, lamb + alpha*h)
    k = k + 1

# plot results
plt.figure()
plt.semilogy([x-y for x,y in zip(fbest,gbest)], label='duality gap')
plt.semilogy([x-f_min for x in fbest],label = 'optim gap dual')
plt.semilogy([x-f_min for x in fpbest], label= 'optim gap primal')
plt.legend()
plt.show()

plt.figure()
plt.plot(g,label='gval label')
plt.plot(f,label='fval label')
plt.xlabel('k')
plt.ylabel('best values')
plt.axis([1, 40, -50, 0])
plt.legend()
plt.show()

plt.figure()
plt.plot(gbest,label='gbest label')
plt.plot(fbest,label='fbest label')
plt.xlabel('k')
plt.ylabel('best values')
plt.axis([1, 40, -50, 0])
plt.legend()
plt.show()
