##########################################################
# Solves regression problem
# min f(x) = 0.5*||Ax-b||_2^2
# using gradient descent with constant step size
#
# Convex optimization: Algorithms and Complexity, S. Bubeck
# Thm 3.3 & 3.10 implementation, Written by Kiyeon, 02/24/21
# this problem is about strongly convex and smoothness
##########################################################
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# generate a problem instance
n = 1000
d = 100
np.random.seed(42)
A = np.random.rand(n, d)
b = np.random.rand(n)
beta = np.linalg.norm(A, ord=2) ** 2    # smoothness
alpha = np.linalg.norm(A, ord=-2) ** 2
kappa = beta/ alpha


# compute the optimal value by using cvxpy
x = cp.Variable(d)
prob = cp.Problem(objective=cp.Minimize(0.5 * cp.sum_squares(A @ x - b)))
prob.solve()
f_opt = prob.value
x_opt = x.value

# run gradient descent with fixed step size (only using smoothness)
x = np.random.rand(d)*2-1
MAX_ITERS = 3000
xs = [x]

for k in range(1,MAX_ITERS):
    delta = -A.T.dot(A.dot(x)-b)
    x = x + 1/beta * delta
    xs.append(x)

fs =[0.5*sum((A.dot(x)-b)**2) for x in xs[1:]]
vec = np.array(range(1,MAX_ITERS))
bounds = 2*beta * sum((xs[0] -x_opt )**2) / vec


plt.figure()
plt.semilogy([f - f_opt for f in fs], label='gradient descent')
plt.semilogy(bounds, label='bounds')
plt.xlabel('iteration')
plt.ylabel('f_val-f_opt')
plt.title(r'$\min$ $0.5||Ax-b||_2^2$')
plt.legend()
plt.show()

# run gradient descent with fixed step size (only using strongly convex and smoothness)
x = np.random.rand(d)*2-1
MAX_ITERS = 3000
xs = [x]
for k in range(1,MAX_ITERS):
    delta = -A.T.dot(A.dot(x)-b)
    x = x + 2/(alpha+beta) * delta
    xs.append(x)

fs =[0.5*sum((A.dot(x)-b)**2) for x in xs]
vec = np.array(range(MAX_ITERS+1))
bounds = beta/2 * np.exp(-4*vec/(kappa+1)) * sum((xs[0]-x_opt)**2)


plt.figure()
plt.semilogy([f - f_opt for f in fs], label='gradient descent')
plt.semilogy(bounds, label='bounds')
plt.xlabel('iteration')
plt.ylabel('f_val-f_opt')
plt.title('f is smooth and strongly convex')
plt.legend()
plt.show()


##########################################################
# Solves regression problem
# min f(x) = 0.5*||Ax-b||_2^2
# s.t ||x||_2<=R
# using projected gradient descent with constant step size
#
# Convex optimization: Algorithms and Complexity, S. Bubeck
# Thm 3.10 implementation, Written by Kiyeon, 02/24/21
# this problem is that f is convex and smoothness
##########################################################

# generate a problem instance
n = 1000
d = 100
np.random.seed(42)
A = np.random.rand(n, d)
b = np.random.rand(n)
R = 10
beta = np.linalg.norm(A, ord=2) ** 2    # smoothness
alpha = np.linalg.norm(A, ord=-2) ** 2
kappa = beta/ alpha


# compute the optimal value by using cvxpy
x = cp.Variable(d)
prob = cp.Problem(objective=cp.Minimize(0.5 * cp.sum_squares(A @ x - b)),
                  constraints=[cp.norm2(x)<=R])
prob.solve()
f_opt = prob.value
x_opt = x.value

# run gradient descent with fixed step size
x = np.random.rand(d)*2-1
MAX_ITERS = 3000
xs = [x]

for k in range(1,MAX_ITERS):
    delta = -A.T.dot(A.dot(x)-b)
    x = x + 1/beta * delta
    if np.linalg.norm(x)>R:
        x = R* x /np.linalg.norm(x)
    xs.append(x)

vec = np.array(range(MAX_ITERS))
bounds_x = np.exp(-vec/kappa) * sum((xs[0]-x_opt)**2)

plt.figure()
plt.semilogy([sum((x - x_opt)**2) for x in xs], label='gradient descent')
plt.semilogy(bounds_x, label='bounds')
plt.xlabel('iteration')
plt.ylabel('||x-x_opt||')
plt.title(r'$\min$ $0.5||Ax-b||_2^2$ s.t $||x||_2\leq R$')
plt.legend()
plt.show()

# #########################################################
# Solves regression problem
# min f(x) = 0.5*||Ax-b||_2^2
# using gradient descent with constant step size
#
# Convex optimization: Algorithms and Complexity, S. Bubeck
# Thm 3.3 & 3.10 implementation, Written by Kiyeon, 02/24/21
# this problem is about strongly convex and smoothness
# #########################################################

# generate a problem instance
n = 1000
d = 50
np.random.seed(42)
A = np.random.rand(n, d)
b = np.random.rand(n)
beta = np.linalg.norm(A, ord=2) ** 2    # smoothness
alpha = np.linalg.norm(A, ord=-2) ** 2
kappa = beta/ alpha
# x0 =np.zeros(d)
x0 = np.random.rand(d)*2 -1 # starting point


# compute the optimal value by using cvxpy
x = cp.Variable(d)
prob = cp.Problem(objective=cp.Minimize(0.5 * cp.sum_squares(A @ x - b)))
prob.solve()
f_opt = prob.value
x_opt = x.value

# run gradient descent with fixed step size (only using smoothness)
x = x0.copy()
MAX_ITERS = 200
xs = [x]

for k in range(1,MAX_ITERS):
    delta = -A.T.dot(A.dot(x)-b)
    x = x + 1/beta * delta
    xs.append(x)

fs_1 =[0.5*sum((A.dot(x)-b)**2) for x in xs]

# run gradient descent with fixed step size (only using strongly convex and smoothness)
x = x0.copy()
# MAX_ITERS = 100
xs = [x]
for k in range(1,MAX_ITERS):
    delta = -A.T.dot(A.dot(x)-b)
    x = x + 2/(alpha+beta) * delta
    xs.append(x)

fs_2 =[0.5*sum((A.dot(x)-b)**2) for x in xs]


# run accelerated gradient descent with fixed step size (only using strongly convex and smoothness)
x = x0.copy()
# MAX_ITERS = 100
y = x0.copy()
xs = [y]
for k in range(1,MAX_ITERS):
    delta = -A.T.dot(A.dot(x)-b)
    y_new = x + 1/beta * delta
    gamma = (np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)
    x_new = (1+gamma)*y_new -gamma*y
    xs.append(y_new)
    x = x_new
    y = y_new

fs_3 =[0.5*sum((A.dot(x)-b)**2) for x in xs]

vec = np.array(range(MAX_ITERS))
bounds = (alpha+beta)/2 * sum((x0-x_opt)**2)* np.exp(-vec/np.sqrt(kappa))


plt.figure()
plt.semilogy([f - f_opt for f in fs_1], label='gradient descent with smoothness')
plt.semilogy([f - f_opt for f in fs_2], label='gradient descent with smoothness and strongly convex')
plt.semilogy([f - f_opt for f in fs_3], label='accelerated gradient descent')
plt.semilogy(bounds, label='bounds')
plt.xlabel('iteration')
plt.ylabel('f_val-f_opt')
plt.title(r'$\min$ $0.5||Ax-b||_2^2$')
plt.legend()
plt.show()