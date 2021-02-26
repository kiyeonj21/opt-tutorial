# ##########################################################
# Solves the following regression problem
# min f(x) = ||x||_1
# s.t. ||x-x1||_2<=R
#
# Convex optimization: Algorithms and Complexity, S. Bubeck
# Thm 3.2 implementation, Written by K. Jeon, 02/24/21
# the problem is about Lipschitz function
# ##########################################################
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# generate a problem instance
n = 50
d = 1000
np.random.seed(42)
A = np.random.randn(n, d)
b = np.random.randn(n)
R = 5
L = np.sqrt(d)  # Lipschtz
DELTA = 1e-12
x1 = np.random.rand(d) * 2 - 1  # starting point

# compute optimal value by solving a problem using cvxpy
x = cp.Variable(d)
prob = cp.Problem(objective=cp.Minimize(cp.norm1(x)),
                  constraints=[cp.norm(x - x1) <= R])
prob.solve()
f_opt = prob.value
x_opt = x.value

# run projected subgradient method with decreasing step size
x = x1.copy()
MAX_ITERS = 3000
xs = [x]
for k in range(1, MAX_ITERS):
    g = np.zeros(x.shape[0])
    g[np.where(x > +DELTA)] = +1
    g[np.where(x < -DELTA)] = -1
    x = x - (R / (L * np.sqrt(k))) * g
    if np.linalg.norm(x - x1) > R:
        x = R * (x - x1) / np.linalg.norm(x - x1) + x1
    xs.append(x)

vec = np.array(range(MAX_ITERS)) + 1
xs = np.cumsum(xs, axis=0) / vec[:, None]
bounds = R * L / np.sqrt(vec)

fs = [sum(np.abs(x)) for x in xs]

plt.figure()
plt.semilogy([f - f_opt for f in fs], label='projected subgradient method')
plt.semilogy(bounds, label='bounds')
plt.xlabel('iteration')
plt.ylabel('f_val-f_opt')
plt.title(r'$\min$ $||x||_1$ s.t $||x-x_1||_2\leq R$')
plt.legend()
plt.show()


# ##########################################################
# Solves the following regression problem
# min f(x) = ||x||_1 + alpha/2 * ||x||^2
# s.t. ||x-x1||_2<=R
#
# Convex optimization: Algorithms and Complexity, S. Bubeck
# Thm 3.9 implementation, Written by Kiyeon, 02/24/21
# This problem is about strongly-convex and Lipschitz Function
# ##########################################################

# generate a problem instance
n = 50
d = 1000
alpha = 1.
np.random.seed(42)
A = np.random.randn(n,d)
b = np.random.randn(n)
R = 5
DELTA = 1e-12
x1 = np.random.rand(d)*2-1  # starting point
L = np.sqrt(d) + alpha * (R+np.linalg.norm(x1))     # Lipschtz

# compute optimal value by using cvxpy
x = cp.Variable(d)
prob = cp.Problem(objective=cp.Minimize(cp.norm1(x)+alpha/2 *cp.sum_squares(x)),
                  constraints=[cp.norm(x-x1)<=R])
prob.solve()
f_opt = prob.value
x_opt = x.value


# run projected subgradient algorithm with decreasing step size
x = x1.copy()
MAX_ITERS = 3000
xs =[x]
for k in range(1, MAX_ITERS):
    g = np.zeros(x.shape[0])
    g[np.where(x > +DELTA)] = +1
    g[np.where(x < -DELTA)] = -1
    g = g + alpha*x
    x = x - (2/(alpha*(k+1)))*g
    if np.linalg.norm(x-x1)>R:
        x = R*(x-x1)/np.linalg.norm(x-x1) +x1
    xs.append(x)

vec0 = np.array(range(MAX_ITERS))+1
vec1 = np.array(range(MAX_ITERS))+2
xs = xs*vec0[:,None]
xs = np.cumsum(xs, axis=0)
xs = 2*xs / vec0[:,None] / vec1[:,None]
bounds = 2*L**2 / (alpha*vec1)

fs = [sum(np.abs(x))+alpha/2 * sum(x**2) for x in xs]

plt.figure()
plt.semilogy([f-f_opt for f in fs], label='projected subgradient')
plt.semilogy(bounds,label='bounds')
plt.xlabel('iteration')
plt.ylabel('f_val-f_opt')
plt.title(r'$\min$ $||x||_1 + \frac{\alpha}{2} ||x||^2$ s.t $||x-x_1||_2\leq R$')
plt.legend()
plt.show()


# ##########################################################
# Solve lasso problem
# min f(x) = 0.5*||Ax-b||_2^2
# s.t ||x||_1<=R
# using conditional gradient descent with decreasing step size
#
# Convex optimization: Algorithms and Complexity, S. Bubeck
# Thm 3.8 implementation, Written by K. Jeon, 02/24/21
# this problem is about convex and smoothness function
# ##########################################################

# generate a problem instance
n = 100
d = 1000
np.random.seed(42)
A = np.random.rand(n, d)
b = np.random.rand(n)
R = 1
beta = np.linalg.norm(A, ord=2) ** 2 # smoothness

# compute the optimal value by using cvxpy
x = cp.Variable(d)
prob = cp.Problem(objective=cp.Minimize(0.5 * cp.sum_squares(A @ x - b)),
                  constraints=[cp.norm1(x) <= R])
prob.solve()
f_opt = prob.value
x_opt = x.value

# run conditional gradient (Frank-Wolfe)
x = np.zeros(d)
MAX_ITERS = 3000
xs = [x]
for k in range(1, MAX_ITERS):
    g = A.T.dot(A.dot(x) - b)
    pos = np.argmax(abs(g))
    y = np.zeros(d)
    y[pos] = -np.sign(g[pos]) * R
    gamma = 2 / (k + 1)
    x = (1 - gamma) * x + gamma * y
    xs.append(x)

vec = np.array(range(MAX_ITERS)) + 2
bounds = 2 * beta * (2*R) ** 2 / vec

fs = [0.5 * sum((A.dot(x) - b) ** 2) for x in xs]

plt.figure()
plt.semilogy([f - f_opt for f in fs], label='projected subgradient')
plt.semilogy(bounds, label='bounds')
plt.xlabel('iteration')
plt.ylabel('f_val-f_opt')
plt.title(r'$\min$ $0.5 ||Ax-b||_2^2$ s.t $||x||_1\leq R$')
plt.legend()
plt.show()