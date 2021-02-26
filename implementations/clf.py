##########################################################
# Solves logistic regression problem
# min f(theta) = sum log(1+exp(-yi * xi'beta))
# using gradient descent with constant step size
#
# Convex optimization: Algorithms and Complexity, S. Bubeck
# Thm 3.3 implementation, Written by Kiyeon, 02/24/21
# this problem is that f is convex and smoothness
##########################################################

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn import datasets

n = 1000
d = 50
s = 30
np.random.seed(42) # fixed seed number
X, y = datasets.make_classification(n_samples=n, n_features=d, n_informative=50, n_redundant=0)
y = y * 2 - 1
beta = np.linalg.norm(X, 2) ** 2 # smoothness
theta1 = np.random.rand(d) * 2 - 1 # starting point

# using cvxpy
theta = cp.Variable(d)
prob = cp.Problem(objective=cp.Minimize(cp.sum(cp.logistic(cp.multiply(X @ theta, -y)))))
prob.solve()
f_opt = prob.value
theta_opt = theta.value
print(f'optimal function value is {f_opt}')

# gradient descent with smoothness setup
theta = theta1.copy()
MAX_ITERS = 1000
thetas = []


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for k in range(1, MAX_ITERS):
    delta = -X.T.dot(-y * (sigmoid(-X.dot(theta) * y)))
    eta = 1 / beta
    theta = theta + eta * delta
    thetas.append(theta)

fvals = []
for theta in thetas:
    fvals.append(np.sum(np.log(1 + np.exp(-y * X.dot(theta)))))

vec = np.array(range(1,MAX_ITERS))
bounds = 2 * beta * sum((theta1 - theta_opt)**2) / vec

plt.figure()
plt.semilogy([fval - f_opt for fval in fvals], label='gd')
plt.semilogy(bounds, label='bounds')
plt.xlabel('iteration')
plt.ylabel(r'f_val-f_opt')
plt.title(r'$ min f(\theta) = \sum \log(1+\exp(-y_i \cdot x_i^T \beta))$')
plt.legend()
plt.show()

# ##########################################################
# Solves logistic regression problem
# min f(theta) = sum log(1+exp(-yi * xi'beta))
# s.t ||theta-theta1||_2<=R
# using gradient descent with constant step size
#
# Convex optimization: Algorithms and Complexity, S. Bubeck
# Thm 3.3 implementation, Written by Kiyeon, 02/24/21
# this problem is that f is convex and smoothness
# ##########################################################

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn import datasets

n = 1000
d = 50
s = 30
R = 100
np.random.seed(42) # fixed seed number
X, y = datasets.make_classification(n_samples=n, n_features=d, n_informative=50, n_redundant=0)
y = y * 2 - 1
beta = np.linalg.norm(X, 2) ** 2 # smoothness
theta1 = np.random.rand(d) * 2 - 1 # starting point


# using cvxpy
theta = cp.Variable(d)
prob = cp.Problem(objective=cp.Minimize(cp.sum(cp.logistic(cp.multiply(X @ theta, -y)))),
                  constraints=[cp.norm(theta - theta1)<=R])
prob.solve()
f_opt = prob.value
theta_opt = theta.value
print(f'optimal function value is {f_opt}')

# theta1 = theta_opt +  np.random.rand(d) * 2 - 1 # starting point

# gradient descent with smoothness setup
theta = theta1.copy()
MAX_ITERS = 1000
thetas = [theta]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def proj(x, x1, R):
    if np.linalg.norm(x-x1)>R:
        x = R*(x-x1)/np.linalg.norm(x-x1) +x1
    return x


for k in range(1, MAX_ITERS):
    delta = -X.T.dot(-y * (sigmoid(-X.dot(theta) * y)))
    eta = 1 / beta
    theta = theta + eta * delta
    theta = proj(theta,theta1,R)
    thetas.append(theta)

fvals = []
for theta in thetas:
    fvals.append(np.sum(np.log(1 + np.exp(-y * X.dot(theta)))))

vec = np.array(range(MAX_ITERS))+1
bounds = (3 * beta * sum((theta1 - theta_opt)**2) +fvals[0]-f_opt )/ vec

plt.figure()
plt.semilogy([fval - f_opt for fval in fvals], label='gd')
plt.semilogy(bounds, label='bounds')
plt.xlabel('iteration')
plt.ylabel('f_val-f_opt')
plt.title(r'$ min f(\theta) = \sum \log(1+\exp(-y_i \cdot x_i^T \beta))$ s.t $||x-x_1||_2\leq R$')
plt.legend()
plt.show()
