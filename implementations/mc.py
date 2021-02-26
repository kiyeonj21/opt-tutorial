# ##########################################################
# Positive semidefinite matrix completion
#   find        X
#   subject to  X(i,j) = val,   where (i,j,val) is the data
#               X is PSD mtx with size n-by-n
# via alternating projections method.
#
# EE364b Convex Optimization II, S. Boyd
# Written by Almir Mutapcic, 01/19/07
# ##########################################################


import numpy as np
import matplotlib.pyplot as plt

# generate a problem instance
n = 50 # matrix size
p = 500 # number of missing entries
# generate a positive definite matrix
# random.seed(1)  # set state so problem is reproducable
A = np.random.randn(n,n)
X = A.T.dot(A)
print(f'Constructed PD mtx with min eig ={np.linalg.norm(X,ord=-2):3.4f}');

# getting a random sparsity pattern
# random.seed(1)
S = np.random.rand(n,n)
# make sure sparsity pattern is symmetric and has ones on the diagonal
S = ((S + S.T) + np.eye(n))>1

# zero out missing elements
X = X * S
I = np.where(X!=0)
V = X[I]
print(f'Initial mtx cmplt. with zeros has min eig = {np.linalg.norm(X,ord=-2):3.4f}')

# plt.figure()
# plt.spy(X)
# plt.show()

# # ********************************************************************
# #  subgradient method computation
# # ********************************************************************
f, fbest, dist = [+np.inf], [+np.inf], []
print('Starting alternating projections ...')

k = 1
MAX_ITERS = 100

while k <= MAX_ITERS:

    # project on PSD cone
    lamb, T = np.linalg.eig(X)
    ind = np.where( lamb < 0 )
    lamb[ind] = 0
    Xproj = T.dot(np.diag(lamb)).dot(T.T)
    dist.append(np.linalg.norm(Xproj - X,'fro'))
    X = Xproj.copy()

    # project on the fixed matrix values
    Xproj[I] = V
    dist.append(np.linalg.norm(Xproj - X,'fro'))
    X = Xproj

    k = k + 2
    if k % 100 == 0:
        print(f'iter: {k}')

# ********************************************************************
#  plot results
# ********************************************************************
plt.figure()
plt.semilogy( dist[1:])
plt.xlabel('k')
plt.ylabel('dist')
plt.show()