import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# ========== 1. Create Data ==========
np.random.seed(0)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.vstack((X0, X1))  # shape (2N, 2)
y = np.hstack((np.ones(N), -np.ones(N)))  # shape (2N,)

# ========== 2. Form the QP Dual ==========
K = X @ X.T  # shape (2N, 2N)

# Build the QP matrices
Q = matrix((y[:, None] * y[None, :]) * K)  # shape (2N, 2N)
p = matrix(-np.ones((2 * N, 1)))  # shape (2N, 1)
G = matrix(-np.eye(2 * N))        # α_i ≥ 0 → -α ≤ 0
h = matrix(np.zeros((2 * N, 1)))  # shape (2N, 1)
A = matrix(y.reshape(1, -1))      # equality constraint: yᵀα = 0
b = matrix(np.zeros((1, 1)))

solvers.options['show_progress'] = False
sol = solvers.qp(Q, p, G, h, A, b)
alphas = np.array(sol['x']).flatten()  # shape (2N,)

# ========== 4. Compute Primal Parameters ==========
w = np.sum((alphas * y)[:, None] * X, axis=0) 
S = np.where(alphas > 1e-8)[0]  
b = np.mean(y[S] - X[S] @ w)

print("Number of support vectors:", len(S))
print("w:", w)
print("b:", b)


