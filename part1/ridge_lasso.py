import numpy as np
from ols_implementation import gaussian_eliminate


def ridge_fit(X, y, lam=1.0):
    p = X.shape[1]
    XtX = X.T.dot(X)
    Xty = X.T.dot(y)

    D = np.eye(p)
    D[0, 0] = 0.0
    A = XtX + lam * D

    A_list = A.tolist()
    if Xty.ndim > 1:
        Xty_list = Xty.flatten().tolist()
    else:
        Xty_list = Xty.tolist()

    beta_list = gaussian_eliminate(A_list, [[float(val)] for val in Xty_list])
    beta_ridge = np.array([row[0] for row in beta_list])
    return beta_ridge


def _soft_threshold(rho, lam):
    if rho > lam:
        return rho - lam
    if rho < -lam:
        return rho + lam
    return 0.0


def lasso_fit(X, y, alpha=1.0, max_iter=1000, tol=1e-6):
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    X_norms = np.sum(X ** 2, axis=0)

    for iteration in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            residual = y - X.dot(beta) + beta[j] * X[:, j]
            rho = np.dot(X[:, j], residual)
            if j == 0:
                beta[j] = rho / X_norms[j]
            else:
                beta[j] = _soft_threshold(rho, alpha) / X_norms[j]
        if np.linalg.norm(beta - beta_old, ord=2) < tol:
            break
    return beta
