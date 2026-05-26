import numpy as np
import pandas as pd
from scipy import stats


def generate_synthetic_data(n=200, p=3, beta=None, sigma=1.0, seed=42, collinear=False):
    rng = np.random.RandomState(seed)
    if beta is None:
        beta = np.arange(1, p + 1, dtype=float)
    beta = np.asarray(beta, dtype=float).reshape(-1, 1)
    X_raw = rng.normal(size=(n, p - 1))
    if collinear and p - 1 >= 2:
        X_raw[:, 1] = X_raw[:, 0] * 0.95 + rng.normal(scale=0.1, size=n)
    X = np.hstack([np.ones((n, 1)), X_raw])
    eps = rng.normal(scale=sigma, size=(n, 1))
    y = X.dot(beta) + eps
    return X, y.ravel(), beta.ravel()


def gaussian_eliminate(A, b):
    """Solve a square linear system using Gauss-Jordan elimination with partial pivoting."""
    n = len(A)
    epsilon = 1e-9
    is_matrix = isinstance(b[0], list)
    b_cols = len(b[0]) if is_matrix else 1

    M = []
    for i in range(n):
        row = [float(val) for val in A[i]]
        if is_matrix:
            row.extend([float(val) for val in b[i]])
        else:
            row.append(float(b[i]))
        M.append(row)

    for k in range(n):
        max_row = k
        for r in range(k + 1, n):
            if abs(M[r][k]) > abs(M[max_row][k]):
                max_row = r

        if abs(M[max_row][k]) < epsilon:
            raise ValueError("Vo nghiem")

        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]

        pivot = M[k][k]
        for c in range(k, n + b_cols):
            M[k][c] /= pivot

        for r in range(n):
            if r != k:
                factor = M[r][k]
                for c in range(k, n + b_cols):
                    M[r][c] -= factor * M[k][c]

    if is_matrix:
        return [M[r][n:] for r in range(n)]
    return [M[r][n] for r in range(n)]


def inverse(A):
    n = len(A)
    if n != len(A[0]):
        raise ValueError("Ma tran khong vuong")
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return gaussian_eliminate(A, I)


def ols_fit(X, y):
    n, p = X.shape
    XtX = X.T.dot(X)
    Xty = X.T.dot(y)

    XtX_list = XtX.tolist()
    if Xty.ndim > 1:
        Xty_list = Xty.flatten().tolist()
    else:
        Xty_list = Xty.tolist()

    beta_list = gaussian_eliminate(XtX_list, Xty_list)
    beta_hat = np.array(beta_list)
    y_hat = X.dot(beta_hat)
    resid = y - y_hat
    RSS = resid.T.dot(resid)
    sigma2_hat = RSS / float(n - p)

    return beta_hat, sigma2_hat, resid, y_hat


def hat_matrix(X):
    XtX = X.T.dot(X)
    XtX_list = XtX.tolist()
    inv_XtX_list = inverse(XtX_list)
    inv_XtX = np.array(inv_XtX_list)
    H = X.dot(inv_XtX).dot(X.T)
    return H


def model_metrics(y, y_hat, p):
    n = len(y)
    p = p - 1
    resid = y - y_hat
    RSS = np.sum(resid ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)

    R2 = 1 - (RSS / TSS)
    adjR2 = 1 - (RSS / (n - p)) / (TSS / (n - 1))
    F = ((TSS - RSS) / p) / (RSS / (n - p))
    return {
        "RSS": RSS,
        "TSS": TSS,
        "R2": R2,
        "adjR2": adjR2,
        "F": F,
    }


def coef_inference(X, y, beta_hat, sigma2_hat, alpha=0.05):
    n, p = X.shape
    XtX = X.T.dot(X)
    XtX_list = XtX.tolist()
    inv_XtX_list = inverse(XtX_list)
    inv_XtX = np.array(inv_XtX_list)

    cov_beta = sigma2_hat * inv_XtX
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta_hat / se
    df = n - p
    pvals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ci_lower = beta_hat - t_crit * se
    ci_upper = beta_hat + t_crit * se
    summary = pd.DataFrame({
        "coef": beta_hat,
        "se": se,
        "t": t_stats,
        "pval": pvals,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    })
    return summary

def vif(X):
    p = X.shape[1]
    vifs = np.zeros(p)
    for j in range(p):
        mask = np.arange(p) != j
        X_j = X[:, j]
        X_others = X[:, mask]

        XtX = X_others.T.dot(X_others)
        Xty = X_others.T.dot(X_j)
        
        XtX_list = XtX.tolist()
        Xty_list = Xty.flatten().tolist() if Xty.ndim > 1 else Xty.tolist()
        try:
            beta_list = gaussian_eliminate(XtX_list, Xty_list)
        except ValueError:
            vifs[j] = np.inf # Đa cộng tuyến hoàn hảo -> VIF là vô cùng
            continue
        
        beta_j = np.array(beta_list)
        pred = X_others.dot(beta_j)
        RSS = np.sum((X_j - pred) ** 2)
        if np.all(X_j == X_j[0]): 
            # Dùng Uncentered TSS
            TSS = np.sum(X_j ** 2)
        else:
            # Nếu là cột biến số bình thường, dùng Centered TSS
            TSS = np.sum((X_j - np.mean(X_j)) ** 2)
        #TSS = np.sum((X_j - np.mean(X_j)) ** 2)
        R2j = 1 - RSS / TSS if TSS > 0 else 0.0
        vifs[j] = 1.0 / (1.0 - R2j) if (1.0 - R2j) > 1e-12 else np.inf
    return vifs