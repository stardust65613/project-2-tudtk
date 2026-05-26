import numpy as np
from ols_implementation import ols_fit


def kfold_cv(X, y, k=5, seed=0):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    mses = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        beta_hat, _, _, _ = ols_fit(X[train_idx], y[train_idx])
        y_val = X[val_idx].dot(beta_hat)
        mses.append(np.mean((y[val_idx] - y_val) ** 2))

    return np.mean(mses), mses
