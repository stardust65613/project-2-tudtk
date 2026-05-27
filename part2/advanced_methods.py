import numpy as np

class KernelRidgeRegressionScratch:
    def __init__(self, lam=1.0, length_scale=1.0):
        self.lam = lam
        self.length_scale = length_scale
        self.X_train = None
        self.alpha = None

    def _rbf_kernel(self, X1, X2):
        # Tính toán ma trận khoảng cách Euclidean bình phương
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-sq_dists / (2 * (self.length_scale ** 2)))

    def fit(self, X, y):
        self.X_train = np.array(X)
        y_mat = np.array(y).reshape(-1, 1)
        n = self.X_train.shape[0]
        K = self._rbf_kernel(self.X_train, self.X_train)
        # Thực hiện nghiệm đóng không gian Hilbert: alpha = (K + lambda * I)^(-1) * y
        self.alpha = np.linalg.inv(K + self.lam * np.eye(n)) @ y_mat

    def predict(self, X):
        #temp fix
        X_target = np.array(X)
        if X_target.ndim == 1:
            X_target = X_target.reshape(-1, 1)

        K_star = self._rbf_kernel(X_target, self.X_train)
        return (K_star @ self.alpha).flatten()