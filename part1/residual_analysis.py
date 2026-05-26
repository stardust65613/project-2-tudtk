import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ols_implementation import hat_matrix


def residual_plots(X, y, beta_hat, plot=True):
    n, p = X.shape
    y_hat = X.dot(beta_hat)
    resid = y - y_hat

    H = hat_matrix(X)
    h = np.clip(np.diag(H), 1e-12, 1 - 1e-12)

    sigma2 = np.sum(resid ** 2) / float(n - p)
    std_resid = resid / np.sqrt(sigma2 * (1 - h))

    cooks = (std_resid ** 2 / p) * (h / (1 - h))

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].scatter(y_hat, resid, alpha=0.7)
        axes[0, 0].axhline(0, color="red", linestyle="--")
        axes[0, 0].set_xlabel("Fitted values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Fitted")

        stats.probplot(std_resid, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Normal Q-Q")

        axes[1, 0].scatter(y_hat, np.sqrt(np.abs(std_resid)), alpha=0.7)
        axes[1, 0].set_xlabel("Fitted values")
        axes[1, 0].set_ylabel("Sqrt{|Standardized Residuals|}")
        axes[1, 0].set_title("Scale-Location")

        axes[1, 1].stem(np.arange(n), cooks, markerfmt=',')
        axes[1, 1].axhline(4 / n, color="red", linestyle="dashed")
        axes[1, 1].set_xlabel("Observation Index")
        axes[1, 1].set_ylabel("Cook's Distance")
        axes[1, 1].set_title("Cook's Distance")

        plt.tight_layout()
        plt.show()

    return {"y_hat": y_hat, "resid": resid, "std_resid": std_resid, "cooks_distance": cooks}
