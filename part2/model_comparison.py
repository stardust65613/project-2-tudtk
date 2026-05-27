import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def compute_metrics(y_true, y_pred):
    #temp fix
    y_true = np.atleast_1d(y_true)
    y_pred = np.atleast_1d(y_pred)

    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(rss / n)
    r2 = 1 - (rss / tss)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def add_intercept(X):
    X_arr = np.array(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if not np.all(X_arr[:, 0] == 1):
        return np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
    return X_arr

def ols_fit(X, y):
    X_mat = add_intercept(X)
    y_mat = np.array(y).reshape(-1, 1)
    beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_mat
    return beta

def ols_predict(X, beta):
    X_mat = add_intercept(X)
    return (X_mat @ beta).flatten()

def compute_p_values(X, y, beta):
    X_mat = add_intercept(X)
    y_mat = np.array(y).reshape(-1, 1)
    n, p_plus_1 = X_mat.shape
    
    y_hat = X_mat @ beta
    rss = np.sum((y_mat - y_hat) ** 2)
    sigma_sq = rss / (n - p_plus_1)
    
    vcov = sigma_sq * np.linalg.inv(X_mat.T @ X_mat)
    standard_errors = np.sqrt(np.diagonal(vcov)).reshape(-1, 1)
    
    t_stats = beta / standard_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p_plus_1))
    return p_values.flatten()

def compute_vif(df_X):
    X_mat = np.array(df_X)
    n, p = X_mat.shape
    vif_dict = {}
    
    for i in range(p):
        y_vif = X_mat[:, i]
        X_vif = np.delete(X_mat, i, axis=1)
        X_vif_inc = add_intercept(X_vif)
        
        beta_vif = np.linalg.inv(X_vif_inc.T @ X_vif_inc) @ X_vif_inc.T @ y_vif.reshape(-1, 1)
        y_vif_hat = X_vif_inc @ beta_vif
        
        rss = np.sum((y_vif - y_vif_hat.flatten()) ** 2)
        tss = np.sum((y_vif - np.mean(y_vif)) ** 2)

        r2_j = 1 - (rss / tss)
        #fix temp
        r2_j = np.clip(r2_j, 0, 1) 
        
        # Tính VIF với ngưỡng an toàn
        denominator = 1 - r2_j
        if denominator < 1e-10:
            vif_dict[df_X.columns[i]] = np.inf
        else:
            vif_dict[df_X.columns[i]] = 1 / denominator

        '''
        vif_dict[df_X.columns[i]] = 1 / (1 - r2_j + 1e-10)
        '''
    return vif_dict

def ridge_fit(X, y, lam):
    X_mat = add_intercept(X)
    y_mat = np.array(y).reshape(-1, 1)
    p = X_mat.shape[1]
    I = np.eye(p)
    I[0, 0] = 0  # Không phạt tham số chênh lệch Intercept
    beta = np.linalg.inv(X_mat.T @ X_mat + lam * I) @ X_mat.T @ y_mat
    return beta

def kfold_cv_ridge(X, y, k_folds, lam_list, seed=42):
    np.random.seed(seed)
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    lam_errors = {}
    
    for lam in lam_list:
        fold_mses = []
        for i in range(k_folds):
            test_idx = folds[i]
            train_idx = np.setdiff1d(indices, test_idx)
            
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_val, y_val = X_arr[test_idx], y_arr[test_idx]
            
            beta = ridge_fit(X_tr, y_tr, lam)
            preds = ols_predict(X_val, beta)
            fold_mses.append(np.mean((y_val - preds) ** 2))
        lam_errors[lam] = np.mean(fold_mses)
    return min(lam_errors, key=lam_errors.get), lam_errors

def plot_residual_analysis(X, y, beta):
    """
    Vẽ 4 biểu đồ phần dư tối ưu toán học - KHÔNG TRÀN RAM (Không khởi tạo ma trận H vuông)
    """
    X_mat = add_intercept(X)
    y_val = np.array(y).flatten()
    y_hat = (X_mat @ beta).flatten()
    residuals = y_val - y_hat
    
    n, p = X_mat.shape
    mse = np.sum(residuals**2) / (n - p)
    
    # ----------------==========================================================
    # KIẾN TRÚC TOÁN HỌC TỐI ƯU: TÍNH LEVERAGES THEO HÀNG (TRÁNH BIẾN N^2 MEMORY)
    # ----------------==========================================================
    X_T_X_inv = np.linalg.inv(X_mat.T @ X_mat)
    # Thay vì X @ Inv @ X.T, ta tính tích chập từng hàng để lấy ngay đường chéo chính
    leverages = np.sum((X_mat @ X_T_X_inv) * X_mat, axis=1)
    
    stand_residuals = residuals / (np.sqrt(mse * (1 - leverages)) + 1e-10)
    cooks_d = (residuals**2 / (p * mse)) * (leverages / ((1 - leverages)**2 + 1e-10))
    
    # Kỹ thuật Downsampling đồ thị: Rút ngẫu nhiên 5000 điểm ĐỂ VẼ HÌNH (tránh đơ UI đồ thị khi có 68k điểm đè lên nhau)
    np.random.seed(42)
    plot_idx = np.random.choice(n, size=min(5000, n), replace=False) if n > 5000 else np.arange(n)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Biểu đồ 1: Residuals vs Fitted
    axs[0, 0].scatter(y_hat[plot_idx], residuals[plot_idx], alpha=0.4, color='#3498db', edgecolor='w', s=25)
    axs[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    axs[0, 0].set_title('Residuals vs Fitted (Kiểm tra tính tuyến tính)')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')
    
    # Biểu đồ 2: Normal Q-Q Plot
    sorted_residuals = np.sort(stand_residuals[plot_idx])
    norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(plot_idx)))
    axs[0, 1].scatter(norm_quantiles, sorted_residuals, alpha=0.4, color='#e67e22', edgecolor='w', s=25)
    min_val = min(norm_quantiles.min(), sorted_residuals.min())
    max_val = max(norm_quantiles.max(), sorted_residuals.max())
    axs[0, 1].plot([min_val, max_val], [min_val, max_val], color='r', linestyle='--', linewidth=1.5)
    axs[0, 1].set_title('Normal Q-Q Plot (Kiểm tra phân phối chuẩn)')
    axs[0, 1].set_xlabel('Theoretical Quantiles')
    axs[0, 1].set_ylabel('Standardized Residuals')
    
    # Biểu đồ 3: Scale-Location
    axs[1, 0].scatter(y_hat[plot_idx], np.sqrt(np.abs(stand_residuals[plot_idx])), alpha=0.4, color='#9b59b6', edgecolor='w', s=25)
    axs[1, 0].set_title('Scale-Location (Kiểm tra đồng phương sai)')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel('$\sqrt{|Standardized Residuals|}$')
    
    # Biểu đồ 4: Cook's Distance (Thay plt.stem nặng nề bằng line plot mượt mà)
    axs[1, 1].plot(cooks_d, color='#e74c3c', linewidth=0.8, alpha=0.8)
    axs[1, 1].set_title("Cook's Distance (Phát hiện điểm ảnh hưởng lớn)")
    axs[1, 1].set_xlabel('Observation index')
    axs[1, 1].set_ylabel("Cook's distance")
    
    plt.tight_layout()
    plt.show()