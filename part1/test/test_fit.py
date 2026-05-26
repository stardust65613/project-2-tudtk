import pytest
import numpy as np
from part1.ridge_lasso import ridge_fit
from part1.ridge_lasso import lasso_fit 

def standardize(X):
    X_scaled = X.copy().astype(float)
    # Chỉ chuẩn hóa từ cột 1 đến hết, giữ nguyên cột 0 (Intercept)
    for j in range(1, X.shape[1]):
        col = X_scaled[:, j]
        std = np.std(col)
        if std > 1e-9: # Chỉ chuẩn hóa nếu cột không phải là hằng số
            X_scaled[:, j] = (col - np.mean(col)) / std
    return X_scaled

def test_ridge_fit_penalty_effect():
    """
    Test 1: Hiệu ứng của tham số lambda.
    Khi lambda tăng, các hệ số (không phải intercept) phải có xu hướng nhỏ đi.
    """
    X = np.array([[1, 2], [1, 3], [1, 4]])
    y = np.array([2, 3, 5])
    
    beta_small_lam = ridge_fit(X, y, lam=0.01)
    beta_large_lam = ridge_fit(X, y, lam=100.0)
    
    # Hệ số thứ hai (index 1) thường bị ảnh hưởng nhiều bởi lambda
    # so với lambda rất nhỏ (gần như OLS)
    assert np.abs(beta_large_lam[1]) < np.abs(beta_small_lam[1]), \
        "Hệ số beta phải nhỏ hơn khi lambda tăng"

def test_ridge_fit_intercept_invariant():
    """
    Test 2: Intercept (index 0) không được phạt bởi lambda.
    D[0,0] = 0.0 đảm bảo điều này.
    """
    X = np.array([[1, 2], [1, 3], [1, 4]])
    y = np.array([2, 3, 5])
    
    # Với lambda khác nhau, intercept thường ít thay đổi hoặc ổn định hơn
    beta1 = ridge_fit(standardize(X), y, lam=0.1)
    beta2 = ridge_fit(standardize(X), y, lam=1.0)
    
    # Intercept không nên bị 'thu nhỏ' mạnh như các hệ số biến độc lập
    assert np.isclose(beta1[0], beta2[0], atol=0.5)

def test_ridge_fit_singular_matrix():
    """
    Test 3: Ridge xử lý được trường hợp ma trận X không khả nghịch.
    OLS sẽ fail ở đây nhưng Ridge vẫn chạy được.
    """
    # Cột 1 và 2 giống nhau (đa cộng tuyến hoàn hảo) -> XtX suy biến
    X = np.array([[1, 1], [1, 1], [1, 1]])
    y = np.array([1, 1, 1])
    
    # Ridge vẫn phải trả về kết quả nhờ có cộng thêm lam * I
    try:
        beta = ridge_fit(X, y, lam=1.0)
        assert beta.shape == (2,)
    except Exception as e:
        pytest.fail(f"Ridge nên xử lý được ma trận suy biến, nhưng bị lỗi: {e}")

def test_lasso_sparsity():
    """
    Test 1: Đặc tính của Lasso là đưa các hệ số không quan trọng về đúng bằng 0.
    """
    # Tạo dữ liệu: y = 1*X1 + 0*X2 (X2 là nhiễu, không liên quan đến y)
    X = np.array([[1, 0], [1, 1], [1, 2]])
    y = np.array([1, 1, 1]) # Y không phụ thuộc vào X2
    
    # Với alpha đủ lớn, beta[1] phải bằng 0
    beta = lasso_fit(X, y, alpha=0.5)
    
    assert beta[1] == 0, "Hệ số beta của biến không liên quan phải bằng 0 (sparsity)"

def test_lasso_intercept_not_penalized():
    """
    Test 2: Intercept (beta[0]) không được áp dụng soft threshold.
    """
    X = np.array([[1, 2], [1, 3]])
    y = np.array([2, 3])
    
    # Chạy lasso với alpha rất lớn
    beta = lasso_fit(X, y, alpha=10.0)
    
    assert beta[1] == 0
    assert beta[0] != 0, "Intercept không được phép bị triệt tiêu về 0"

def test_lasso_convergence():
    """
    Test 3: Thuật toán phải hội tụ trong phạm vi số lần lặp cho phép.
    """
    n, p = 50, 5
    X = np.random.randn(n, p)
    y = X.dot(np.array([1, 0, 0.5, 0, -1])) + np.random.normal(0, 0.1, n)
    
    beta = lasso_fit(X, y, alpha=0.1, max_iter=2000)
    
    # Kiểm tra xem beta có trả về đúng số lượng phần tử không
    assert len(beta) == p
    assert not np.isnan(beta).any(), "Kết quả beta không được chứa NaN"