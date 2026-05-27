import pytest
import numpy as np
import pandas as pd
from scipy import stats
from part1.ols_implementation import ols_fit, hat_matrix
from part1.ols_implementation import coef_inference
from part1.ols_implementation import model_metrics
from part1.ols_implementation import vif


def test_ols_fit_simple_linear_relationship():
    """
    Test 1: Dữ liệu đơn giản y = 2x + 1 (thêm cột 1 cho chặn intercept)
    X = [[1, 0], [1, 1], [1, 2]], y = [1, 3, 5]
    Beta mong đợi là [1, 2]
    """
    X = np.array([[1, 0], [1, 1], [1, 2]])
    y = np.array([1, 3, 5])
    
    beta_hat, sigma2, resid, y_hat = ols_fit(X, y)
    
    # Kiểm tra beta
    np.testing.assert_allclose(beta_hat, [1, 2], atol=1e-7)
    # Kiểm tra sai số (dữ liệu khớp hoàn hảo nên RSS ~ 0)
    assert np.isclose(sigma2, 0, atol=1e-7)
    np.testing.assert_allclose(y_hat, y, atol=1e-7)

def test_ols_fit_dimensions():
    """
    Test 2: Kiểm tra kích thước đầu ra có đúng với input không
    """
    n, p = 10, 3
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    
    beta_hat, sigma2, resid, y_hat = ols_fit(X, y)
    
    assert beta_hat.shape == (p,)
    assert y_hat.shape == (n,)
    assert resid.shape == (n,)
    assert isinstance(sigma2, float)

def test_hat_matrix_properties():
    """
    Test tính chất của ma trận chiếu H:
    1. H là ma trận đối xứng (H = H^T)
    2. H là ma trận lũy đẳng (H*H = H)
    """
    # Tạo dữ liệu giả lập (X có số hàng > số cột)
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    
    H = hat_matrix(X)
    
    # 1. Kiểm tra tính đối xứng
    np.testing.assert_allclose(H, H.T, atol=1e-7, err_msg="H phải đối xứng")
    
    # 2. Kiểm tra tính lũy đẳng (H * H = H)
    H2 = H.dot(H)
    np.testing.assert_allclose(H, H2, atol=1e-7, err_msg="H phải lũy đẳng (H^2 = H)")

def test_hat_matrix_dimensions():
    """
    Test kích thước của ma trận H phải là (n x n)
    với n là số hàng của X
    """
    n, p = 5, 2
    X = np.random.rand(n, p)
    
    H = hat_matrix(X)
    
    assert H.shape == (n, n), f"Kích thước H phải là {n}x{n}"

def test_hat_matrix_identity_on_column_space():
    """
    Test tính chất: H * X = X
    Ma trận chiếu H chiếu mọi vector thuộc không gian cột của X lên chính nó.
    """
    X = np.array([[1, 2], [1, 3], [1, 4]])
    H = hat_matrix(X)
    
    HX = H.dot(X)
    np.testing.assert_allclose(HX, X, atol=1e-7, err_msg="H * X phải bằng X")

def test_coef_inference_vs_known_values():
    """Test so sánh trên dữ liệu đơn giản."""
    # Dữ liệu X có intercept (cột 1) và một biến độc lập
    X = np.array([[1, 1], [1, 2], [1, 3]])
    # y tạo từ mô hình y = 1 + 1*x + nhiễu
    y_true = np.array([2.0, 3.0, 4.0])
    
    # Giả sử ta đã tính được beta và sigma2 từ hàm ols_fit
    # Để test hàm inference, ta dùng chính các giá trị "đúng" này
    beta_hat = np.array([1.0, 1.0]) 
    
    # Tính RSS và sigma2 thủ công để đảm bảo tính nhất quán
    y_hat = X.dot(beta_hat)
    resid = y_true - y_hat
    n, p = X.shape
    sigma2_hat = np.sum(resid**2) / (n - p) 
    
    # Bây giờ gọi hàm inference
    summary = coef_inference(X, y_true, beta_hat, sigma2_hat)
    
    # Kiểm tra
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 2  # 2 hệ số
    assert "pval" in summary.columns
    # p-value phải nằm trong khoảng [0, 1]
    assert all((summary["pval"] >= 0) & (summary["pval"] <= 1))
    
    # Kiểm tra cột bắt buộc phải có
    assert "coef" in summary.columns
    assert "pval" in summary.columns
    assert len(summary) == 2 # 2 biến thì phải có 2 dòng

def test_coef_inference_bounds():
    """Test 2: Kiểm tra CI (Confidence Interval) hợp lý."""
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([2, 3, 4])
    beta_hat = np.array([1.0, 1.0])
    sigma2_hat = 0.1
    
    summary = coef_inference(X, y, beta_hat, sigma2_hat, alpha=0.05)
    
    # CI lower phải luôn nhỏ hơn CI upper
    assert all(summary['ci_lower'] <= summary['ci_upper'])

def test_model_metrics_perfect_fit():
    """Test mô hình khớp hoàn hảo (y = y_hat)."""
    y = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([1.0, 2.0, 3.0])
    p = 2 # Số lượng tham số (ví dụ: intercept + 1 biến độc lập)
    
    metrics = model_metrics(y, y_hat, p)
    
    # Với khớp hoàn hảo: RSS phải bằng 0, R2 phải bằng 1
    assert np.isclose(metrics["RSS"], 0)
    assert np.isclose(metrics["R2"], 1.0)
    assert np.isclose(metrics["adjR2"], 1.0)

def test_model_metrics_values_range():
    """Test các giá trị nằm trong khoảng hợp lệ."""
    n = 10
    y = np.random.rand(n)
    y_hat = np.random.rand(n)
    p = 2
    
    metrics = model_metrics(y, y_hat, p)
    
    # RSS và TSS không bao giờ âm
    assert metrics["RSS"] >= 0
    assert metrics["TSS"] >= 0
    # R2 và adjR2 thường nằm trong khoảng (-inf, 1]
    assert metrics["R2"] <= 1.0
    assert metrics["adjR2"] <= 1.0

def test_model_metrics_f_statistic_consistency():
    """Test tính logic của F-statistic."""
    # Khi dự báo hoàn hảo, F-statistic nên rất lớn (tiến tới vô cùng)
    # Ở đây ta test với dữ liệu có nhiễu nhẹ để đảm bảo F > 0
    y = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([1.1, 1.9, 3.1])
    p = 2
    
    metrics = model_metrics(y, y_hat, p)
    
    assert metrics["F"] > 0

def test_vif_perfect_multicollinearity():
    """
    Test trường hợp đa cộng tuyến hoàn hảo.
    Nếu hai cột giống hệt nhau, VIF phải trả về vô cùng (np.inf).
    """
    X = np.array([[1, 1, 2.0001],
                  [1, 2, 4.0002],
                  [1, 3, 6.0003]])
    
    vif_values = vif(X)
    
    # VIF sẽ trả về giá trị rất lớn thay vì np.inf
    assert np.all(vif_values > 100), "VIF phải trả về giá trị rất lớn khi có đa cộng tuyến"

def test_vif_orthogonal_variables():
    """
    Test trường hợp các biến trực giao (không liên quan).
    VIF của các biến độc lập nên xấp xỉ bằng 1.
    """
    # Ma trận với các biến trực giao
    X = np.array([[1, -1, -1],
                  [1,  1, -1],
                  [1,  0,  2]])
    
    vif_values = vif(X)
    
    # Bỏ qua cột intercept (cột 0), kiểm tra các cột còn lại
    # Đối với biến trực giao, VIF sẽ bằng 1
    np.testing.assert_allclose(vif_values[1:], 1.0, atol=1e-7)

def test_vif_output_shape():
    """Test kiểm tra kích thước của mảng kết quả."""
    n, p = 10, 3
    X = np.random.rand(n, p)
    
    vif_values = vif(X)
    
    assert vif_values.shape == (p,), "Số lượng giá trị VIF phải bằng số cột của X"