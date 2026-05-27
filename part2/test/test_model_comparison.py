import pytest
import numpy as np
from part2.model_comparison import *

def test_compute_metrics_perfect_prediction():
    """Kiểm tra trường hợp dự đoán hoàn hảo (sai số bằng 0)."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0

def test_compute_metrics_constant_prediction():
    """Kiểm tra khi dự đoán sai lệch một khoảng cố định."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    
    metrics = compute_metrics(y_true, y_pred)
    
    # MAE = |1-2| + |2-3| + |3-4| / 3 = 1.0
    assert metrics["MAE"] == 1.0
    # RMSE = sqrt(((1-2)^2 + (2-3)^2 + (3-4)^2) / 3) = sqrt(1) = 1.0
    assert metrics["RMSE"] == 1.0

def test_compute_metrics_r2_zero():
    """
    Kiểm tra R2 khi mô hình dự đoán bằng giá trị trung bình.
    Khi y_pred = mean(y_true), R2 sẽ bằng 0.
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0]) # Trung bình là 2.0
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert metrics["R2"] == 0.0

def test_compute_metrics_input_types():
    """Kiểm tra xem hàm có hoạt động với list thay vì numpy array không."""
    y_true = [1, 2]
    y_pred = [1, 2]
    
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0

def test_add_intercept_matrix_2d():
    """Kiểm tra việc thêm cột intercept vào ma trận 2x2."""
    X = np.array([[10, 20], [30, 40]])
    X_new = add_intercept(X)
    
    # Kiểm tra kích thước: từ 2x2 lên 2x3
    assert X_new.shape == (2, 3)
    # Kiểm tra cột đầu tiên có toàn là số 1 không
    assert np.all(X_new[:, 0] == 1)
    # Kiểm tra các cột còn lại vẫn giữ nguyên dữ liệu gốc
    np.testing.assert_array_equal(X_new[:, 1:], X)

def test_add_intercept_array_1d():
    """Kiểm tra việc chuyển đổi mảng 1 chiều thành 2 chiều và thêm intercept."""
    X = np.array([5, 10, 15])
    X_new = add_intercept(X)
    
    # Kiểm tra reshape: từ (3,) thành (3, 2)
    assert X_new.shape == (3, 2)
    # Kiểm tra cột đầu tiên là 1
    assert np.all(X_new[:, 0] == 1)
    # Kiểm tra cột thứ hai là dữ liệu gốc
    np.testing.assert_array_equal(X_new[:, 1].flatten(), X)

def test_ols_fit_simple_linear():
    """
    Test hồi quy đơn giản: y = 2x + 1
    X = [0, 1, 2], y = [1, 3, 5]
    Hệ số chặn (beta0) phải là 1, hệ số (beta1) phải là 2.
    """
    X = np.array([[0], [1], [2]])
    y = np.array([1, 3, 5])
    
    beta = ols_fit(X, y)
    
    # Kết quả kỳ vọng: beta = [[1], [2]]
    # beta[0] là intercept (1), beta[1] là slope (2)
    assert beta.shape == (2, 1)
    np.testing.assert_allclose(beta.flatten(), [1.0, 2.0], atol=1e-10)

def test_ols_fit_handles_perfect_fit():
    """
    Test dữ liệu khớp hoàn hảo.
    """
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6]) # y = 2x + 0
    
    beta = ols_fit(X, y)
    
    # Intercept phải gần bằng 0, Slope phải bằng 2
    assert beta[0] == pytest.approx(0.0, abs=1e-10)
    assert beta[1] == pytest.approx(2.0, abs=1e-10)

def test_ols_predict_simple():
    # Giả sử ta đã có beta cho phương trình y = 1 + 2x
    # beta[0] = 1 (intercept), beta[1] = 2 (slope)
    beta = np.array([[1.0], [2.0]]) 
    X_new = np.array([[1.0], [2.0], [3.0]])
    
    y_pred = ols_predict(X_new, beta)
    
    # Kết quả mong đợi:
    # y1 = 1 + 2*1 = 3
    # y2 = 1 + 2*2 = 5
    # y3 = 1 + 2*3 = 7
    expected = np.array([3.0, 5.0, 7.0])
    
    np.testing.assert_allclose(y_pred, expected, atol=1e-10)

def test_ols_predict_shape():
    beta = np.array([[0.5], [1.5]])
    X_new = np.array([[1], [2]])
    y_pred = ols_predict(X_new, beta)
    
    # Kết quả phải là 1D array có 2 phần tử
    assert y_pred.ndim == 1
    assert y_pred.shape == (2,)

def test_ridge_fit_penalty_effect():
    # X có 2 cột (cột 0 là intercept, cột 1 là feature)
    X = np.array([[1, 2], [1, 3], [1, 4]])
    y = np.array([2, 3, 5])
    
    beta_small_lam = ridge_fit(X, y, lam=0.01)
    beta_large_lam = ridge_fit(X, y, lam=100.0)
    
    # 1. Kiểm tra shape là (2, 1) vì hàm add_intercept thấy đã có intercept rồi
    assert beta_small_lam.shape == (2, 1), f"Kỳ vọng shape (2, 1), nhận được {beta_small_lam.shape}"
    
    # 2. Kiểm tra penalty effect: 
    # beta[0] là intercept (không bị phạt), beta[1] là feature (bị phạt)
    # Hệ số feature (index 1) phải giảm khi lambda tăng
    assert np.abs(beta_large_lam[1]) < np.abs(beta_small_lam[1]), "Hệ số beta feature không giảm khi lambda tăng"

def test_ridge_fit_intercept_invariant():
    X = np.array([[1, 2], [1, 3], [1, 4]])
    y = np.array([2, 3, 5])
    
    # 1. Chạy hàm thực tế (đã không phạt intercept)
    beta2 = ridge_fit(X, y, lam=1.0)
    
    # 2. Tính expected_beta2 theo đúng logic
    p = X.shape[1]
    penalty = 1.0 * np.eye(p)
    penalty[0, 0] = 0 
    
    y_col = y.reshape(-1, 1)
    X_mat = add_intercept(X) 
    expected_beta2 = np.linalg.inv(X_mat.T @ X_mat + penalty) @ X_mat.T @ y_col
    
    # 3. So sánh
    np.testing.assert_allclose(beta2, expected_beta2, atol=1e-5)

def test_ridge_fit_singular_matrix():
    """
    Test ridge xử lý được ma trận suy biến.
    """
    # X có 2 cột giống hệt nhau, XtX sẽ suy biến nếu không có lambda
    X = np.array([[1, 2], [1, 2], [1, 2]])
    y = np.array([1, 2, 3])
    
    try:
        beta = ridge_fit(X, y, lam=1.0)
        
        assert beta.shape == (2, 1) or beta.shape == (2,), \
            f"Kỳ vọng shape (2,1), nhận được {beta.shape}"
            
    except Exception as e:
        pytest.fail(f"Ridge nên xử lý được ma trận suy biến bằng penalty, nhưng bị lỗi: {e}")

def test_compute_vif_perfect_independence():
    """
    Test với dữ liệu hoàn toàn độc lập (ma trận đơn vị).
    VIF kỳ vọng phải gần bằng 1.0 vì R2_j = 0.
    """
    df = pd.DataFrame({
        'A': [1.0, 0.0, 0.0],
        'B': [0.0, 1.0, 0.0],
        'C': [0.0, 0.0, 1.0]
    })
    
    vifs = compute_vif(df)
    
    # Kết quả trả về phải là dict và các giá trị VIF gần bằng 1
    assert isinstance(vifs, dict)
    for col in df.columns:
        assert vifs[col] == pytest.approx(1.0, abs=1e-5)

def test_compute_vif_perfect_independence():
    # Tăng số dòng lên để mô hình hồi quy ổn định hơn
    df = pd.DataFrame({
        'A': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'B': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'C': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    vifs = compute_vif(df)
    
    assert isinstance(vifs, dict)
    for col in df.columns:
        # Với 10 dòng, VIF sẽ tiến rất gần về 1.0
        assert vifs[col] == pytest.approx(1.0, abs=0.1), f"VIF của {col} không đạt chuẩn!"

def test_compute_p_values_shape():
    """Kiểm tra số lượng p-value trả về phải bằng số lượng hệ số beta."""
    X = np.array([[1, 2], [2, 3], [5, 1], [3, 4]]) 
    y = np.array([1, 2, 3, 4])
    
    beta = np.array([[0.1], [0.5], [0.2]]) 
    
    p_values = compute_p_values(X, y, beta)
    
    assert len(p_values) == len(beta), f"Kỳ vọng {len(beta)} p-values, nhận được {len(p_values)}"

def test_compute_p_values_range():
    """Kiểm tra p-value luôn nằm trong khoảng từ 0 đến 1."""
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]]) 
    y = np.array([1, 2, 3, 4])
    beta = np.array([[0.1], [0.5], [0.2]])
    
    p_values = compute_p_values(X, y, beta)
    
    # Kiểm tra p-value trong khoảng [0, 1]
    assert np.all((p_values >= 0) & (p_values <= 1)), "P-value phải nằm trong khoảng từ 0 đến 1"
    
def test_kfold_cv_ridge_returns_correct_types():
    """Kiểm tra cấu trúc đầu ra của hàm."""
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    lam_list = [0.0, 0.1, 1.0]
    
    best_lam, lam_errors = kfold_cv_ridge(X, y, k_folds=3, lam_list=lam_list)
    
    # Kiểm tra kiểu dữ liệu
    assert isinstance(lam_errors, dict)
    assert len(lam_errors) == len(lam_list)
    # Kiểm tra best_lam phải nằm trong danh sách lam_list
    assert best_lam in lam_list

def test_kfold_cv_ridge_consistent_errors():
    """Kiểm tra lỗi MSE không được âm và giá trị trả về hợp lệ."""
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([2, 3, 4, 5, 6])
    lam_list = [0.1]
    
    best_lam, lam_errors = kfold_cv_ridge(X, y, k_folds=2, lam_list=lam_list)
    
    # MSE phải >= 0
    assert lam_errors[0.1] >= 0
    # best_lam phải là 0.1 vì chỉ có 1 lựa chọn
    assert best_lam == 0.1