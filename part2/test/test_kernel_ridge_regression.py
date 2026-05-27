import pytest
import numpy as np
from part2.advanced_methods import KernelRidgeRegressionScratch

@pytest.fixture
def krr_model():
    return KernelRidgeRegressionScratch(lam=1.0, length_scale=1.0)

def test_rbf_kernel_shape(krr_model):
    X1 = np.array([[1, 2], [3, 4], [5, 6]])
    X2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    K = krr_model._rbf_kernel(X1, X2)
    assert K.shape == (3, 4)

def test_rbf_kernel_diagonal(krr_model):
    X = np.array([[1, 2], [3, 4]])
    K = krr_model._rbf_kernel(X, X)
    # Dùng np.testing để so sánh mảng số thực
    np.testing.assert_array_almost_equal(np.diag(K), [1.0, 1.0])

def test_rbf_kernel_symmetry(krr_model):
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8], [9, 10]])
    K1 = krr_model._rbf_kernel(X1, X2)
    K2 = krr_model._rbf_kernel(X2, X1)
    np.testing.assert_array_almost_equal(K1, K2.T)

def test_fit_perfect_fit_with_zero_lambda(krr_model):
    # Cấu hình lại lambda = 0 để mô hình khớp hoàn hảo
    krr_model.lam = 0.0
    
    # Dữ liệu huấn luyện phức tạp hơn một chút (ví dụ: hàm bậc hai)
    X = np.array([[1], [2], [3]])
    y = np.array([1, 4, 9])
    
    krr_model.fit(X, y)
    
    # Dự đoán lại chính tập train
    y_pred = krr_model.predict(X)
    
    # Với lambda = 0, mô hình phải khớp gần như hoàn hảo (atol rất nhỏ)
    np.testing.assert_allclose(y_pred, y, atol=1e-5)

def test_fit_calculates_alpha_correctly(krr_model):
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    krr_model.fit(X, y)
    
    # Kiểm tra xem alpha đã được khởi tạo đúng kích thước (3 mẫu, 1 cột)
    assert krr_model.alpha is not None
    assert krr_model.alpha.shape == (3, 1)

def test_predict_returns_correct_shape(krr_model):
    # Setup: Huấn luyện trước
    krr_model.fit(np.array([[1], [2]]), np.array([1, 2]))
    
    # Test: Dự đoán trên 3 mẫu dữ liệu mới
    X_new = np.array([[1.5], [2.5], [3.5]])
    y_pred = krr_model.predict(X_new)
    
    assert y_pred.shape == (3,), "Kết quả dự đoán phải có cùng số dòng với dữ liệu đầu vào"
    
def test_predict_accuracy_on_simple_data(krr_model):
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    krr_model.fit(X, y)
    
    y_pred = krr_model.predict(X)
    
    # Kiểm tra tính đồng biến: Nếu x tăng thì y_pred phải tăng
    # Điều này đảm bảo mô hình đã học được xu hướng của dữ liệu
    diffs = np.diff(y_pred)
    assert np.all(diffs > 0), "Mô hình dự đoán không đồng biến với dữ liệu đầu vào"

def test_predict_handles_1d_input(krr_model):
    # Test trường hợp truyền mảng 1 chiều thay vì 2 chiều
    krr_model.fit(np.array([[1], [2]]), np.array([1, 2]))
    
    try:
        X_1d = np.array([1, 2, 3])
        krr_model.predict(X_1d)
    except Exception as e:
        pytest.fail(f"predict() bị crash khi nhận đầu vào 1D: {e}")