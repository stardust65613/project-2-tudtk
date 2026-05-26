import pytest
import numpy as np
from part1.ols_implementation import ols_fit 
from part1.cross_validation import kfold_cv 

def test_kfold_cv_output_format():
    """Test 1: Kiểm tra cấu trúc dữ liệu trả về."""
    # Tạo dữ liệu giả lập
    n, p = 20, 3
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    k = 4
    
    mean_mse, mses = kfold_cv(X, y, k=k)
    
    # Kiểm tra số lượng fold
    assert len(mses) == k, "Số lượng MSE trong list phải bằng k"
    # Kiểm tra giá trị trả về là số thực
    assert isinstance(mean_mse, float), "Mean MSE phải là một số thực"
    assert all(isinstance(m, float) or isinstance(m, np.float64) for m in mses), "Mỗi MSE phải là số thực"

def test_kfold_cv_reproducibility():
    """Test 2: Kiểm tra tính tái lập (đảm bảo seed hoạt động)."""
    n, p = 20, 2
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    seed = 123
    
    # Chạy 2 lần với cùng một seed
    mean_mse1, mses1 = kfold_cv(X, y, k=5, seed=seed)
    mean_mse2, mses2 = kfold_cv(X, y, k=5, seed=seed)
    
    # Kết quả phải khớp nhau hoàn toàn
    np.testing.assert_allclose(mses1, mses2, err_msg="Kết quả không đồng nhất khi dùng cùng seed")
    assert mean_mse1 == mean_mse2

def test_kfold_cv_full_data_usage():
    """Test 3: Kiểm tra xem kfold có sử dụng đúng dữ liệu không (logic indexing)."""
    # Dữ liệu rất đơn giản: y = 0
    n = 10
    X = np.random.rand(n, 1)
    y = np.zeros(n)
    
    mean_mse, _ = kfold_cv(X, y, k=5)
    
    # Với y=0, mô hình OLS hoàn hảo thì MSE phải bằng 0
    assert mean_mse >= 0 and mean_mse < 1e-10, "MSE phải xấp xỉ bằng 0 đối với dữ liệu y=0"