import pytest
import pandas as pd
import numpy as np
from part2.data_pipeline import DataPipeline

def test_fit_calculates_correct_stats():
    data = {'A': [10, 200, 32, 400, 1000, np.nan]}
    df = pd.DataFrame(data)
    
    # Lấy dữ liệu sạch để tính toán kỳ vọng (giống cách hàm fit làm)
    clean_data = df['A'].dropna()
    q1 = np.quantile(clean_data, 0.25)
    q3 = np.quantile(clean_data, 0.75)
    iqr = q3 - q1
    expected_lower = q1 - 1.5 * iqr
    expected_upper = q3 + 1.5 * iqr

    preprocessor = DataPipeline(feature_cols=['A'])
    preprocessor.fit(df)
    
    # Kiểm tra
    assert preprocessor.medians['A'] == np.median(clean_data)
    assert preprocessor.lower_bounds['A'] == pytest.approx(expected_lower)
    assert preprocessor.upper_bounds['A'] == pytest.approx(expected_upper)

def test_fit_with_empty_data():
    # Kiểm tra trường hợp dữ liệu toàn là NaN
    df = pd.DataFrame({'A': [np.nan, np.nan]})
    preprocessor = DataPipeline(feature_cols=['A'])
    
    preprocessor.fit(df)
    
    assert preprocessor.medians['A'] == 0.0

def test_transform_applies_correct_rules():
    # 1. Setup: Giả lập một instance của lớp Preprocessor
    preprocessor = DataPipeline(feature_cols=['A'])
    
    # Ép buộc các tham số đã học
    preprocessor.medians = {'A': 20.0}
    preprocessor.lower_bounds = {'A': 10.0}
    preprocessor.upper_bounds = {'A': 40.0}
    preprocessor.means = {'A': 25.0}
    preprocessor.stds = {'A': 10.0}
    
    # 2. Input dữ liệu: Chứa giá trị cần điền khuyết (NaN), outlier (>40), và giá trị bình thường
    df_test = pd.DataFrame({'A': [np.nan, 50.0, 30.0]})
    
    # 3. Thực thi
    result = preprocessor.transform(df_test)
    
    # 4. Kiểm chứng từng bước (Logic Pipeline)
    # Bước 1 (Fillna): [20.0, 50.0, 30.0]
    # Bước 2 (Clip): [20.0, 40.0, 30.0] (50 bị clip về 40)
    # Bước 3 (Z-score): 
    #   (20 - 25) / 10 = -0.5
    #   (40 - 25) / 10 = 1.5
    #   (30 - 25) / 10 = 0.5
    
    expected_result = np.array([-0.5, 1.5, 0.5])
    
    assert np.allclose(result.flatten(), expected_result)
    assert result.shape == (3, 1)

def test_transform_does_not_modify_original():
    # Đảm bảo không làm thay đổi dữ liệu đầu vào
    preprocessor = DataPipeline(feature_cols=['A'])
    preprocessor.medians = {'A': 0}
    preprocessor.lower_bounds = {'A': -10}
    preprocessor.upper_bounds = {'A': 10}
    preprocessor.means = {'A': 0}
    preprocessor.stds = {'A': 1}
    
    df_orig = pd.DataFrame({'A': [5.0]})
    df_copy = df_orig.copy()
    
    _ = preprocessor.transform(df_orig)
    
    # Kiểm tra df_orig vẫn giữ nguyên
    pd.testing.assert_frame_equal(df_orig, df_copy)