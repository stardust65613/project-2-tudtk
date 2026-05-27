import numpy as np
import pandas as pd

class DataPipeline:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.medians = {}
        self.means = {}
        self.stds = {}
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, df_train):
        """
        Chỉ tính toán các thông số thống kê trên tập huấn luyện (Train Set)
        """
        for col in self.feature_cols:
            # 1. Tính toán khoảng tứ phân vị IQR để xác định outlier
            q1 = df_train[col].quantile(0.25)
            q3 = df_train[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds[col] = q1 - 1.5 * iqr
            self.upper_bounds[col] = q3 + 1.5 * iqr
            
            # Lọc phân phối không chứa giá trị khuyết để tìm điểm trung vị chuẩn xác
            v_col = df_train[col].values
            v_valid = v_col[~np.isnan(v_col)]
            
            self.medians[col] = np.median(v_valid) if len(v_valid) > 0 else 0.0

        # 2. Giả lập điền khuyết tập train để tính toán chính xác Mean và Std phục vụ Scaling
        df_imputed = df_train[self.feature_cols].copy()
        for col in self.feature_cols:
            df_imputed[col] = df_imputed[col].fillna(self.medians[col])
            # Thực hiện kỹ thuật Winsorization để thu hẹp biên độ nhiễu ngoại lai
            df_imputed[col] = np.clip(df_imputed[col], self.lower_bounds[col], self.upper_bounds[col])
            
            self.means[col] = df_imputed[col].mean()
            self.stds[col] = df_imputed[col].std()
            if self.stds[col] == 0:
                self.stds[col] = 1e-10

    def transform(self, df_input):
        """
        Áp đặt các tham số thống kê của tập Train lên một tập dữ liệu bất kỳ
        """
        df_out = df_input[self.feature_cols].copy()
        for col in self.feature_cols:
            # Bước 1: Điền khuyết bằng trung vị của tập Train
            df_out[col] = df_out[col].fillna(self.medians[col])
            # Bước 2: Winsorization xử lý triệt để ngoại lai
            df_out[col] = np.clip(df_out[col], self.lower_bounds[col], self.upper_bounds[col])
            # Bước 3: Chuẩn hóa phân phối Z-score toán học
            df_out[col] = (df_out[col] - self.means[col]) / self.stds[col]
        return df_out.values