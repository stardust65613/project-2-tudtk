# ĐỒ ÁN 2 Data Fitting và Phương Pháp OLS
## Nhóm Thực Hiện
* **Thành viên 1:** Nguyễn Hải Đăng - 23120027
* **Thành viên 2:** [Họ tên] - [MSSV]
* **Giảng viên hướng dẫn:** [Tên giảng viên]

---

## Mô tả
Dự án này tập trung vào việc nghiên cứu và triển khai các mô hình học máy (Machine Learning), bao gồm từ các kỹ thuật nền tảng như **OLS (Ordinary Least Squares)** được viết từ đầu (scratch) đến các phương pháp chính quy hóa như **Ridge/Lasso Regression**. Ngoài ra, dự án còn thực hiện quy trình xây dựng Pipeline để xử lý dữ liệu và đánh giá hiệu năng mô hình trên tập dữ liệu thực tế.

## Cài đặt môi trường và test

```bash
#### Chạy lệnh sau để tạo môi trường ảo
python -m venv .venv

#### Kích hoạt môi trường ảo
.venv\Scripts\activate   

#### Tắt môi trường ảo
deactivate

#### Cài dependencies
pip install -r requirements.txt

#### Chạy test
python -m pytest

#### Kiểm tra các thư viện đã cài
pip list
```

```markdown
## 📁 Cấu trúc dự án
```text
Group_<ID>/
├── README.md               # Hướng dẫn dự án
├── requirements.txt        # Danh sách thư viện
├── report/                 # Báo cáo (LaTeX & PDF)
├── part1/                  # Triển khai thuật toán từ cơ bản
│   ├── ols_implementation.py
│   ├── ridge_lasso.py
│   ├── part1_notebook.ipynb
    └── test/               # Chứa các unit test
└── part2/                  # Phân tích thực tế
    ├── data/               # Tập dữ liệu (.csv)
    ├── data_pipeline.py    # Xử lý dữ liệu
    ├── model_comparison.py # So sánh mô hình
    ├── part2_notebook.ipynb
    └── test/               # Chứa các unit test

## 🛠 Hướng dẫn sử dụng
### 1. Part 1: Cơ sở lý thuyết
Thư mục này tập trung vào việc tìm hiểu và triển khai các thuật toán từ nền tảng:
* `ols_implementation.py`: Cài đặt OLS thủ công bằng công thức ma trận.
* `ridge_lasso.py`: Triển khai Ridge và Lasso regression để xử lý hiện tượng overfitting.
* `part1_notebook.ipynb`: File Jupyter Notebook dùng để minh họa đồ thị và so sánh các kết quả tính toán thủ công.

### 2. Part 2: Ứng dụng thực tế
Quy trình thực hiện theo các bước sau:
* **Bước 1 (Tiền xử lý):** Chạy `data_pipeline.py` để làm sạch dữ liệu.
* **Bước 2 (Huấn luyện & So sánh):** Chạy `model_comparison.py` để so sánh hiệu năng các mô hình.
* **Bước 3 (Phân tích):** Mở `part2_notebook.ipynb` để xem các kết quả đồ thị và đánh giá chuyên sâu.

