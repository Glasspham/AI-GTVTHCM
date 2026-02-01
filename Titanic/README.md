# So sánh ANN vs Deep Learning - Titanic Dataset

## 📋 Mô tả

So sánh hiệu suất giữa:

- **ANN (Shallow)**: 2 hidden layers (64, 32)
- **Deep Learning**: 5 hidden layers (128, 64, 64, 32, 16)

## 🚀 Chạy chương trình

### Script 1: So sánh đơn giản (2 mô hình)

```bash
python titanic_comparison.py
```

### Script 2: Thử nghiệm nhiều cấu hình (10 mô hình)

```bash
python titanic_grid_search.py
```

Script này sẽ:

- Thử nghiệm 5 cấu hình ANN khác nhau
- Thử nghiệm 5 cấu hình Deep Learning khác nhau
- Tạo bảng thống kê so sánh
- Tìm cấu hình tốt nhất
- Lưu kết quả vào `titanic_results.csv`

## ⚙️ Chỉnh siêu tham số

### Cách 1: Chỉnh trong `titanic_comparison.py`

Mở file và tìm phần **CẤU HÌNH SIÊU THAM SỐ** (dòng 15-45):

```python
# Cấu hình ANN
ANN_CONFIG = {
    'hidden_layers': (64, 32),          # Số neurons mỗi layer
    'activation': 'relu',                # relu, tanh, logistic
    'learning_rate_init': 0.001,        # Learning rate ban đầu
    'max_iter': 500,                    # Số epochs tối đa
    'alpha': 0.0001,                    # L2 regularization
}
```

### Cách 2: Thêm cấu hình vào `titanic_grid_search.py`

Thêm cấu hình mới vào list `ANN_CONFIGS` hoặc `DEEP_CONFIGS`:

```python
ANN_CONFIGS = [
    {
        'name': 'ANN-Custom',
        'hidden_layers': (256, 128),
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
    },
    # ... thêm cấu hình khác
]
```

### Các tham số có thể chỉnh:

- **hidden_layers**: Kiến trúc mạng - VD: `(128, 64)`, `(256, 128, 64)`
- **activation**: Hàm kích hoạt - `'relu'`, `'tanh'`, `'logistic'`
- **learning_rate_init**: Tốc độ học - `0.0001` đến `0.01`
- **max_iter**: Số epochs - `200` đến `1000`
- **alpha**: Regularization - `0` đến `0.01`

📖 **Xem chi tiết**: `HUONG_DAN_SIEU_THAM_SO.md`

## 📊 Kết quả mẫu

### Script 1: `titanic_comparison.py`

```
============================================================
KẾT QUẢ SO SÁNH
============================================================

┌─────────────────────┬──────────────┬──────────────┐
│ Metric              │ ANN (Shallow)│ Deep Learning│
├─────────────────────┼──────────────┼──────────────┤
│ Train Accuracy      │    79.78%    │    83.15%    │
│ Test Accuracy       │    79.33%    │    81.01%    │
│ Precision           │    0.9000    │    0.8182    │
│ Recall              │    0.5217    │    0.6522    │
│ F1-Score            │    0.6606    │    0.7258    │
└─────────────────────┴──────────────┴──────────────┘
```

### Script 2: `titanic_grid_search.py`

```
🏆 CẤU HÌNH TỐT NHẤT - ANN:
  • Tên: ANN-3: More neurons
  • Kiến trúc: (128, 64)
  • Test Accuracy: 81.01%
  • F1-Score: 0.7302
  • Parameters: 9,280

🏆 CẤU HÌNH TỐT NHẤT - DEEP LEARNING:
  • Tên: Deep-1: Baseline
  • Kiến trúc: (128, 64, 64, 32, 16)
  • Test Accuracy: 81.01%
  • F1-Score: 0.7258
  • Parameters: 15,888
```

## 💡 Kết luận

- Deep Learning tốt hơn ANN: **+1.68% accuracy**
- Nhưng ANN **đơn giản hơn 6.1 lần** về số parameters
- **Khuyến nghị**: Dùng ANN (đơn giản, hiệu quả tương đương)

## 📦 Yêu cầu

```bash
pip install pandas seaborn scikit-learn
```

## 📁 Files trong project

- `titanic_comparison.py` - So sánh đơn giản (2 mô hình)
- `titanic_grid_search.py` - Thử nghiệm nhiều cấu hình (10 mô hình)
- `titanic_results.csv` - Kết quả chi tiết (được tạo tự động)
- `HUONG_DAN_SIEU_THAM_SO.md` - Hướng dẫn chi tiết về siêu tham số
- `README.md` - File này
