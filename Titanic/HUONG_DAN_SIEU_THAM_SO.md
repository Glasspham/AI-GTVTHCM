# Hướng dẫn Chỉnh Siêu Tham Số

## 📝 Vị trí chỉnh sửa

Mở file `titanic_comparison.py` và tìm phần **CẤU HÌNH SIÊU THAM SỐ** (dòng 15-45)

## ⚙️ Các siêu tham số có thể chỉnh

### 1. **hidden_layers** - Kiến trúc mạng

Số lượng neurons trong mỗi hidden layer.

```python
# Ví dụ:
'hidden_layers': (64, 32)           # 2 layers: 64 và 32 neurons
'hidden_layers': (128, 64, 32)      # 3 layers
'hidden_layers': (100,)             # 1 layer với 100 neurons
'hidden_layers': (256, 128, 64, 32) # 4 layers
```

**Khuyến nghị:**

- ANN (Shallow): 1-3 layers, mỗi layer 32-128 neurons
- Deep Learning: 4-7 layers, mỗi layer 16-256 neurons

---

### 2. **activation** - Hàm kích hoạt

Hàm phi tuyến giữa các layer.

```python
# Các lựa chọn:
'activation': 'relu'      # ✅ Tốt nhất cho hầu hết trường hợp
'activation': 'tanh'      # Tốt cho dữ liệu đã chuẩn hóa
'activation': 'logistic'  # Sigmoid, chậm hơn relu
```

**Khuyến nghị:** Dùng `'relu'` (mặc định)

---

### 3. **learning_rate** - Chiến lược learning rate

```python
# Các lựa chọn:
'learning_rate': 'adaptive'    # ✅ Tự động giảm khi không cải thiện
'learning_rate': 'constant'    # Giữ nguyên learning rate
'learning_rate': 'invscaling'  # Giảm dần theo công thức
```

**Khuyến nghị:** Dùng `'adaptive'` (mặc định)

---

### 4. **learning_rate_init** - Learning rate ban đầu

Tốc độ học của mô hình.

```python
# Ví dụ:
'learning_rate_init': 0.001   # ✅ Mặc định, tốt cho hầu hết
'learning_rate_init': 0.01    # Học nhanh hơn (có thể bỏ qua minimum)
'learning_rate_init': 0.0001  # Học chậm hơn (ổn định hơn)
```

**Khuyến nghị:**

- Bắt đầu với `0.001`
- Nếu loss dao động: giảm xuống `0.0001`
- Nếu học quá chậm: tăng lên `0.01`

---

### 5. **max_iter** - Số epochs tối đa

```python
# Ví dụ:
'max_iter': 500    # ✅ Mặc định
'max_iter': 1000   # Cho phép huấn luyện lâu hơn
'max_iter': 200    # Huấn luyện nhanh (có thể chưa hội tụ)
```

**Khuyến nghị:**

- Dataset nhỏ: 200-500
- Dataset lớn: 500-1000

---

### 6. **alpha** - L2 Regularization

Giảm overfitting bằng cách phạt weights lớn.

```python
# Ví dụ:
'alpha': 0.0001    # ✅ Mặc định, regularization nhẹ
'alpha': 0.001     # Regularization mạnh hơn (giảm overfitting)
'alpha': 0.00001   # Regularization yếu hơn
'alpha': 0         # Không regularization
```

**Khuyến nghị:**

- Nếu overfitting (train acc >> test acc): tăng alpha lên `0.001` hoặc `0.01`
- Nếu underfitting: giảm alpha xuống `0.00001` hoặc `0`

---

### 7. **batch_size** - Kích thước batch

```python
# Ví dụ:
'batch_size': 'auto'  # ✅ Mặc định, tự động = min(200, n_samples)
'batch_size': 32      # Batch nhỏ (cập nhật weights thường xuyên)
'batch_size': 64      # Batch trung bình
'batch_size': 128     # Batch lớn (huấn luyện nhanh hơn)
```

**Khuyến nghị:**

- Dataset nhỏ (<1000): `'auto'` hoặc `32`
- Dataset lớn: `64` hoặc `128`

---

## 🔧 Ví dụ Cấu Hình

### Cấu hình 1: Tăng độ phức tạp ANN

```python
ANN_CONFIG = {
    'hidden_layers': (128, 64, 32),     # Thêm 1 layer
    'activation': 'relu',
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'alpha': 0.0001,
    'batch_size': 'auto',
}
```

### Cấu hình 2: Giảm overfitting

```python
DEEP_CONFIG = {
    'hidden_layers': (128, 64, 64, 32, 16),
    'activation': 'relu',
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'alpha': 0.01,                      # Tăng regularization
    'batch_size': 64,                   # Batch lớn hơn
}
```

### Cấu hình 3: Học chậm và ổn định

```python
ANN_CONFIG = {
    'hidden_layers': (64, 32),
    'activation': 'relu',
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.0001,       # Learning rate nhỏ
    'max_iter': 1000,                   # Nhiều epochs hơn
    'alpha': 0.0001,
    'batch_size': 32,                   # Batch nhỏ
}
```

### Cấu hình 4: Thử activation function khác

```python
DEEP_CONFIG = {
    'hidden_layers': (128, 64, 64, 32, 16),
    'activation': 'tanh',               # Thử tanh thay vì relu
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'alpha': 0.0001,
    'batch_size': 'auto',
}
```

---

## 📊 Cách Thử Nghiệm

1. **Chỉnh 1 tham số tại 1 thời điểm** để biết tác động của nó
2. **Ghi lại kết quả** sau mỗi lần chạy
3. **So sánh Test Accuracy** để chọn cấu hình tốt nhất

### Quy trình thử nghiệm:

```bash
# 1. Chỉnh cấu hình trong file
# 2. Chạy thử nghiệm
python titanic_comparison.py

# 3. Ghi lại kết quả
# 4. Thử cấu hình khác
```

---

## 🎯 Mục Tiêu Tối Ưu

### Nếu muốn tăng Test Accuracy:

1. Thử tăng số neurons: `(128, 64)` → `(256, 128)`
2. Thử thêm layers: `(64, 32)` → `(64, 32, 16)`
3. Giảm learning rate: `0.001` → `0.0001`
4. Tăng epochs: `500` → `1000`

### Nếu bị Overfitting (Train >> Test):

1. Tăng regularization: `alpha=0.0001` → `alpha=0.01`
2. Giảm số neurons: `(128, 64)` → `(64, 32)`
3. Giảm số layers
4. Tăng batch size: `32` → `128`

### Nếu bị Underfitting (Train và Test đều thấp):

1. Tăng số neurons và layers
2. Giảm regularization: `alpha=0.001` → `alpha=0.0001`
3. Tăng epochs
4. Thử activation function khác

---

## 📝 Ghi Chú

- **Early Stopping**: Mô hình sẽ tự động dừng nếu không cải thiện sau 10 epochs
- **Validation**: 10% train set được dùng để validation
- **Random State**: Đặt `RANDOM_STATE=42` để kết quả có thể tái lập

---

**Chúc bạn thử nghiệm thành công!** 🚀
