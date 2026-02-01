# KẾT QUẢ SO SÁNH MÔ HÌNH ANN VÀ DEEP LEARNING

## 📊 Dataset: Titanic (Sống/Chết)

- **Tổng số mẫu**: 891
- **Train set**: 712 mẫu (80%)
- **Test set**: 179 mẫu (20%)
- **Số features**: 8 (pclass, sex, age, sibsp, parch, fare, embarked, alone)
- **Target**: survived (0 = Chết, 1 = Sống)
- **Tỷ lệ sống**: 38.38%

---

## 🏗️ KIẾN TRÚC MÔ HÌNH

### 1. ANN (Shallow Neural Network)

```
Input Layer (8 neurons)
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (32 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

**Thông số:**

- Số hidden layers: **2**
- Tổng số parameters: **2,592**
- Optimizer: Adam
- Số iterations: 28
- Loss cuối cùng: 0.4100

### 2. Deep Learning (Deep Neural Network)

```
Input Layer (8 neurons)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Hidden Layer 3 (64 neurons, ReLU)
    ↓
Hidden Layer 4 (32 neurons, ReLU)
    ↓
Hidden Layer 5 (16 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

**Thông số:**

- Số hidden layers: **5**
- Tổng số parameters: **15,888**
- Optimizer: Adam
- Số iterations: 37
- Loss cuối cùng: 0.3575

---

## 📈 KẾT QUẢ ĐÁNH GIÁ

### Bảng so sánh tổng quan

| Metric                | ANN (Shallow) | Deep Learning | Chênh lệch |
| --------------------- | ------------- | ------------- | ---------- |
| **Training Accuracy** | 79.78%        | 83.15%        | +3.37%     |
| **Test Accuracy**     | **79.33%**    | **81.01%**    | **+1.68%** |
| **Precision**         | 0.9000        | 0.8182        | -0.0818    |
| **Recall**            | 0.5217        | 0.6522        | +0.1305    |
| **F1-Score**          | 0.6606        | 0.7258        | +0.0652    |

### Chi tiết Classification Report

#### ANN (Shallow Network)

```
              precision    recall  f1-score   support
        Died       0.76      0.96      0.85       110
    Survived       0.90      0.52      0.66        69

    accuracy                           0.79       179
   macro avg       0.83      0.74      0.76       179
weighted avg       0.82      0.79      0.78       179
```

#### Deep Learning (Deep Network)

```
              precision    recall  f1-score   support
        Died       0.81      0.91      0.85       110
    Survived       0.82      0.65      0.73        69

    accuracy                           0.81       179
   macro avg       0.81      0.78      0.79       179
weighted avg       0.81      0.81      0.81       179
```

---

## 🔍 PHÂN TÍCH CHI TIẾT

### 1. Accuracy

- **Deep Learning tốt hơn ANN: +1.68%**
- Deep Learning: 81.01%
- ANN: 79.33%
- Cả 2 mô hình đều đạt accuracy khá tốt (>79%)

### 2. Precision vs Recall

- **ANN**: Precision cao (0.90) nhưng Recall thấp (0.52)
  - Dự đoán "Survived" rất chính xác nhưng bỏ sót nhiều trường hợp
  - Phù hợp khi cần độ chính xác cao
- **Deep Learning**: Cân bằng hơn
  - Precision: 0.82, Recall: 0.65
  - F1-Score cao hơn (0.7258 vs 0.6606)
  - Phù hợp cho bài toán tổng quát

### 3. Độ phức tạp

- **Deep Learning có 6.1x parameters hơn ANN**
  - Deep: 15,888 parameters
  - ANN: 2,592 parameters
  - Chênh lệch: 13,296 parameters

### 4. Thời gian huấn luyện

- **Deep Learning**: 37 iterations
- **ANN**: 28 iterations
- Deep Learning cần nhiều iterations hơn để hội tụ

### 5. Loss Function

- **Deep Learning**: Loss thấp hơn (0.3575 vs 0.4100)
- Cho thấy Deep Learning học được patterns tốt hơn

---

## 💡 NHẬN XÉT

### Ưu điểm của Deep Learning:

✅ **Accuracy cao hơn** (+1.68%)  
✅ **F1-Score tốt hơn** (+0.0652)  
✅ **Recall cao hơn** (+0.1305) - Phát hiện được nhiều trường hợp "Survived" hơn  
✅ **Loss thấp hơn** - Học được patterns tốt hơn  
✅ **Cân bằng giữa Precision và Recall**

### Ưu điểm của ANN:

✅ **Đơn giản hơn** - Ít parameters hơn 6.1 lần  
✅ **Huấn luyện nhanh hơn** - Ít iterations hơn  
✅ **Precision rất cao** (0.90) - Ít dự đoán sai "Survived"  
✅ **Ít overfitting hơn** - Khoảng cách Train-Test accuracy nhỏ hơn

### Nhược điểm của Deep Learning:

❌ **Phức tạp hơn nhiều** - 15,888 parameters  
❌ **Dễ overfitting** - Train accuracy cao hơn Test accuracy 2.14%  
❌ **Cần nhiều dữ liệu hơn** để phát huy hết tiềm năng

### Nhược điểm của ANN:

❌ **Recall thấp** (0.52) - Bỏ sót nhiều trường hợp "Survived"  
❌ **Accuracy thấp hơn** Deep Learning

---

## 🎯 KẾT LUẬN VÀ KHUYẾN NGHỊ

### Kết luận:

1. **Deep Learning cho kết quả tốt hơn ANN** với chênh lệch accuracy **+1.68%**
2. Tuy nhiên, **chênh lệch không quá lớn** (< 2%)
3. Deep Learning **phức tạp hơn 6.1 lần** về số lượng parameters
4. Với dataset nhỏ như Titanic (891 mẫu), **ANN đã đủ hiệu quả**

### Khuyến nghị:

#### ✅ Nên dùng **ANN (Shallow Network)** khi:

- Dataset nhỏ (< 10,000 mẫu)
- Cần mô hình đơn giản, dễ giải thích
- Ưu tiên Precision (độ chính xác dự đoán)
- Tài nguyên tính toán hạn chế
- Cần huấn luyện nhanh

#### ✅ Nên dùng **Deep Learning** khi:

- Dataset lớn (> 100,000 mẫu)
- Cần accuracy cao nhất có thể
- Ưu tiên Recall (phát hiện nhiều trường hợp positive)
- Cần F1-Score cân bằng
- Có đủ tài nguyên tính toán

### Cho bài toán Titanic này:

**→ Khuyến nghị sử dụng ANN (Shallow Network)**

**Lý do:**

1. Dataset nhỏ (891 mẫu) - không cần Deep Learning
2. Hiệu quả tương đương (chênh lệch chỉ 1.68%)
3. Đơn giản hơn nhiều (2,592 vs 15,888 parameters)
4. Ít overfitting hơn
5. Huấn luyện nhanh hơn

---

## 📁 Files đã tạo

1. **titanic_ann_deep_comparison.py** - Script chính
2. **titanic_comparison_results.png** - Biểu đồ so sánh
3. **README_TITANIC.md** - Hướng dẫn sử dụng
4. **KET_QUA_SO_SANH.md** - File này (Tóm tắt kết quả)

---

## 🔗 Tài liệu tham khảo

- Scikit-learn MLPClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- Titanic Dataset: https://www.kaggle.com/c/titanic
- Neural Networks: https://www.deeplearningbook.org/

---

**Ngày thực hiện**: 16/01/2026  
**Công cụ**: Python 3.12, scikit-learn, pandas, matplotlib, seaborn
