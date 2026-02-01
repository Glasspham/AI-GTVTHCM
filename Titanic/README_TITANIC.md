# So sánh Mô hình ANN và Deep Learning trên Dataset Titanic

## 📋 Mô tả Bài tập

Bài tập này thực hiện so sánh hiệu suất giữa hai loại mô hình Neural Network:
- **ANN (Artificial Neural Network)**: Mạng nơ-ron nông với 2 hidden layers
- **Deep Learning**: Mạng nơ-ron sâu với 5 hidden layers

**Dataset**: Titanic (Binary Classification - Sống/Chết)  
**Mục tiêu**: So sánh accuracy và các metrics đánh giá khác

## 🏗️ Kiến trúc Mô hình

### 1. ANN (Shallow Network)
```
Input (8 features) 
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (32 neurons, ReLU)
    ↓
Output (1 neuron, Sigmoid)
```

### 2. Deep Learning (Deep Network)
```
Input (8 features)
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
Output (1 neuron, Sigmoid)
```

## 📦 Cài đặt

### Yêu cầu
- Python 3.7+
- pip

### Cài đặt thư viện
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 🚀 Chạy chương trình

```bash
python titanic_ann_deep_comparison.py
```

## 📊 Kết quả

Chương trình sẽ xuất ra:

1. **Thông tin dataset**: Kích thước, phân bố, missing values
2. **Kết quả huấn luyện**:
   - Training Accuracy
   - Test Accuracy
   - Precision, Recall, F1-Score
   - Classification Report
   - Confusion Matrix

3. **So sánh chi tiết**:
   - Bảng so sánh các metrics
   - Phân tích chênh lệch
   - Số lượng parameters

4. **Biểu đồ trực quan**:
   - So sánh Accuracy
   - So sánh Precision, Recall, F1-Score
   - Confusion Matrix (cả 2 mô hình)
   - Learning Curves (Loss theo iterations)

5. **File output**: `titanic_comparison_results.png`

## 📈 Các bước thực hiện

1. **Tải dữ liệu**: Load dataset Titanic từ seaborn
2. **Tiền xử lý**:
   - Xử lý missing values
   - Mã hóa biến categorical (sex, embarked)
   - Chuẩn hóa dữ liệu (StandardScaler)
3. **Chia dữ liệu**: Train/Test split (80/20)
4. **Huấn luyện ANN**: MLPClassifier với 2 hidden layers
5. **Huấn luyện Deep Learning**: MLPClassifier với 5 hidden layers
6. **Đánh giá**: So sánh accuracy và các metrics
7. **Trực quan hóa**: Tạo biểu đồ so sánh
8. **Kết luận**: Phân tích và khuyến nghị

## 🔍 Features được sử dụng

- `pclass`: Hạng vé (1, 2, 3)
- `sex`: Giới tính (male/female)
- `age`: Tuổi
- `sibsp`: Số anh chị em/vợ chồng trên tàu
- `parch`: Số cha mẹ/con cái trên tàu
- `fare`: Giá vé
- `embarked`: Cảng lên tàu (C, Q, S)
- `alone`: Đi một mình hay không

## 📝 Metrics đánh giá

- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Độ chính xác của dự đoán positive
- **Recall**: Khả năng tìm ra các trường hợp positive
- **F1-Score**: Trung bình điều hòa của Precision và Recall

## 🎯 Kết luận dự kiến

Chương trình sẽ tự động phân tích và đưa ra kết luận về:
- Mô hình nào cho kết quả tốt hơn
- Chênh lệch accuracy giữa 2 mô hình
- Độ phức tạp (số lượng parameters)
- Khuyến nghị sử dụng mô hình nào

## 📚 Tài liệu tham khảo

- [Scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Titanic Dataset](https://www.kaggle.com/c/titanic)

## 👨‍💻 Tác giả

Bài tập Machine Learning - So sánh ANN và Deep Learning

---

**Lưu ý**: Kết quả có thể thay đổi tùy thuộc vào random_state và quá trình huấn luyện.
