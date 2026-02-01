"""
Bài tập: So sánh mô hình ANN và Deep Learning trên dataset Titanic
Dataset: Titanic (sống/chết) - Binary Classification
Mục tiêu: So sánh accuracy giữa ANN (shallow) và Deep Learning (deep network)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style cho plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("BÀI TẬP: SO SÁNH MÔ HÌNH ANN VÀ DEEP LEARNING - DATASET TITANIC")
print("="*70)

# ============================================================================
# BƯỚC 1: TẢI VÀ KHÁM PHÁ DỮ LIỆU
# ============================================================================
print("\n[BƯỚC 1] Tải và khám phá dữ liệu Titanic...")

# Tải dataset Titanic từ seaborn
titanic_df = sns.load_dataset('titanic')
print(f"\nKích thước dataset: {titanic_df.shape}")
print(f"Số lượng mẫu: {len(titanic_df)}")
print(f"Số lượng features: {titanic_df.shape[1]}")

print("\n5 dòng đầu tiên của dataset:")
print(titanic_df.head())

print("\nThông tin về dataset:")
print(titanic_df.info())

print("\nThống kê mô tả:")
print(titanic_df.describe())

print("\nPhân bố target (survived):")
print(titanic_df['survived'].value_counts())
print(f"Tỷ lệ sống: {titanic_df['survived'].mean()*100:.2f}%")

# ============================================================================
# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU
# ============================================================================
print("\n[BƯỚC 2] Tiền xử lý dữ liệu...")

# Tạo bản sao để xử lý
df = titanic_df.copy()

# Chọn các features quan trọng
features_to_use = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']
df = df[features_to_use + ['survived']]

print(f"\nCác features được sử dụng: {features_to_use}")

# Xử lý missing values
print("\nSố lượng missing values trước khi xử lý:")
print(df.isnull().sum())

# Điền missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

print("\nSố lượng missing values sau khi xử lý:")
print(df.isnull().sum())

# Mã hóa biến categorical
print("\nMã hóa các biến categorical...")

# Label encoding cho 'sex'
le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])

# Label encoding cho 'embarked'
le_embarked = LabelEncoder()
df['embarked'] = le_embarked.fit_transform(df['embarked'])

# Chuyển đổi 'alone' thành số
df['alone'] = df['alone'].astype(int)

print("\nDữ liệu sau khi mã hóa:")
print(df.head())

# ============================================================================
# BƯỚC 3: CHIA DỮ LIỆU VÀ CHUẨN HÓA
# ============================================================================
print("\n[BƯỚC 3] Chia dữ liệu và chuẩn hóa...")

# Tách features và target
X = df.drop('survived', axis=1)
y = df['survived']

print(f"\nShape của X: {X.shape}")
print(f"Shape của y: {y.shape}")

# Chia train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} mẫu")
print(f"Test set: {X_test.shape[0]} mẫu")

# Chuẩn hóa dữ liệu (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDữ liệu đã được chuẩn hóa (mean=0, std=1)")

# ============================================================================
# BƯỚC 4: XÂY DỰNG MÔ HÌNH ANN (SHALLOW NETWORK)
# ============================================================================
print("\n" + "="*70)
print("[BƯỚC 4] Xây dựng mô hình ANN (Shallow Neural Network)")
print("="*70)

# Mô hình ANN: 1-2 hidden layers
# Kiến trúc: Input -> Hidden(64) -> Hidden(32) -> Output
ann_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2 hidden layers
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False
)

print("\nKiến trúc mô hình ANN:")
print(f"  - Input layer: {X_train.shape[1]} neurons")
print(f"  - Hidden layer 1: 64 neurons (ReLU)")
print(f"  - Hidden layer 2: 32 neurons (ReLU)")
print(f"  - Output layer: 1 neuron (Sigmoid)")
print(f"  - Optimizer: Adam")
print(f"  - Max iterations: 500")

print("\nĐang huấn luyện mô hình ANN...")
ann_model.fit(X_train_scaled, y_train)

print(f"Số iterations thực tế: {ann_model.n_iter_}")
print(f"Loss cuối cùng: {ann_model.loss_:.4f}")

# Dự đoán
y_train_pred_ann = ann_model.predict(X_train_scaled)
y_test_pred_ann = ann_model.predict(X_test_scaled)

# Đánh giá
ann_train_acc = accuracy_score(y_train, y_train_pred_ann)
ann_test_acc = accuracy_score(y_test, y_test_pred_ann)
ann_precision = precision_score(y_test, y_test_pred_ann)
ann_recall = recall_score(y_test, y_test_pred_ann)
ann_f1 = f1_score(y_test, y_test_pred_ann)

print("\n" + "-"*70)
print("KẾT QUẢ MÔ HÌNH ANN (SHALLOW NETWORK)")
print("-"*70)
print(f"Training Accuracy:   {ann_train_acc*100:.2f}%")
print(f"Test Accuracy:       {ann_test_acc*100:.2f}%")
print(f"Precision:           {ann_precision:.4f}")
print(f"Recall:              {ann_recall:.4f}")
print(f"F1-Score:            {ann_f1:.4f}")

print("\nClassification Report (ANN):")
print(classification_report(y_test, y_test_pred_ann, target_names=['Died', 'Survived']))

# ============================================================================
# BƯỚC 5: XÂY DỰNG MÔ HÌNH DEEP LEARNING (DEEP NETWORK)
# ============================================================================
print("\n" + "="*70)
print("[BƯỚC 5] Xây dựng mô hình Deep Learning (Deep Neural Network)")
print("="*70)

# Mô hình Deep: 5 hidden layers
# Kiến trúc: Input -> Hidden(128) -> Hidden(64) -> Hidden(64) -> Hidden(32) -> Hidden(16) -> Output
deep_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 64, 32, 16),  # 5 hidden layers
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False
)

print("\nKiến trúc mô hình Deep Learning:")
print(f"  - Input layer: {X_train.shape[1]} neurons")
print(f"  - Hidden layer 1: 128 neurons (ReLU)")
print(f"  - Hidden layer 2: 64 neurons (ReLU)")
print(f"  - Hidden layer 3: 64 neurons (ReLU)")
print(f"  - Hidden layer 4: 32 neurons (ReLU)")
print(f"  - Hidden layer 5: 16 neurons (ReLU)")
print(f"  - Output layer: 1 neuron (Sigmoid)")
print(f"  - Optimizer: Adam")
print(f"  - Max iterations: 500")

print("\nĐang huấn luyện mô hình Deep Learning...")
deep_model.fit(X_train_scaled, y_train)

print(f"Số iterations thực tế: {deep_model.n_iter_}")
print(f"Loss cuối cùng: {deep_model.loss_:.4f}")

# Dự đoán
y_train_pred_deep = deep_model.predict(X_train_scaled)
y_test_pred_deep = deep_model.predict(X_test_scaled)

# Đánh giá
deep_train_acc = accuracy_score(y_train, y_train_pred_deep)
deep_test_acc = accuracy_score(y_test, y_test_pred_deep)
deep_precision = precision_score(y_test, y_test_pred_deep)
deep_recall = recall_score(y_test, y_test_pred_deep)
deep_f1 = f1_score(y_test, y_test_pred_deep)

print("\n" + "-"*70)
print("KẾT QUẢ MÔ HÌNH DEEP LEARNING (DEEP NETWORK)")
print("-"*70)
print(f"Training Accuracy:   {deep_train_acc*100:.2f}%")
print(f"Test Accuracy:       {deep_test_acc*100:.2f}%")
print(f"Precision:           {deep_precision:.4f}")
print(f"Recall:              {deep_recall:.4f}")
print(f"F1-Score:            {deep_f1:.4f}")

print("\nClassification Report (Deep Learning):")
print(classification_report(y_test, y_test_pred_deep, target_names=['Died', 'Survived']))

# ============================================================================
# BƯỚC 6: SO SÁNH KẾT QUẢ
# ============================================================================
print("\n" + "="*70)
print("[BƯỚC 6] SO SÁNH KẾT QUẢ GIỮA ANN VÀ DEEP LEARNING")
print("="*70)

# Tạo bảng so sánh
comparison_df = pd.DataFrame({
    'Metric': ['Training Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'ANN (Shallow)': [
        f"{ann_train_acc*100:.2f}%",
        f"{ann_test_acc*100:.2f}%",
        f"{ann_precision:.4f}",
        f"{ann_recall:.4f}",
        f"{ann_f1:.4f}"
    ],
    'Deep Learning': [
        f"{deep_train_acc*100:.2f}%",
        f"{deep_test_acc*100:.2f}%",
        f"{deep_precision:.4f}",
        f"{deep_recall:.4f}",
        f"{deep_f1:.4f}"
    ]
})

print("\nBảng so sánh chi tiết:")
print(comparison_df.to_string(index=False))

# So sánh accuracy
print("\n" + "-"*70)
print("PHÂN TÍCH SO SÁNH:")
print("-"*70)

acc_diff = deep_test_acc - ann_test_acc
print(f"\nChênh lệch Test Accuracy: {acc_diff*100:.2f}%")

if acc_diff > 0:
    print(f"✓ Mô hình Deep Learning tốt hơn ANN {acc_diff*100:.2f}%")
elif acc_diff < 0:
    print(f"✓ Mô hình ANN tốt hơn Deep Learning {abs(acc_diff)*100:.2f}%")
else:
    print("✓ Hai mô hình có accuracy tương đương")

# Số lượng parameters
ann_params = sum([layer.size for layer in ann_model.coefs_])
deep_params = sum([layer.size for layer in deep_model.coefs_])

print(f"\nSố lượng parameters:")
print(f"  - ANN: {ann_params:,} parameters")
print(f"  - Deep Learning: {deep_params:,} parameters")
print(f"  - Deep có nhiều hơn: {deep_params - ann_params:,} parameters")

# ============================================================================
# BƯỚC 7: TRỰC QUAN HÓA KẾT QUẢ
# ============================================================================
print("\n[BƯỚC 7] Tạo các biểu đồ trực quan hóa...")

# Tạo figure với nhiều subplots
fig = plt.figure(figsize=(16, 12))

# 1. So sánh Accuracy
ax1 = plt.subplot(2, 3, 1)
metrics = ['Train Acc', 'Test Acc']
ann_scores = [ann_train_acc*100, ann_test_acc*100]
deep_scores = [deep_train_acc*100, deep_test_acc*100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, ann_scores, width, label='ANN (Shallow)', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, deep_scores, width, label='Deep Learning', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Metrics', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('So sánh Accuracy', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Thêm giá trị lên bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

# 2. So sánh các metrics khác
ax2 = plt.subplot(2, 3, 2)
metrics2 = ['Precision', 'Recall', 'F1-Score']
ann_scores2 = [ann_precision, ann_recall, ann_f1]
deep_scores2 = [deep_precision, deep_recall, deep_f1]

x2 = np.arange(len(metrics2))
bars3 = ax2.bar(x2 - width/2, ann_scores2, width, label='ANN (Shallow)', color='#3498db', alpha=0.8)
bars4 = ax2.bar(x2 + width/2, deep_scores2, width, label='Deep Learning', color='#e74c3c', alpha=0.8)

ax2.set_xlabel('Metrics', fontsize=11, fontweight='bold')
ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('So sánh Precision, Recall, F1-Score', fontsize=12, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(metrics2)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 3. Confusion Matrix - ANN
ax3 = plt.subplot(2, 3, 3)
cm_ann = confusion_matrix(y_test, y_test_pred_ann)
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=True)
ax3.set_title('Confusion Matrix - ANN', fontsize=12, fontweight='bold')
ax3.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax3.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax3.set_xticklabels(['Died', 'Survived'])
ax3.set_yticklabels(['Died', 'Survived'])

# 4. Confusion Matrix - Deep Learning
ax4 = plt.subplot(2, 3, 4)
cm_deep = confusion_matrix(y_test, y_test_pred_deep)
sns.heatmap(cm_deep, annot=True, fmt='d', cmap='Reds', ax=ax4, cbar=True)
ax4.set_title('Confusion Matrix - Deep Learning', fontsize=12, fontweight='bold')
ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax4.set_xticklabels(['Died', 'Survived'])
ax4.set_yticklabels(['Died', 'Survived'])

# 5. Learning Curves - ANN
ax5 = plt.subplot(2, 3, 5)
if hasattr(ann_model, 'loss_curve_'):
    ax5.plot(ann_model.loss_curve_, color='#3498db', linewidth=2, label='ANN Loss')
    ax5.set_xlabel('Iterations', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax5.set_title('Learning Curve - ANN', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

# 6. Learning Curves - Deep Learning
ax6 = plt.subplot(2, 3, 6)
if hasattr(deep_model, 'loss_curve_'):
    ax6.plot(deep_model.loss_curve_, color='#e74c3c', linewidth=2, label='Deep Learning Loss')
    ax6.set_xlabel('Iterations', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax6.set_title('Learning Curve - Deep Learning', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('c:/Users/VAN_VO/OneDrive/Documents/AI/titanic_comparison_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Đã lưu biểu đồ: titanic_comparison_results.png")

# ============================================================================
# BƯỚC 8: KẾT LUẬN
# ============================================================================
print("\n" + "="*70)
print("KẾT LUẬN")
print("="*70)

print("\n1. KIẾN TRÚC MÔ HÌNH:")
print(f"   - ANN (Shallow): 2 hidden layers (64, 32)")
print(f"   - Deep Learning: 5 hidden layers (128, 64, 64, 32, 16)")

print("\n2. HIỆU SUẤT:")
print(f"   - ANN Test Accuracy: {ann_test_acc*100:.2f}%")
print(f"   - Deep Learning Test Accuracy: {deep_test_acc*100:.2f}%")

print("\n3. ĐỘ PHỨC TạP:")
print(f"   - ANN: {ann_params:,} parameters")
print(f"   - Deep Learning: {deep_params:,} parameters")

print("\n4. NHẬN XÉT:")
if deep_test_acc > ann_test_acc:
    print("   ✓ Mô hình Deep Learning cho kết quả tốt hơn ANN")
    print(f"   ✓ Tăng {(deep_test_acc - ann_test_acc)*100:.2f}% accuracy")
elif ann_test_acc > deep_test_acc:
    print("   ✓ Mô hình ANN cho kết quả tốt hơn Deep Learning")
    print(f"   ✓ Tăng {(ann_test_acc - deep_test_acc)*100:.2f}% accuracy")
    print("   ✓ ANN đơn giản hơn và có thể tránh overfitting tốt hơn")
else:
    print("   ✓ Hai mô hình có hiệu suất tương đương")

print("\n5. KHUYẾN NGHỊ:")
if deep_test_acc > ann_test_acc + 0.02:  # Nếu Deep tốt hơn >2%
    print("   → Nên sử dụng Deep Learning cho bài toán này")
elif ann_test_acc > deep_test_acc + 0.02:  # Nếu ANN tốt hơn >2%
    print("   → Nên sử dụng ANN (đơn giản hơn, ít overfitting hơn)")
else:
    print("   → Nên sử dụng ANN (đơn giản hơn, hiệu quả tương đương)")

print("\n" + "="*70)
print("HOÀN THÀNH!")
print("="*70)

plt.show()
