"""
So sánh mô hình ANN và Deep Learning trên dataset Titanic
Hỗ trợ thử nghiệm nhiều siêu tham số khác nhau
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH SIÊU THAM SỐ - CHỈNH TẠI ĐÂY
# ============================================================

# Cấu hình ANN (Shallow Network)
ANN_CONFIG = {
    'hidden_layers': (64, 32),          # Số neurons mỗi layer
    'activation': 'tanh',                # relu, tanh, logistic
    'learning_rate': 'adaptive',         # constant, invscaling, adaptive
    'learning_rate_init': 0.001,        # Learning rate ban đầu
    'max_iter': 500,                    # Số epochs tối đa
    'alpha': 0.0001,                    # L2 regularization
    'batch_size': 'auto',               # Kích thước batch
}

# Cấu hình Deep Learning (Deep Network)
DEEP_CONFIG = {
    'hidden_layers': (128, 64, 64, 32, 16),  # Số neurons mỗi layer
    'activation': 'tanh',                     # relu, tanh, logistic
    'learning_rate': 'adaptive',              # constant, invscaling, adaptive
    'learning_rate_init': 0.001,             # Learning rate ban đầu
    'max_iter': 500,                         # Số epochs tối đa
    'alpha': 0.0001,                         # L2 regularization
    'batch_size': 'auto',                    # Kích thước batch
}

# Cấu hình chung
RANDOM_STATE = 42
TEST_SIZE = 0.2
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1

def train_model(config, name):
    """Huấn luyện mô hình với cấu hình cho trước"""
    model = MLPClassifier(
        hidden_layer_sizes=config['hidden_layers'],
        activation=config['activation'],
        solver='adam',
        learning_rate=config['learning_rate'],
        learning_rate_init=config['learning_rate_init'],
        max_iter=config['max_iter'],
        alpha=config['alpha'],
        batch_size=config['batch_size'],
        random_state=RANDOM_STATE,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        verbose=False
    )
    
    print(f"\n[{name}] Cấu hình:")
    print(f"  • Hidden layers: {config['hidden_layers']}")
    print(f"  • Activation: {config['activation']}")
    print(f"  • Learning rate: {config['learning_rate']} (init={config['learning_rate_init']})")
    print(f"  • Alpha (L2): {config['alpha']}")
    print(f"  • Batch size: {config['batch_size']}")
    
    model.fit(X_train_scaled, y_train)
    
    print(f"  ✓ Hoàn thành sau {model.n_iter_} iterations")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Đánh giá mô hình"""
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

df = sns.load_dataset('titanic')
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone', 'survived']]

# Xử lý missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Mã hóa categorical
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['embarked'] = LabelEncoder().fit_transform(df['embarked'])
df['alone'] = df['alone'].astype(int)

# 3. Chia dữ liệu
X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
ann_model = train_model(ANN_CONFIG, "ANN (Shallow)")
deep_model = train_model(DEEP_CONFIG, "Deep Learning")
ann_results = evaluate_model(ann_model, X_train_scaled, X_test_scaled, y_train, y_test)
deep_results = evaluate_model(deep_model, X_train_scaled, X_test_scaled, y_train, y_test)

# 6. Kết quả
print("\n" + "="*60)
print("KẾT QUẢ SO SÁNH")
print("="*60)

print("\n┌─────────────────────┬──────────────┬──────────────┐")
print("│ Metric              │ ANN (Shallow)│ Deep Learning│")
print("├─────────────────────┼──────────────┼──────────────┤")
print(f"│ Train Accuracy      │    {ann_results['train_acc']*100:5.2f}%    │    {deep_results['train_acc']*100:5.2f}%    │")
print(f"│ Test Accuracy       │    {ann_results['test_acc']*100:5.2f}%    │    {deep_results['test_acc']*100:5.2f}%    │")
print(f"│ Precision           │    {ann_results['precision']:6.4f}    │    {deep_results['precision']:6.4f}    │")
print(f"│ Recall              │    {ann_results['recall']:6.4f}    │    {deep_results['recall']:6.4f}    │")
print(f"│ F1-Score            │    {ann_results['f1']:6.4f}    │    {deep_results['f1']:6.4f}    │")
print("└─────────────────────┴──────────────┴──────────────┘")

# Số parameters
ann_params = sum([layer.size for layer in ann_model.coefs_])
deep_params = sum([layer.size for layer in deep_model.coefs_])

print(f"\nSố lượng parameters:")
print(f"  • ANN:           {ann_params:,} parameters")
print(f"  • Deep Learning: {deep_params:,} parameters")
print(f"  • Chênh lệch:    {deep_params - ann_params:,} parameters")

acc_diff = deep_results['test_acc'] - ann_results['test_acc']
if acc_diff > 0:
    print(f"\n✓ Deep Learning tốt hơn ANN: +{acc_diff*100:.2f}% accuracy")
else:
    print(f"\n✓ ANN tốt hơn Deep Learning: +{abs(acc_diff)*100:.2f}% accuracy")

if abs(acc_diff) < 0.02:
    print("✓ Khuyến nghị: Dùng ANN (đơn giản hơn, hiệu quả tương đương)")
elif deep_results['test_acc'] > ann_results['test_acc']:
    print("✓ Khuyến nghị: Dùng Deep Learning (accuracy cao hơn đáng kể)")
else:
    print("✓ Khuyến nghị: Dùng ANN (accuracy cao hơn)")