"""
Thử nghiệm nhiều cấu hình siêu tham số và so sánh kết quả
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
# DANH SÁCH CÁC CẤU HÌNH THỬ NGHIỆM
# ============================================================

# Danh sách cấu hình cho ANN
ANN_CONFIGS = [
    {
        'name': 'ANN-1: Baseline',
        'hidden_layers': (64),
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
    },
    {
        'name': 'ANN-2: Tanh',
        'hidden_layers': (64),
        'activation': 'tanh',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
    },
    {
        'name': 'ANN-3: More neurons',
        'hidden_layers': (128),
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
    },
    {
        'name': 'ANN-4: Logistic',
        'hidden_layers': (64),
        'activation': 'logistic',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
    },
    {
        'name': 'ANN-5: High regularization',
        'hidden_layers': (32),
        'activation': 'tanh',
        'learning_rate_init': 0.001,
        'alpha': 0.01,
    },
]

# Danh sách cấu hình cho Deep Learning
DEEP_CONFIGS = [
    {
        'name': 'Deep-1: Baseline',
        'hidden_layers': (128, 64, 64, 32, 16),
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
    },
    {
        'name': 'Deep-2: Tanh',
        'hidden_layers': (128, 64, 64, 32, 16),
        'activation': 'tanh',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
    },
    {
        'name': 'Deep-3: Wider',
        'hidden_layers': (256, 128, 64, 32, 16),
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
    },
    {
        'name': 'Deep-4: Deeper',
        'hidden_layers': (128, 64, 64, 32, 16, 8),
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
    },
    {
        'name': 'Deep-5: High regularization',
        'hidden_layers': (128, 64, 64, 32, 16),
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'alpha': 0.01,
    },
]

# Cấu hình chung
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ITER = 500

# ============================================================
# HÀM HỖ TRỢ
# ============================================================

def train_and_evaluate(config, X_train, X_test, y_train, y_test):
    """Huấn luyện và đánh giá mô hình"""
    model = MLPClassifier(
        hidden_layer_sizes=config['hidden_layers'],
        activation=config['activation'],
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=config['learning_rate_init'],
        max_iter=MAX_ITER,
        alpha=config['alpha'],
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    # Đánh giá
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))
    
    # Số parameters
    n_params = sum([layer.size for layer in model.coefs_])
    
    return {
        'name': config['name'],
        'architecture': str(config['hidden_layers']),
        'activation': config['activation'],
        'alpha': config['alpha'],
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iterations': model.n_iter_,
        'parameters': n_params
    }

# ============================================================
# MAIN PROGRAM
# ============================================================

print("="*70)
print("THỬ NGHIỆM NHIỀU CẤU HÌNH SIÊU THAM SỐ - TITANIC DATASET")
print("="*70)

# 1. Load và tiền xử lý dữ liệu
print("\n[1] Đang tải và xử lý dữ liệu...")
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

# Chia dữ liệu
X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Đã tải {len(df)} mẫu (Train: {len(X_train)}, Test: {len(X_test)})")

# 2. Thử nghiệm các cấu hình ANN
print(f"\n[2] Thử nghiệm {len(ANN_CONFIGS)} cấu hình ANN...")
ann_results = []
for i, config in enumerate(ANN_CONFIGS, 1):
    print(f"  [{i}/{len(ANN_CONFIGS)}] {config['name']}...", end=' ')
    result = train_and_evaluate(config, X_train_scaled, X_test_scaled, y_train, y_test)
    ann_results.append(result)
    print(f"✓ Test Acc: {result['test_acc']*100:.2f}%")

# 3. Thử nghiệm các cấu hình Deep Learning
print(f"\n[3] Thử nghiệm {len(DEEP_CONFIGS)} cấu hình Deep Learning...")
deep_results = []
for i, config in enumerate(DEEP_CONFIGS, 1):
    print(f"  [{i}/{len(DEEP_CONFIGS)}] {config['name']}...", end=' ')
    result = train_and_evaluate(config, X_train_scaled, X_test_scaled, y_train, y_test)
    deep_results.append(result)
    print(f"✓ Test Acc: {result['test_acc']*100:.2f}%")

# 4. Tạo bảng thống kê
print("\n" + "="*70)
print("KẾT QUẢ THỬ NGHIỆM - ANN (SHALLOW NETWORK)")
print("="*70)

ann_df = pd.DataFrame(ann_results)
ann_df['train_acc'] = ann_df['train_acc'].apply(lambda x: f"{x*100:.2f}%")
ann_df['test_acc'] = ann_df['test_acc'].apply(lambda x: f"{x*100:.2f}%")
ann_df['precision'] = ann_df['precision'].apply(lambda x: f"{x:.4f}")
ann_df['recall'] = ann_df['recall'].apply(lambda x: f"{x:.4f}")
ann_df['f1'] = ann_df['f1'].apply(lambda x: f"{x:.4f}")

print("\n" + ann_df.to_string(index=False))

print("\n" + "="*70)
print("KẾT QUẢ THỬ NGHIỆM - DEEP LEARNING")
print("="*70)

deep_df = pd.DataFrame(deep_results)
deep_df['train_acc'] = deep_df['train_acc'].apply(lambda x: f"{x*100:.2f}%")
deep_df['test_acc'] = deep_df['test_acc'].apply(lambda x: f"{x*100:.2f}%")
deep_df['precision'] = deep_df['precision'].apply(lambda x: f"{x:.4f}")
deep_df['recall'] = deep_df['recall'].apply(lambda x: f"{x:.4f}")
deep_df['f1'] = deep_df['f1'].apply(lambda x: f"{x:.4f}")

print("\n" + deep_df.to_string(index=False))