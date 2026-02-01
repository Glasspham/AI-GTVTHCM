"""
======================================================
FLOW HOẠT ĐỘNG CỦA HỆ THỐNG GỢI Ý (ITEM-BASED CF – DUAL MODEL)
======================================================

Flow 1️⃣: DB → train_df
- Truy xuất dữ liệu lịch sử từ database
- Bao gồm: UserID, BookID, Rating (explicit), Score (implicit)
- Mục tiêu: tạo tập dữ liệu đầu vào cho mô hình gợi ý dựa trên item

Flow 2️⃣: Xác định min / max từ tập train (DB-driven)
- Lấy giá trị nhỏ nhất và lớn nhất của Rating và Score trực tiếp từ train_df
- Không giả định trước thang điểm (1–5, 1–10, …)
- Mục tiêu: chuẩn bị cho bước chuẩn hoá dữ liệu về [0,1]

Flow 3️⃣: Xây dựng ma trận User × Item và Item × User
- Tạo:
  + rating_matrix (User × Item)
  + implicit_matrix (User × Item)
- Chuyển vị để thu được:
  + rating_item_matrix (Item × User)
  + implicit_item_matrix (Item × User)
- Mục tiêu: biểu diễn item dưới dạng vector hành vi của user

Flow 4️⃣: Tính ma trận độ tương đồng item-item
- Explicit feedback (Rating) → Pearson similarity (item-item)
- Implicit feedback (Score)  → Cosine similarity (item-item)
- Mục tiêu: xác định các item có hành vi tương tự nhau

Flow 5️⃣: Dự đoán điểm cho cặp (user, item)
- Dự đoán mức độ yêu thích của user với item chưa từng tương tác
- Dựa trên:
  + Các item tương tự mà user đã tương tác
  + Trọng số theo độ tương đồng item-item
- Mục tiêu: ước lượng mức độ quan tâm tiềm năng

Flow 6️⃣: Chuẩn hoá kết quả dự đoán
- Chuẩn hoá rating prediction và implicit prediction về cùng thang [0,1]
- Mục tiêu: loại bỏ khác biệt đơn vị giữa hai loại feedback

Flow 7️⃣: Ensemble & Top-N Recommendation
- Kết hợp explicit và implicit theo công thức:
  final_score = α ⋅ explicit + (1 − α) ⋅ implicit
- Sắp xếp giảm dần theo final_score
- Loại bỏ item user đã rating
- Lấy ra Top-N item tốt nhất để gợi ý cho người dùng

======================================================
Item-based Collaborative Filtering (Dual Model)

✔ DB-driven
✔ Không giả định thang điểm
✔ Chuẩn hoá theo tập train
✔ Dùng cho evaluate.py / FastAPI service
======================================================
"""

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True, precision=6)

# ======================================================
# 1. BIẾN TOÀN CỤC (GLOBAL STATE)
# ======================================================

rating_matrix = None
implicit_matrix = None

rating_item_matrix = None
implicit_item_matrix = None

rating_sims = None
implicit_sims = None

RATING_MIN = None
RATING_MAX = None
IMPLICIT_MIN = None
IMPLICIT_MAX = None


# ======================================================
# 2. CHUẨN HOÁ DỮ LIỆU (DB-DRIVEN NORMALIZATION)
# ======================================================

def normalize_rating(r):
    if r is None:
        return None
    if RATING_MAX == RATING_MIN:
        return 0.0
    return (r - RATING_MIN) / (RATING_MAX - RATING_MIN)


def normalize_implicit(i):
    if i is None:
        return None
    if IMPLICIT_MAX == IMPLICIT_MIN:
        return 0.0
    return (i - IMPLICIT_MIN) / (IMPLICIT_MAX - IMPLICIT_MIN)


# ======================================================
# 3. HÀM TÍNH ĐỘ TƯƠNG ĐỒNG
# ======================================================

def pearson_similarity_rows(row1: pd.Series, row2: pd.Series) -> float:
    mask = ~row1.isna() & ~row2.isna()
    if mask.sum() < 2:
        return 0.0

    m1 = row1.dropna().mean()
    m2 = row2.dropna().mean()

    x = row1[mask].to_numpy(dtype=float) - m1
    y = row2[mask].to_numpy(dtype=float) - m2

    num = np.sum(x * y)
    den = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))

    return 0.0 if den == 0 else float(num / den)


def cosine_similarity_full(mat: pd.DataFrame) -> pd.DataFrame:
    M = mat.fillna(0).to_numpy(dtype=float)
    norms = np.linalg.norm(M, axis=1)

    sim = np.zeros((M.shape[0], M.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            if norms[i] == 0 or norms[j] == 0:
                sim[i, j] = 0.0
            else:
                sim[i, j] = np.dot(M[i], M[j]) / (norms[i] * norms[j])

    return pd.DataFrame(sim, index=mat.index, columns=mat.index)


# ======================================================
# 4. BUILD MODEL
# ======================================================

def build_model(train_df: pd.DataFrame):
    global rating_matrix, implicit_matrix
    global rating_item_matrix, implicit_item_matrix
    global rating_sims, implicit_sims
    global RATING_MIN, RATING_MAX, IMPLICIT_MIN, IMPLICIT_MAX

    rating_df = train_df.dropna(subset=["Rating"]).copy()
    implicit_df = train_df.dropna(subset=["Score"]).copy()

    RATING_MIN = rating_df["Rating"].min()
    RATING_MAX = rating_df["Rating"].max()
    IMPLICIT_MIN = implicit_df["Score"].min()
    IMPLICIT_MAX = implicit_df["Score"].max()

    all_users = sorted(train_df["UserID"].unique())
    all_items = sorted(train_df["BookID"].unique())

    rating_matrix = (
        rating_df
        .pivot(index="UserID", columns="BookID", values="Rating")
        .reindex(index=all_users, columns=all_items)
    )

    implicit_matrix = (
        implicit_df
        .pivot(index="UserID", columns="BookID", values="Score")
        .reindex(index=all_users, columns=all_items)
    )

    rating_item_matrix = rating_matrix.T
    implicit_item_matrix = implicit_matrix.T

    rating_sims = pd.DataFrame(
        index=rating_item_matrix.index,
        columns=rating_item_matrix.index,
        dtype=float
    )

    for i1 in rating_item_matrix.index:
        for i2 in rating_item_matrix.index:
            rating_sims.loc[i1, i2] = pearson_similarity_rows(
                rating_item_matrix.loc[i1],
                rating_item_matrix.loc[i2]
            )

    implicit_sims = cosine_similarity_full(implicit_item_matrix)


# ======================================================
# 5. DỰ ĐOÁN
# ======================================================

def predict_rating(user: str, item: str, k: int) -> float | None:
    if user not in rating_matrix.index:
        return None

    if not pd.isna(rating_matrix.loc[user, item]):
        return float(rating_matrix.loc[user, item])

    user_ratings = rating_matrix.loc[user].dropna()
    if user_ratings.empty:
        return None

    neighbors = []
    for j in rating_matrix.columns:
        if j == item or pd.isna(rating_matrix.loc[user, j]):
            continue

        sim = float(rating_sims.loc[item, j])
        if sim == 0:
            continue

        neighbors.append((sim, rating_matrix.loc[user, j]))

    if not neighbors:
        return float(user_ratings.mean())

    neighbors = sorted(neighbors, key=lambda x: abs(x[0]), reverse=True)[:k]

    num = sum(sim * r for sim, r in neighbors)
    den = sum(abs(sim) for sim, _ in neighbors)

    return num / den if den != 0 else float(user_ratings.mean())


def predict_implicit(user: str, item: str, k: int) -> float | None:
    if user not in implicit_matrix.index:
        return None

    if not pd.isna(implicit_matrix.loc[user, item]):
        return float(implicit_matrix.loc[user, item])

    neighbors = []
    for j in implicit_matrix.columns:
        if j == item or pd.isna(implicit_matrix.loc[user, j]):
            continue

        sim = float(implicit_sims.loc[item, j])
        if sim == 0:
            continue

        neighbors.append((sim, implicit_matrix.loc[user, j]))

    if not neighbors:
        return None

    neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)[:k]

    num = sum(sim * s for sim, s in neighbors)
    den = sum(abs(sim) for sim, _ in neighbors)

    return num / den if den != 0 else None


# ======================================================
# 6. ENSEMBLE
# ======================================================

def ensemble_predict(user: str, item: str, alpha: float, k: int):
    r = normalize_rating(predict_rating(user, item, k))
    i = normalize_implicit(predict_implicit(user, item, k))

    if r is not None and i is not None:
        return alpha * r + (1 - alpha) * i
    return r if r is not None else i


# ======================================================
# 7. TOP-N RECOMMENDATION
# ======================================================

def recommend_top_n(user: str, top_n: int, k: int, alpha: float):
    if user not in rating_matrix.index:
        return []

    results = {}
    for item in rating_matrix.columns:
        if not pd.isna(rating_matrix.loc[user, item]):
            continue

        score = ensemble_predict(user, item, alpha, k)
        if score is not None:
            results[item] = score

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_n]
