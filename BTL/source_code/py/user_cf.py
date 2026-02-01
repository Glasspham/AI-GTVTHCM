"""
User-based Collaborative Filtering (Dual model)

- Explicit feedback: Rating  -> Pearson similarity (user-user)
- Implicit feedback: Score   -> Cosine similarity (user-user)
- Ensemble: alpha * explicit + (1 - alpha) * implicit

✔ DB-driven
✔ Không giả định thang điểm
✔ Normalize theo dữ liệu train
✔ Dùng cho evaluate.py / FastAPI service
"""

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True, precision=6)

# ======================================================
# GLOBAL STATE
# ======================================================
rating_matrix = None
implicit_matrix = None

rating_sims = None
implicit_sims = None

RATING_MIN = None
RATING_MAX = None
IMPLICIT_MIN = None
IMPLICIT_MAX = None


# ======================================================
# NORMALIZATION (DB-DRIVEN)
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
# SIMILARITY FUNCTIONS
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
# BUILD MODEL
# ======================================================
def build_model(train_df: pd.DataFrame):
    global rating_matrix, implicit_matrix
    global rating_sims, implicit_sims
    global RATING_MIN, RATING_MAX, IMPLICIT_MIN, IMPLICIT_MAX

    rating_df = train_df.dropna(subset=["Rating"]).copy()
    implicit_df = train_df.dropna(subset=["Score"]).copy()

    # normalize range (DB-driven)
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

    # similarity matrices
    rating_sims = pd.DataFrame(
        index=rating_matrix.index,
        columns=rating_matrix.index,
        dtype=float
    )

    for u1 in rating_matrix.index:
        for u2 in rating_matrix.index:
            rating_sims.loc[u1, u2] = pearson_similarity_rows(
                rating_matrix.loc[u1],
                rating_matrix.loc[u2]
            )

    implicit_sims = cosine_similarity_full(implicit_matrix)


# ======================================================
# PREDICTION FUNCTIONS
# ======================================================
def predict_rating(user: str, item: str, k: int) -> float | None:
    if user not in rating_matrix.index:
        return None

    if not pd.isna(rating_matrix.loc[user, item]):
        return float(rating_matrix.loc[user, item])

    user_ratings = rating_matrix.loc[user].dropna()
    if user_ratings.empty:
        return None

    user_mean = float(user_ratings.mean())
    neighbors = []

    for other in rating_matrix.index:
        if other == user:
            continue
        if pd.isna(rating_matrix.loc[other, item]):
            continue

        sim = float(rating_sims.loc[user, other])
        if sim == 0:
            continue

        other_mean = rating_matrix.loc[other].dropna().mean()
        neighbors.append((sim, rating_matrix.loc[other, item] - other_mean))

    if not neighbors:
        return user_mean

    neighbors = sorted(neighbors, key=lambda x: abs(x[0]), reverse=True)[:k]

    num = sum(sim * diff for sim, diff in neighbors)
    den = sum(abs(sim) for sim, _ in neighbors)

    return user_mean + (num / den if den != 0 else 0.0)


def predict_implicit(user: str, item: str, k: int) -> float | None:
    if user not in implicit_matrix.index:
        return None

    if not pd.isna(implicit_matrix.loc[user, item]):
        return float(implicit_matrix.loc[user, item])

    neighbors = []
    for other in implicit_matrix.index:
        if other == user:
            continue
        if pd.isna(implicit_matrix.loc[other, item]):
            continue

        sim = float(implicit_sims.loc[user, other])
        if sim == 0:
            continue

        neighbors.append((sim, implicit_matrix.loc[other, item]))

    if not neighbors:
        return None

    neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)[:k]

    num = sum(sim * s for sim, s in neighbors)
    den = sum(abs(sim) for sim, _ in neighbors)

    return num / den if den != 0 else None


# ======================================================
# ENSEMBLE
# ======================================================
def ensemble_predict(user: str, item: str, alpha: float, k: int):
    r = normalize_rating(predict_rating(user, item, k))
    i = normalize_implicit(predict_implicit(user, item, k))

    if r is not None and i is not None:
        return alpha * r + (1 - alpha) * i
    return r if r is not None else i


# ======================================================
# RECOMMEND TOP-N
# ======================================================
def recommend_top_n(user: str, top_n: int, k: int, alpha: float):
    if user not in rating_matrix.index:
        return []

    results = {}
    for item in rating_matrix.columns:
        # chỉ exclude item đã có explicit rating
        if not pd.isna(rating_matrix.loc[user, item]):
            continue

        score = ensemble_predict(user, item, alpha, k)
        if score is not None:
            results[item] = score

    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_n]
