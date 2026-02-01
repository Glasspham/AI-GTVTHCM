"""
matrix_factorization.py
Fully DB-driven Matrix Factorization (SGD)
"""

import numpy as np
import pandas as pd
from load_data import load_interactions, load_mf_config_from_db

np.set_printoptions(suppress=True, precision=6)

# ======================================================
# GLOBALS
# ======================================================
all_users: list[str] = []
all_items: list[str] = []
P: np.ndarray | None = None
Q: np.ndarray | None = None
rating_df: pd.DataFrame | None = None
cfg: pd.Series | None = None   # ⭐ PHẢI LÀ SERIES (1 ROW)


# ======================================================
# NORMALIZE PREDICTION (DB-driven)
# ======================================================
def normalize_pred(pred: float | None) -> float | None:
    if pred is None or cfg is None:
        return None

    pred_min = float(cfg.pred_min)
    pred_max = float(cfg.pred_max)

    pred = max(pred_min, min(pred_max, float(pred)))

    # scale về [0,1]
    if pred_max > pred_min:
        return (pred - pred_min) / (pred_max - pred_min)
    return None


# ======================================================
# TRAIN
# ======================================================
def train(train_df: pd.DataFrame | None = None):
    """
    Train MF model using FULLY DB-driven config
    """
    global all_users, all_items, P, Q, rating_df, cfg

    # ---------- Load MF config ----------
    cfg_df = load_mf_config_from_db()
    if cfg_df is None or cfg_df.empty:
        raise ValueError("❌ MF config not found in DB")

    cfg = cfg_df.iloc[0]   # ⭐ FIX QUAN TRỌNG NHẤT

    # ---------- Load interactions ----------
    if train_df is None:
        df = load_interactions()
    else:
        df = train_df.copy()

    if df.empty:
        raise ValueError("❌ Interaction data is empty")

    rating_df = df.dropna(subset=["Rating"]).copy()

    all_users = sorted(df["UserID"].unique())
    all_items = sorted(df["BookID"].unique())

    # ---------- Safe normalize ----------
    def safe_norm(s: pd.Series):
        if s.isna().all():
            return s.fillna(0)
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else s.fillna(0)

    df["Rating_norm"] = safe_norm(df["Rating"])
    df["Score_norm"] = safe_norm(df["Score"])

    # ---------- DB-driven ensemble ----------
    df["Combined"] = (
        float(cfg.weight_rating) * df["Rating_norm"].fillna(0)
        + float(cfg.weight_score) * df["Score_norm"].fillna(0)
    )

    # ---------- User–Item matrix ----------
    R_df = (
        df.pivot(index="UserID", columns="BookID", values="Combined")
          .reindex(index=all_users, columns=all_items)
          .fillna(0)
    )

    R = R_df.to_numpy()
    n_users, n_items = R.shape

    # ---------- Init latent factors ----------
    k = int(cfg.latent_k)
    P = np.random.normal(scale=1.0 / k, size=(n_users, k))
    Q = np.random.normal(scale=1.0 / k, size=(n_items, k))

    lr = float(cfg.learning_rate)
    lam = float(cfg.reg_lambda)
    n_iter = int(cfg.n_iter)

    # ---------- SGD ----------
    for _ in range(n_iter):
        for u in range(n_users):
            for i in range(n_items):
                if R[u, i] == 0:
                    continue
                err = R[u, i] - np.dot(P[u], Q[i])
                P[u] += lr * (err * Q[i] - lam * P[u])
                Q[i] += lr * (err * P[u] - lam * Q[i])

    return R_df


# ======================================================
# PREDICT SINGLE RATING
# ======================================================
def predict_rating(user: str, item: str) -> float | None:
    if P is None or Q is None or cfg is None:
        raise RuntimeError("❌ MF model not trained")

    if user not in all_users or item not in all_items:
        return None

    u = all_users.index(user)
    i = all_items.index(item)

    return normalize_pred(float(np.dot(P[u], Q[i])))


# ======================================================
# RECOMMEND TOP-N
# ======================================================
def recommend_top_n(user: str, top_n: int | None = None):
    if P is None or Q is None or cfg is None:
        raise RuntimeError("❌ MF model not trained")

    if user not in all_users:
        return []

    if top_n is None:
        top_n = int(cfg.top_n)

    u = all_users.index(user)
    scores = P[u] @ Q.T

    rated_items = set(
        rating_df[rating_df["UserID"] == user]["BookID"]
    )

    preds = [
        (it, normalize_pred(scores[i]))
        for i, it in enumerate(all_items)
        if it not in rated_items
    ]

    preds = [(it, p) for it, p in preds if p is not None]

    return sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
