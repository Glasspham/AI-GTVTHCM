# evaluate.py
"""
======================================================
FLOW ĐÁNH GIÁ HỆ THỐNG GỢI Ý (EVALUATION) – 1 → 7
======================================================

[1] Load dữ liệu tương tác & chia Train / Test (per-user)
    - Load toàn bộ interaction từ Database
    - Chia train/test theo từng UserID (user-based split)
    - Đảm bảo mỗi user có dữ liệu trong cả train và test
    → Mô phỏng đúng kịch bản recommendation thực tế

[2] Lọc user & item hợp lệ để đánh giá
    - Chỉ giữ user có đủ lịch sử (min_history)
    - Chỉ đánh giá trên item đã xuất hiện trong tập train
    → Tránh cold-start làm sai lệch kết quả

[3] Huấn luyện từng mô hình bằng dữ liệu train
    - User-based CF: build similarity user–user
    - Item-based CF: build similarity item–item
    - Matrix Factorization: train bằng SGD (DB-driven)
    → Mỗi model được train độc lập, cùng một tập dữ liệu

[4] Sinh danh sách gợi ý Top-K cho từng user
    - Với mỗi user trong test:
        + Gọi recommend_top_n() của model tương ứng
        + Không gợi ý item user đã tương tác trong train
    - K (Top-N) lấy từ config trong Database
    → Đánh giá thuần ranking, không quan tâm rating value

[5] So sánh gợi ý với ground-truth (test set)
    - Relevant items = các item user thực sự tương tác trong test
    - So sánh Top-K recommend với relevant items
    → Xác định đúng/sai theo nghĩa “có gợi ý trúng hay không”

[6] Tính các metric Ranking chuẩn (Paper-style)
    - Precision@K: độ chính xác của danh sách gợi ý
    - Recall@K: mức độ bao phủ item liên quan
    - NDCG@K: chất lượng thứ hạng (ưu tiên đúng item ở vị trí cao)
    - Lấy trung bình trên toàn bộ user hợp lệ
    → Phản ánh chất lượng recommendation tổng thể

[7] Tổng hợp FAIR SCORE & chọn model tốt nhất
    - FAIR SCORE = kết hợp:
        + 50% NDCG
        + 25% Precision
        + 25% Recall
    - So sánh FAIR SCORE giữa các model
    - Chọn model có điểm cao nhất
    → Đảm bảo đánh giá cân bằng, không thiên lệch 1 metric

======================================================

"""

import random
import numpy as np
import pandas as pd
from typing import Dict

from load_data import (
    load_interactions,
    load_model_config_from_db,
    load_mf_config_from_db
)

import user_cf
import item_cf
import matrix_factorization as mf


# ======================================================
# 1️⃣ Train / Test split (per-user)
# ======================================================
def train_test_split(df, test_ratio=0.2, seed=42):
    random.seed(seed)
    train, test = [], []

    for user, group in df.groupby("UserID"):
        rows = group.to_dict("records")

        if len(rows) < 2:
            train.extend(rows)
            continue

        n_test = max(1, int(len(rows) * test_ratio))
        idx = set(random.sample(range(len(rows)), n_test))

        for i, r in enumerate(rows):
            (test if i in idx else train).append(r)

    return pd.DataFrame(train), pd.DataFrame(test)


# ======================================================
# 2️⃣ Ranking metrics
# ======================================================
def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / k if k else 0.0


def recall_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / len(rel) if rel else 0.0


def ndcg_at_k(rec, rel, k):
    dcg = sum(
        1 / np.log2(i + 2)
        for i, it in enumerate(rec[:k])
        if it in rel
    )
    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(min(len(rel), k))
    )
    return dcg / idcg if idcg > 0 else 0.0


# ======================================================
# 3️⃣ Evaluate ONE model
# ======================================================
def evaluate_model(
    model,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg_row: pd.Series,
    min_history=2
):
    ps, rs, ns = [], [], []

    top_k = int(cfg_row.top_n)
    train_items = set(train_df["BookID"])

    eligible_users = (
        train_df.groupby("UserID")
        .size()
        .loc[lambda x: x >= min_history]
        .index
    )

    test_df = test_df[
        (test_df.UserID.isin(eligible_users)) &
        (test_df.BookID.isin(train_items))
    ]

    for u in test_df["UserID"].unique():
        relevant = test_df[test_df.UserID == u]["BookID"].tolist()
        if not relevant:
            continue

        # ---------- Recommend ----------
        if model_name in ("user_cf", "item_cf"):
            recs = model.recommend_top_n(
                user=u,
                top_n=top_k,
                k=int(cfg_row.k),
                alpha=float(cfg_row.alpha)
            )
        else:  # MF
            recs = model.recommend_top_n(u, top_k)

        if not recs:
            continue

        rec_items = [i for i, _ in recs]

        ps.append(precision_at_k(rec_items, relevant, top_k))
        rs.append(recall_at_k(rec_items, relevant, top_k))
        ns.append(ndcg_at_k(rec_items, relevant, top_k))

    return {
        "Precision@K": float(np.mean(ps)) if ps else 0.0,
        "Recall@K": float(np.mean(rs)) if rs else 0.0,
        "NDCG@K": float(np.mean(ns)) if ns else 0.0
    }


# ======================================================
# 4️⃣ FAIR SCORE
# ======================================================
def fair_score(m):
    return (
        0.5 * m["NDCG@K"] +
        0.25 * m["Precision@K"] +
        0.25 * m["Recall@K"]
    )


# ======================================================
# 5️⃣ MAIN ENTRY
# ======================================================
def run_evaluation() -> Dict:
    df = load_interactions()
    train_df, test_df = train_test_split(df)

    metrics = {}

    # ===============================
    # USER CF
    # ===============================
    user_cfg = load_model_config_from_db("user_cf")
    if user_cfg is not None:
        cfg = user_cfg.iloc[0]
        user_cf.build_model(train_df)

        metrics["user_cf"] = evaluate_model(
            user_cf, "user_cf", train_df, test_df, cfg
        )

    # ===============================
    # ITEM CF
    # ===============================
    item_cfg = load_model_config_from_db("item_cf")
    if item_cfg is not None:
        cfg = item_cfg.iloc[0]
        item_cf.build_model(train_df)

        metrics["item_cf"] = evaluate_model(
            item_cf, "item_cf", train_df, test_df, cfg
        )

    # ===============================
    # MATRIX FACTORIZATION (DB-driven)
    # ===============================
    mf_cfg = load_mf_config_from_db()
    if mf_cfg is not None:
        cfg = mf_cfg.iloc[0]

        mf.train(train_df)

        metrics["mf"] = evaluate_model(
            mf, "mf", train_df, test_df, cfg
        )

    # ===============================
    # FAIR SCORE
    # ===============================
    scores = {k: fair_score(v) for k, v in metrics.items()}
    best_model = max(scores, key=scores.get)

    return {
        "best_model": best_model,
        "metrics": metrics,
        "fair_scores": scores
    }


# ======================================================
# Local test
# ======================================================
if __name__ == "__main__":
    from pprint import pprint
    pprint(run_evaluation())
