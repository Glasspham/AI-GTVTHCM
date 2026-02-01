# recommender.py

from user_profiler import get_user_profile
from model_selector import select_model

import user_cf
import item_cf
import matrix_factorization as mf

from load_data import load_interactions


# ===== 1. Gợi ý phổ biến =====

def popular_recommend(k=10):
    """
    Cho user chưa đăng nhập
    Dựa trên tổng Score
    """

    df = load_interactions()

    if df.empty:
        return []

    top = (
        df.groupby("BookID")["Score"]
          .sum()
          .sort_values(ascending=False)
          .head(k)
    )

    return [(bid, float(score)) for bid, score in top.items()]


# ===== 2. Session-based =====

def session_based(df, top_k=10):
    """
    Fallback cho user mới (có ít data)
    """

    if df.empty:
        return []

    df = df.copy()

    # Score tổng hợp từ hành vi + rating
    df["final_score"] = df["Score"].fillna(0) + \
                        df["Rating"].fillna(0) * 2

    popular = (
        df.groupby("BookID")["final_score"]
          .mean()
          .reset_index()
          .sort_values("final_score", ascending=False)
    )

    return [
        (row.BookID, float(row.final_score))
        for _, row in popular.head(top_k).iterrows()
    ]


# ===== 3. HÀM CHÍNH =====

def recommend_for_user(user_id, top_k=10):
    """
    Entry chính cho ASP.NET gọi
    """

    # --- Load data ---
    df = load_interactions()

    # --- 1. Phân tích user ---
    profile = get_user_profile(df, user_id)

    # --- 2. Chọn model ---
    decision = select_model(profile)
    model = decision["model"]

    # --- 3. Sinh gợi ý ---
    if model == "POPULAR":
        items = popular_recommend(top_k)

    elif model == "SESSION":
        items = session_based(df, top_k)

    elif model == "ITEM_CF":
        items = item_cf.recommend_top_n(user_id, top_k)

    elif model == "USER_CF":
        items = user_cf.recommend_top_n(user_id, top_k, alpha=0.6)

    elif model == "MF":
        items = mf.recommend_top_n(user_id, top_k)

    else:
        items = popular_recommend(top_k)

    # --- 4. Response ---
    return {
        "user_id": user_id,
        "profile": profile,
        "strategy": model,
        "reason": decision["reason"],
        "items": items
    }
