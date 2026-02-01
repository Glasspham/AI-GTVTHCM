# user_profiler.py

import pandas as pd
from user_config_loader import load_user_behavior_config


def get_user_profile(df: pd.DataFrame, user_id):
    """
    Phân tích user dựa trên UserInteractions
    Ngưỡng phân loại lấy từ DB
    """

    config = load_user_behavior_config()
    n_low_max = config["n_low_max"]
    n_medium_max = config["n_medium_max"]

    # 1. User chưa đăng nhập
    if user_id is None:
        return {
            "type": "ANONYMOUS",
            "n_interaction": 0,
            "total_score": 0
        }

    # Ép kiểu string để so khớp an toàn
    user_df = df[df.UserID.astype(str) == str(user_id)]

    # 2. User mới hoàn toàn
    if len(user_df) == 0:
        return {
            "type": "NEW",
            "n_interaction": 0,
            "total_score": 0
        }

    # ===== Tính đặc trưng =====
    n = len(user_df)

    total_score = float(user_df["Score"].sum())
    avg_score = float(user_df["Score"].mean())

    has_rating = bool(user_df["Rating"].notna().any())

    n_positive = int((user_df["Score"] > 0).sum())
    n_negative = int((user_df["Score"] < 0).sum())

    # ===== Phân loại (KHÔNG hardcode) =====
    if n <= n_low_max:
        utype = "LOW_DATA"
    elif n <= n_medium_max:
        utype = "MEDIUM_DATA"
    else:
        utype = "RICH_DATA"

    # Trường hợp đặc biệt
    if n_negative > n_positive:
        utype = "NEGATIVE_USER"

    return {
        "type": utype,
        "n_interaction": n,
        "total_score": total_score,
        "avg_score": avg_score,
        "has_rating": has_rating,
        "n_positive": n_positive,
        "n_negative": n_negative
    }
