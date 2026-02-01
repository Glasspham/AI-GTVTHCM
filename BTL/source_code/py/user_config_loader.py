# user_config_loader.py
from db_connection import fetch_data

def load_user_behavior_config():
    query = """
        SELECT TOP 1
            n_low_max,
            n_medium_max,
            weight_view,
            weight_addtocart,
            weight_purchase,
            weight_rating_1,
            weight_rating_2,
            weight_rating_3,
            weight_rating_4,
            weight_rating_5
        FROM UserBehaviorWeights
        ORDER BY created_at DESC
    """

    df = fetch_data(query)

    if df is None or df.empty:
        raise Exception("❌ Không tìm thấy config trong UserBehaviorWeights")

    row = df.iloc[0]

    return {
        "n_low_max": int(row["n_low_max"]),
        "n_medium_max": int(row["n_medium_max"]),

        "weights": {
            "view": int(row["weight_view"]),
            "addtocart": int(row["weight_addtocart"]),
            "purchase": int(row["weight_purchase"]),
            "rating": {
                1: int(row["weight_rating_1"]),
                2: int(row["weight_rating_2"]),
                3: int(row["weight_rating_3"]),
                4: int(row["weight_rating_4"]),
                5: int(row["weight_rating_5"]),
            }
        }
    }
