# main.py
from fastapi import FastAPI, HTTPException
import random
import numpy as np
import hashlib
import pandas as pd

# ===== Imports =====
import load_data
from load_data import load_interactions
from evaluate import run_evaluation

import user_cf
import item_cf
import matrix_factorization as mf
from user_config_loader import load_user_behavior_config

from recommender import recommend_for_user

# ==================== 0) Fix seed ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==================== 1) Utils ====================

def get_book_name(book_raw):
    if book_raw is None:
        return "Unknown"

    book = str(book_raw)
    if not book.startswith("B"):
        book = f"B{book}"

    name = load_data.BOOK_MAP.get(book)
    if name is None:
        print(f"[WARN] Book not found in MAP: {book}")

    return name or "Unknown"


def to_book_id(book_raw):
    try:
        return int(str(book_raw).replace("B", ""))
    except Exception:
        return -1


# ==================== 2) Load model config from DB ====================

def load_model_config(model_name: str):
    """
    Load config từ DB:
    - RecommenderModel
    - CF_Model_Params (nếu là user_cf / item_cf)
    """

    df = load_data.load_model_config_from_db(model_name)

    if df is None or df.empty:
        raise HTTPException(
            status_code=500,
            detail=f"Config not found for model {model_name}"
        )

    row = df.iloc[0]

    config = {
        "model_name": row["model_name"],
        "top_n": int(row["top_n"])
    }

    if model_name in ("user_cf", "item_cf"):
        config["k"] = int(row["k"])
        config["alpha"] = float(row["alpha"])

    return config


# ==================== 3) Model Manager ====================

class ModelManager:
    def __init__(self):
        self.current_hash = None
        self.eval_result = None
        self.best_model_name = None
        self.best_model = None

    def compute_hash(self, df: pd.DataFrame):
        return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()

    def reload_if_needed(self):
        df = load_interactions()

        if df is None:
            raise HTTPException(
                status_code=500,
                detail="Cannot load interactions from database"
            )

        new_hash = self.compute_hash(df)

        if new_hash != self.current_hash:
            print(">>> Data changed → re-evaluating models")
            self.current_hash = new_hash

            self.eval_result = run_evaluation()
            self.best_model_name = self.eval_result["best_model"]

            self.best_model = {
                "user_cf": user_cf,
                "item_cf": item_cf,
                "mf": mf
            }[self.best_model_name]

        else:
            print(">>> Using cached model")

        return self.best_model_name, self.best_model

    def get_metrics(self):
        return self.eval_result


# ==================== 4) INIT APP ====================

print(">>> Loading BOOK_MAP at startup...")
load_interactions()
print(">>> BOOK_MAP SIZE:", len(load_data.BOOK_MAP))

manager = ModelManager()
manager.reload_if_needed()

app = FastAPI(title="Recommendation API")


# ==================== 5) Response builder ====================

def build_response(model_name, user_id, recs):
    return {
        "model": model_name,
        "user": user_id,
        "recommendations": [
            {
                "bookId": to_book_id(book),
                "bookName": get_book_name(book),
                "score": float(score)
            }
            for book, score in recs
        ]
    }


# ==================== 6) Endpoints ====================

@app.get("/evaluate")
def evaluate():
    manager.reload_if_needed()
    return manager.get_metrics()


@app.get("/recommend/best/{user_id}")
def recommend_best(user_id: str):
    model_name, model = manager.reload_if_needed()
    config = load_model_config(model_name)

    if model_name == "mf":
        recs = model.recommend_top_n(
            user_id,
            top_n=config["top_n"]
        )
    else:
        recs = model.recommend_top_n(
            user_id,
            top_n=config["top_n"],
            k=config["k"],
            alpha=config["alpha"]
        )

    return build_response(model_name, user_id, recs)


@app.get("/recommend/usercf/{user_id}")
def recommend_usercf(user_id: str):
    manager.reload_if_needed()
    config = load_model_config("user_cf")

    recs = user_cf.recommend_top_n(
        user_id,
        top_n=config["top_n"],
        k=config["k"],
        alpha=config["alpha"]
    )

    return build_response("user_cf", user_id, recs)


@app.get("/recommend/itemcf/{user_id}")
def recommend_itemcf(user_id: str):
    manager.reload_if_needed()
    config = load_model_config("item_cf")

    recs = item_cf.recommend_top_n(
        user_id,
        top_n=config["top_n"],
        k=config["k"],
        alpha=config["alpha"]
    )

    return build_response("item_cf", user_id, recs)


@app.get("/recommend/mf/{user_id}")
def recommend_mf(user_id: str):
    manager.reload_if_needed()
    config = load_model_config("mf")

    recs = mf.recommend_top_n(
        user_id,
        top_n=config["top_n"]
    )

    return build_response("mf", user_id, recs)


# ==================== 7) Personalized / fallback ====================

@app.get("/recommend/personal/{user_id}")
def recommend_personal(user_id: str, top_n: int = 10):
    uid = None if user_id == "anonymous" else user_id

    # 1. Gợi ý
    result = recommend_for_user(uid, top_k=top_n)

    # 2. Load dữ liệu interaction
    df = load_interactions()
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="No interaction data")

    # 3. Reload model + config
    model_name, _ = manager.reload_if_needed()
    config = load_model_config(model_name)

    # 4. Thống kê user
    user_df = df[df["UserID"] == user_id]

    rated_item_ids = user_df[user_df["Rating"].notna()]["BookID"].tolist()
    implicit_item_ids = user_df[user_df["Score"].notna()]["BookID"].tolist()

    rated_items = [get_book_name(book_id) for book_id in rated_item_ids]
    implicit_items = [get_book_name(book_id) for book_id in implicit_item_ids]

    user_statistics = {
        "user_exists": not user_df.empty,
        "num_user_ratings": int(user_df["Rating"].notna().sum()),
        "num_user_scores": int(user_df["Score"].notna().sum()),
        "rated_items": rated_items,
        "implicit_items": implicit_items
    }

    # 5. Tham số thuật toán
    algorithm_parameters = {
        "model": model_name,
        "top_n": config["top_n"]
    }

    if model_name in ("user_cf", "item_cf"):
        algorithm_parameters["k"] = config["k"]
        algorithm_parameters["alpha"] = config["alpha"]

    # 6. Response cuối
    return {
        "strategy": result["strategy"],
        "reason": result["reason"],
        "profile": result["profile"],

        "algorithm_parameters": algorithm_parameters,
        "user_statistics": user_statistics,

        "recommendations": [
            {
                "bookId": to_book_id(book),
                "bookName": get_book_name(book),
                "score": float(score)
            }
            for book, score in result["items"]
        ]
    }



@app.get("/recommend/anonymous")
def recommend_anonymous(top_n: int = 10):
    result = recommend_for_user(None, top_k=top_n)

    return {
        "strategy": "POPULAR",
        "recommendations": [
            {
                "bookId": to_book_id(book),
                "bookName": get_book_name(book),
                "score": float(score)
            }
            for book, score in result["items"]
        ]
    }


@app.get("/debug/decision/{user_id}")
def debug_decision(user_id: str):
    uid = None if user_id == "anonymous" else user_id
    result = recommend_for_user(uid, top_k=5)

    return {
        "profile": result["profile"],
        "chosen_strategy": result["strategy"],
        "reason": result["reason"]
    }


@app.get("/debug/books")
def debug_books():
    return {
        "map_size": len(load_data.BOOK_MAP),
        "sample": list(load_data.BOOK_MAP.items())[:20]
    }

@app.get("/recommender/config/{user_id}")
def debug_model_input(user_id: str):
    df = load_interactions()

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="No interaction data")

    # ===== Load config từ DB =====
    config = load_user_behavior_config()

    data_statistics = {
        "total_interactions": int(len(df)),
        "num_users": int(df["UserID"].nunique()),
        "num_items": int(df["BookID"].nunique()),
        "num_ratings": int(df["Rating"].notna().sum()),
        "num_scores": int(df["Score"].notna().sum())
    }

    return {
        "data_statistics": data_statistics,

        "user_thresholds": {
            "n_low_max": config["n_low_max"],
            "n_medium_max": config["n_medium_max"]
        },

        "behavior_weights": {
            "view": config["weights"]["view"],
            "addtocart": config["weights"]["addtocart"],
            "purchase": config["weights"]["purchase"],
            "rating": config["weights"]["rating"]
        }
    }
@app.get("/users")
def get_users():
    df = load_interactions()

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="No interaction data")

    users = []

    grouped = df.groupby("UserID")

    for user_id, g in grouped:
        users.append({
            "userId": str(user_id),
            "totalInteractions": int(len(g)),
            "numRatings": int(g["Rating"].notna().sum()),
            "numScores": int(g["Score"].notna().sum())
        })

    # Sắp xếp user có data nhiều lên trước (admin-friendly)
    users.sort(key=lambda x: x["totalInteractions"], reverse=True)

    return users

# ==================== RUN ====================
# uvicorn main:app --reload
