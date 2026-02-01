# load_data.py
import pandas as pd
import pyodbc
from db_connection import fetch_data

# ===============================
# Global: BookID -> BookName
# ===============================
BOOK_MAP = {}   # { "B1": "Tên sách 1", ... }


# ===============================
# 1️⃣ Load User Interactions
# ===============================
def load_interactions():
    """
    Load dữ liệu tương tác user–book
    Bao gồm:
    - Rating (explicit)
    - Score  (implicit)
    """

    query = """
        SELECT
            ui.UserId,
            ui.BookId,
            ui.Rating,
            ui.Score,
            b.BookName
        FROM UserInteractions ui
        JOIN Book b ON ui.BookId = b.Id
    """

    df = fetch_data(query)

    if df is None or df.empty:
        print("❌ Không có dữ liệu trong UserInteractions")
        return None

    # Chuẩn hoá tên cột
    df = df.rename(columns={
        "UserId": "UserID",
        "BookId": "BookID",
        "BookName": "BookName"
    })

    # BookID -> dạng Bxx
    df["BookID"] = df["BookID"].apply(lambda x: f"B{x}")

    # Fill NaN cho Rating / Score (để tránh crash khi train)
    if "Rating" in df.columns:
        df["Rating"] = df["Rating"].astype(float)

    if "Score" in df.columns:
        df["Score"] = df["Score"].astype(float)

    # Build BOOK_MAP
    global BOOK_MAP
    BOOK_MAP = dict(zip(df["BookID"], df["BookName"]))

    print(f"✅ Loaded interactions: {len(df)} rows")
    print(f"📚 BOOK_MAP size: {len(BOOK_MAP)}")

    return df


# ===============================
# 2️⃣ Load Model Config from DB
# ===============================
def load_model_config_from_db(model_name: str):
    """
    Load config cho recommender model
    Dùng cho:
    - user_cf
    - item_cf
    - mf
    """

    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=MINHKHOI\\SQLEXPRESS;"
        "DATABASE=BookShoppingCartMvcMau;"
        "Trusted_Connection=yes;"
    )

    query = """
        SELECT
            rm.model_id,
            rm.model_name,
            rm.top_n,
            rm.is_active,
            p.k,
            p.alpha
        FROM RecommenderModel rm
        LEFT JOIN CF_Model_Params p
            ON rm.model_id = p.model_id
        WHERE rm.model_name = ?
          AND rm.is_active = 1
    """

    df = pd.read_sql(query, conn, params=[model_name])
    conn.close()

    if df.empty:
        print(f"❌ Không tìm thấy config cho model: {model_name}")
        return None

    return df

def load_mf_config_from_db():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=MINHKHOI\\SQLEXPRESS;"
        "DATABASE=BookShoppingCartMvcMau;"
        "Trusted_Connection=yes;"
    )

    query = """
        SELECT
            rm.top_n,
            p.latent_k,
            p.learning_rate,
            p.reg_lambda,
            p.n_iter,
            p.weight_rating,
            p.weight_score,
            p.pred_min,
            p.pred_max
        FROM RecommenderModel rm
        JOIN MF_Model_Params p
            ON rm.model_id = p.model_id
        WHERE rm.model_name = 'mf'
          AND rm.is_active = 1
    """

    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("❌ Không tìm thấy config cho MF")
        return None

    return df

# ===============================
# 3️⃣ Debug helpers (optional)
# ===============================
def debug_interaction_stats():
    df = load_interactions()
    if df is None:
        return None

    return {
        "total_interactions": len(df),
        "num_users": int(df["UserID"].nunique()),
        "num_items": int(df["BookID"].nunique()),
        "num_ratings": int(df["Rating"].notna().sum()),
        "num_scores": int(df["Score"].notna().sum())
    }


# ===============================
# Local test
# ===============================
if __name__ == "__main__":
    df = load_interactions()
    print(df.head())

    cfg = load_model_config_from_db("user_cf")
    print(cfg)
