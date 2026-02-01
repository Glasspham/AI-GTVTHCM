# model_selector.py

def select_model(profile):
    """
    Chọn model phù hợp cho user
    """

    t = profile["type"]

    # 1. Chưa đăng nhập
    if t == "ANONYMOUS":
        return {
            "model": "POPULAR",
            "reason": "User not logged in"
        }

    # 2. User mới
    if t == "NEW":
        return {
            "model": "SESSION",
            "reason": "New user – use session/item based"
        }

    # 3. Ít dữ liệu
    if t == "LOW_DATA":
        return {
            "model": "ITEM_CF",
            "reason": "Few interactions – ItemCF works best"
        }

    # 4. Trung bình
    if t == "MEDIUM_DATA":
        return {
            "model": "USER_CF",
            "reason": "Enough history – use UserCF"
        }

    # 5. Nhiều dữ liệu
    if t == "RICH_DATA":
        return {
            "model": "MF",
            "reason": "Rich history – Matrix Factorization"
        }

    # 6. Nhiều điểm âm
    if t == "NEGATIVE_USER":
        return {
            "model": "ITEM_CF",
            "reason": "User dislikes many items – safer with ItemCF"
        }

    # Mặc định
    return {
        "model": "ITEM_CF",
        "reason": "Default fallback"
    }
