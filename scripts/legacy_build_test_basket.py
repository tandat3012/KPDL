import pandas as pd
from pathlib import Path

DATA_DIR = Path("processed/cleaned_csv")
OUT_DIR = Path("processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
transactions = pd.read_csv(DATA_DIR / "clean_transactions.csv")

# Gom basket theo customer_id + ngày mua
baskets = (
    transactions
    .groupby(["customer_id", "t_dat"])["article_id"]
    .apply(lambda items: list(set(items)))
    .reset_index()
)

# giữ basket có từ 2 sản phẩm trở lên
baskets = baskets[baskets["article_id"].apply(len) >= 2]

baskets = baskets.rename(columns={"article_id": "items"})

baskets["basket_id"] = range(1, len(baskets) + 1)

baskets = baskets[["basket_id", "customer_id", "t_dat", "items"]]

baskets.to_csv(
    OUT_DIR / "baskets_by_customer_date.csv",
    index=False,
)

print("Basket dataset created!")
print(OUT_DIR / "baskets_by_customer_date.csv")