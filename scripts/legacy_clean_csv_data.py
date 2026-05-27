import pandas as pd
from pathlib import Path

RAW_DIR = Path("orginal_data_csv")
OUT_DIR = Path("processed/cleaned_csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1. Clean transactions
# =========================

transactions = pd.read_csv(RAW_DIR / "transactions_train.csv")

clean_transactions = transactions[
    [
        "t_dat",
        "customer_id",
        "article_id",
        "price",
        "sales_channel_id",
    ]
].copy()

clean_transactions["t_dat"] = pd.to_datetime(clean_transactions["t_dat"])

clean_transactions = clean_transactions.dropna(
    subset=["t_dat", "customer_id", "article_id"]
)

clean_transactions = clean_transactions.drop_duplicates()

clean_transactions.to_csv(
    OUT_DIR / "clean_transactions.csv",
    index=False,
)

# =========================
# 2. Clean articles
# =========================

articles = pd.read_csv(RAW_DIR / "articles.csv")

clean_articles = articles[
    [
        "article_id",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "department_name",
        "section_name",
        "garment_group_name",
        "detail_desc",
    ]
].copy()

clean_articles = clean_articles.dropna(subset=["article_id"])
clean_articles = clean_articles.drop_duplicates(subset=["article_id"])

clean_articles.to_csv(
    OUT_DIR / "clean_articles.csv",
    index=False,
)

# =========================
# 3. Clean customers
# =========================

customers = pd.read_csv(RAW_DIR / "customers.csv")

clean_customers = customers[
    [
        "customer_id",
        "age",
        "club_member_status",
        "fashion_news_frequency",
    ]
].copy()

clean_customers = clean_customers.dropna(subset=["customer_id"])
clean_customers = clean_customers.drop_duplicates(subset=["customer_id"])

clean_customers.to_csv(
    OUT_DIR / "clean_customers.csv",
    index=False,
)

print("Clean data created successfully!")
print(f"- {OUT_DIR / 'clean_transactions.csv'}")
print(f"- {OUT_DIR / 'clean_articles.csv'}")
print(f"- {OUT_DIR / 'clean_customers.csv'}")