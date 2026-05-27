import argparse
import sqlite3
from pathlib import Path

import pandas as pd


TRANSACTION_COLUMNS = ["t_dat", "customer_id", "article_id"]
ARTICLE_COLUMNS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "department_name",
    "section_name",
    "garment_group_name",
]


def normalize_article_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.zfill(10)


def build_baskets(
    transactions_path: Path,
    output_path: Path,
    chunksize: int,
    min_items: int,
    max_rows: int | None,
    keep_db: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    db_path = output_path.parent / "preprocess.sqlite"
    if db_path.exists():
        db_path.unlink()

    total_rows = 0

    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE transactions (
            t_dat TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            article_id TEXT NOT NULL,
            PRIMARY KEY (t_dat, customer_id, article_id)
        )
        """
    )

    reader = pd.read_csv(
        transactions_path,
        usecols=TRANSACTION_COLUMNS,
        dtype={"t_dat": "string", "customer_id": "string", "article_id": "string"},
        chunksize=chunksize,
    )

    try:
        for chunk in reader:
            if max_rows is not None:
                remaining = max_rows - total_rows
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)

            total_rows += len(chunk)
            chunk["article_id"] = normalize_article_id(chunk["article_id"])
            chunk = chunk.drop_duplicates(TRANSACTION_COLUMNS)

            chunk.to_sql("stage_transactions", connection, if_exists="replace", index=False)
            connection.execute(
                """
                INSERT OR IGNORE INTO transactions (t_dat, customer_id, article_id)
                SELECT t_dat, customer_id, article_id
                FROM stage_transactions
                """
            )
            connection.commit()

            if max_rows is not None and total_rows >= max_rows:
                break

        unique_transaction_items = connection.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]

        basket_query = """
            SELECT
                customer_id || '_' || t_dat AS transaction_id,
                customer_id,
                t_dat,
                GROUP_CONCAT(article_id, ' ') AS items,
                COUNT(*) AS item_count
            FROM (
                SELECT t_dat, customer_id, article_id
                FROM transactions
                ORDER BY customer_id, t_dat, article_id
            )
            GROUP BY customer_id, t_dat
            HAVING COUNT(*) >= ?
            ORDER BY t_dat, customer_id
        """

        header = True
        kept_baskets = 0
        for baskets in pd.read_sql_query(
            basket_query,
            connection,
            params=(min_items,),
            chunksize=chunksize,
        ):
            kept_baskets += len(baskets)
            baskets.to_csv(output_path, index=False, header=header, mode="a")
            header = False
    finally:
        connection.close()
        if not keep_db and db_path.exists():
            db_path.unlink()

    print(f"Read transaction rows: {total_rows}")
    print(f"Unique transaction-item rows: {unique_transaction_items}")
    print(f"Kept baskets with at least {min_items} unique items: {kept_baskets}")
    print(f"Wrote: {output_path}")
    if keep_db:
        print(f"Kept SQLite database: {db_path}")


def reduce_articles(articles_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    articles = pd.read_csv(
        articles_path,
        usecols=ARTICLE_COLUMNS,
        dtype={"article_id": "string"},
    )
    articles["article_id"] = normalize_article_id(articles["article_id"])
    articles.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cleaned basket data for Apriori, FP-growth, and Eclat."
    )
    parser.add_argument("--transactions", default="orginal_data_csv/transactions_train.csv")
    parser.add_argument("--articles", default="orginal_data_csv/articles.csv")
    parser.add_argument("--output-dir", default="processed")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--min-items", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--keep-db", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    build_baskets(
        transactions_path=Path(args.transactions),
        output_path=output_dir / "baskets.csv",
        chunksize=args.chunksize,
        min_items=args.min_items,
        max_rows=args.max_rows,
        keep_db=args.keep_db,
    )
    reduce_articles(
        articles_path=Path(args.articles),
        output_path=output_dir / "articles_minimal.csv",
    )


if __name__ == "__main__":
    main()
