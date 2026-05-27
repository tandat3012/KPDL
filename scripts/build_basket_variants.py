import argparse
import json
import re
from pathlib import Path

import pandas as pd


AGE_GROUPS = [
    ("age_under_25", "Tuổi dưới 25", 0, 24),
    ("age_25_34", "Tuổi 25-34", 25, 34),
    ("age_35_44", "Tuổi 35-44", 35, 44),
    ("age_45_plus", "Tuổi từ 45", 45, 200),
]

YEAR_RANGES = [
    ("year_2018", "Năm 2018", "2018-09-20", "2018-12-31"),
    ("year_2019", "Năm 2019", "2019-01-01", "2019-12-31"),
    ("year_2020", "Năm 2020", "2020-01-01", "2020-12-31"),
]

CATEGORY_VARIANTS = [
    (
        "product_group",
        "Basket cấp nhóm sản phẩm",
        "product_group_name",
        "Thay article_id bằng product_group_name để giảm độ thưa và tìm quan hệ ở cấp nhóm hàng.",
    ),
    (
        "garment_group",
        "Basket cấp garment group",
        "garment_group_name",
        "Thay article_id bằng garment_group_name để tìm quan hệ giữa các nhóm trang phục.",
    ),
    (
        "section",
        "Basket cấp section",
        "section_name",
        "Thay article_id bằng section_name để phân tích theo khu vực/đối tượng sản phẩm.",
    ),
]


def safe_token(value: object) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_") or "unknown"


def write_chunk(
    rows: list[dict[str, object]],
    output_path: Path,
    wrote_header: bool,
) -> bool:
    if not rows:
        return wrote_header

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, mode="a", header=not wrote_header)
    return True


def reset_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def load_article_mapping(articles_path: Path, column: str, prefix: str) -> dict[str, str]:
    articles = pd.read_csv(
        articles_path,
        usecols=["article_id", column],
        dtype={"article_id": "string", column: "string"},
    )
    articles["article_id"] = articles["article_id"].astype("string").str.zfill(10)
    articles[column] = articles[column].fillna("unknown")
    return {
        row.article_id: f"{prefix}_{safe_token(getattr(row, column))}"
        for row in articles.itertuples(index=False)
    }


def build_category_variant(
    baskets_path: Path,
    articles_path: Path,
    output_path: Path,
    column: str,
    prefix: str,
    min_items: int,
    chunksize: int,
) -> int:
    reset_output(output_path)
    mapping = load_article_mapping(articles_path, column, prefix)
    wrote_header = False
    total = 0

    for chunk in pd.read_csv(baskets_path, chunksize=chunksize):
        rows = []
        for row in chunk.itertuples(index=False):
            mapped_items = {
                mapping[item]
                for item in str(row.items).split()
                if item in mapping
            }
            if len(mapped_items) < min_items:
                continue

            rows.append(
                {
                    "transaction_id": row.transaction_id,
                    "customer_id": row.customer_id,
                    "t_dat": row.t_dat,
                    "items": " ".join(sorted(mapped_items)),
                    "item_count": len(mapped_items),
                }
            )

        total += len(rows)
        wrote_header = write_chunk(rows, output_path, wrote_header)

    return total


def build_time_variant(
    baskets_path: Path,
    output_path: Path,
    start_date: str,
    end_date: str,
    min_items: int,
    chunksize: int,
) -> int:
    reset_output(output_path)
    wrote_header = False
    total = 0

    for chunk in pd.read_csv(baskets_path, chunksize=chunksize):
        chunk = chunk[(chunk["t_dat"] >= start_date) & (chunk["t_dat"] <= end_date)]
        chunk = chunk[chunk["item_count"] >= min_items]
        total += len(chunk)
        wrote_header = write_chunk(chunk.to_dict("records"), output_path, wrote_header)

    return total


def load_customer_age_groups(customers_path: Path) -> dict[str, str]:
    customers = pd.read_csv(
        customers_path,
        usecols=["customer_id", "age"],
        dtype={"customer_id": "string"},
    )
    customers = customers.dropna(subset=["age"])
    customers["age"] = customers["age"].astype(int)

    age_groups = {}
    for key, _, start, end in AGE_GROUPS:
        group = customers[(customers["age"] >= start) & (customers["age"] <= end)]
        age_groups.update({customer_id: key for customer_id in group["customer_id"]})
    return age_groups


def build_age_variants(
    baskets_path: Path,
    customers_path: Path,
    output_dir: Path,
    min_items: int,
    chunksize: int,
) -> dict[str, int]:
    age_groups = load_customer_age_groups(customers_path)
    output_paths = {
        key: output_dir / f"{key}_baskets.csv"
        for key, _, _, _ in AGE_GROUPS
    }
    for path in output_paths.values():
        reset_output(path)

    wrote_headers = {key: False for key in output_paths}
    totals = {key: 0 for key in output_paths}

    for chunk in pd.read_csv(baskets_path, chunksize=chunksize):
        chunk = chunk[chunk["item_count"] >= min_items].copy()
        chunk["age_group"] = chunk["customer_id"].map(age_groups)
        chunk = chunk.dropna(subset=["age_group"])

        for key, group in chunk.groupby("age_group"):
            group = group.drop(columns=["age_group"])
            rows = group.to_dict("records")
            totals[key] += len(rows)
            wrote_headers[key] = write_chunk(rows, output_paths[key], wrote_headers[key])

    return totals


def build_outfit_variant(
    baskets_path: Path,
    articles_path: Path,
    output_path: Path,
    min_items: int,
    chunksize: int,
) -> int:
    reset_output(output_path)
    mapping = load_article_mapping(articles_path, "garment_group_name", "outfit")
    wrote_header = False
    total = 0

    for chunk in pd.read_csv(baskets_path, chunksize=chunksize):
        rows = []
        for row in chunk.itertuples(index=False):
            outfit_parts = {
                mapping[item]
                for item in str(row.items).split()
                if item in mapping
            }
            if len(outfit_parts) < min_items:
                continue

            rows.append(
                {
                    "transaction_id": row.transaction_id,
                    "customer_id": row.customer_id,
                    "t_dat": row.t_dat,
                    "items": " ".join(sorted(outfit_parts)),
                    "item_count": len(outfit_parts),
                }
            )

        total += len(rows)
        wrote_header = write_chunk(rows, output_path, wrote_header)

    return total


def add_manifest_entry(
    manifest: list[dict[str, object]],
    key: str,
    label: str,
    path: Path,
    description: str,
    rows: int,
) -> None:
    if rows <= 0 or not path.exists():
        return

    manifest.append(
        {
            "key": key,
            "label": label,
            "path": str(path),
            "description": description,
            "rows": rows,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build alternative basket datasets for UI experiments.")
    parser.add_argument("--baskets", default="processed/baskets.csv")
    parser.add_argument("--articles", default="processed/articles_minimal.csv")
    parser.add_argument("--customers", default="orginal_data_csv/customers.csv")
    parser.add_argument("--output-dir", default="processed/basket_variants")
    parser.add_argument("--min-items", type=int, default=2)
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baskets_path = Path(args.baskets)
    articles_path = Path(args.articles)
    customers_path = Path(args.customers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = [
        {
            "key": "base_article",
            "label": "Basket gốc - article_id",
            "path": str(baskets_path),
            "description": "Basket gốc theo article_id, dùng để benchmark cơ sở.",
            "rows": None,
        }
    ]

    for key, label, start_date, end_date in YEAR_RANGES:
        output_path = output_dir / f"{key}_baskets.csv"
        rows = build_time_variant(
            baskets_path,
            output_path,
            start_date,
            end_date,
            args.min_items,
            args.chunksize,
        )
        add_manifest_entry(
            manifest,
            key,
            label,
            output_path,
            f"Basket article_id chỉ gồm giao dịch từ {start_date} đến {end_date}.",
            rows,
        )
        print(f"{key}: {rows} baskets")

    if customers_path.exists():
        age_totals = build_age_variants(
            baskets_path,
            customers_path,
            output_dir,
            args.min_items,
            args.chunksize,
        )
        for key, label, _, _ in AGE_GROUPS:
            add_manifest_entry(
                manifest,
                key,
                label,
                output_dir / f"{key}_baskets.csv",
                f"Basket article_id chỉ gồm khách hàng thuộc nhóm {label.lower()}.",
                age_totals.get(key, 0),
            )
            print(f"{key}: {age_totals.get(key, 0)} baskets")
    else:
        print(f"Skip age variants because customers file was not found: {customers_path}")

    for key, label, column, description in CATEGORY_VARIANTS:
        output_path = output_dir / f"{key}_baskets.csv"
        rows = build_category_variant(
            baskets_path,
            articles_path,
            output_path,
            column,
            key,
            args.min_items,
            args.chunksize,
        )
        add_manifest_entry(manifest, key, label, output_path, description, rows)
        print(f"{key}: {rows} baskets")

    outfit_path = output_dir / "outfit_category_baskets.csv"
    outfit_rows = build_outfit_variant(
        baskets_path,
        articles_path,
        outfit_path,
        args.min_items,
        args.chunksize,
    )
    add_manifest_entry(
        manifest,
        "outfit_category",
        "Basket phối đồ - cấp category",
        outfit_path,
        "Chuyển mỗi basket sang các nhóm garment/outfit để tìm combo phối đồ ở cấp khái niệm.",
        outfit_rows,
    )
    print(f"outfit_category: {outfit_rows} baskets")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
