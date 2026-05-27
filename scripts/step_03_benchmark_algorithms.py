import argparse
import json
import math
import time
from itertools import combinations
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


DEFAULT_ALGORITHMS = ["apriori", "fpgrowth", "eclat"]


def parse_items(value: str) -> list[str]:
    if not isinstance(value, str):
        return []
    cleaned = value.strip("[]")
    if "," in cleaned:
        items = [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip()]
    else:
        items = [item for item in cleaned.split() if item]
    return [item.zfill(10) for item in items]


def load_baskets(
    baskets_path: Path,
    max_baskets: int | None,
    start_date: str | None,
    end_date: str | None,
    top_items: int | None,
    min_items: int,
) -> list[list[str]]:
    usecols = ["t_dat", "items"]
    if start_date is None and end_date is None:
        baskets = pd.read_csv(baskets_path, usecols=usecols, nrows=max_baskets)
    else:
        chunks = []
        collected = 0
        for chunk in pd.read_csv(baskets_path, usecols=usecols, chunksize=200_000):
            if start_date is not None:
                chunk = chunk[chunk["t_dat"] >= start_date]
            if end_date is not None:
                chunk = chunk[chunk["t_dat"] <= end_date]
            if chunk.empty:
                continue

            if max_baskets is not None:
                remaining = max_baskets - collected
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)

            chunks.append(chunk)
            collected += len(chunk)
            if max_baskets is not None and collected >= max_baskets:
                break

        baskets = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)

    transactions = baskets["items"].map(parse_items).tolist()

    if top_items is not None:
        counts: dict[str, int] = {}
        for transaction in transactions:
            for item in transaction:
                counts[item] = counts.get(item, 0) + 1

        allowed_items = {
            item
            for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:top_items]
        }
        transactions = [
            [item for item in transaction if item in allowed_items]
            for transaction in transactions
        ]

    return [sorted(set(transaction)) for transaction in transactions if len(set(transaction)) >= min_items]


def encode_transactions(transactions: list[list[str]]) -> pd.DataFrame:
    encoder = TransactionEncoder()
    matrix = encoder.fit(transactions).transform(transactions, sparse=True)
    return pd.DataFrame.sparse.from_spmatrix(matrix, columns=encoder.columns_)


def normalize_itemsets(itemsets: pd.DataFrame) -> pd.DataFrame:
    if itemsets.empty:
        return pd.DataFrame(columns=["support", "itemsets", "length"])

    normalized = itemsets.copy()
    normalized["itemsets"] = normalized["itemsets"].map(lambda itemset: tuple(sorted(itemset)))
    normalized["length"] = normalized["itemsets"].map(len)
    normalized = normalized.sort_values(["length", "support", "itemsets"], ascending=[True, False, True])
    return normalized.reset_index(drop=True)


def run_mlxtend_algorithm(
    name: str,
    encoded: pd.DataFrame,
    min_support: float,
    max_len: int,
) -> pd.DataFrame:
    algorithm = apriori if name == "apriori" else fpgrowth
    itemsets = algorithm(
        encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    return normalize_itemsets(itemsets)


def run_eclat(
    transactions: list[list[str]],
    min_support: float,
    max_len: int,
) -> pd.DataFrame:
    transaction_count = len(transactions)
    min_count = math.ceil(min_support * transaction_count)
    vertical: dict[str, set[int]] = {}

    for transaction_id, transaction in enumerate(transactions):
        for item in transaction:
            vertical.setdefault(item, set()).add(transaction_id)

    frequent = [
        (item, tids)
        for item, tids in sorted(vertical.items())
        if len(tids) >= min_count
    ]
    rows = [
        {"support": len(tids) / transaction_count, "itemsets": (item,)}
        for item, tids in frequent
    ]

    def extend(prefix: tuple[str, ...], prefix_tids: set[int], suffix: list[tuple[str, set[int]]]) -> None:
        if len(prefix) >= max_len:
            return

        for index, (item, item_tids) in enumerate(suffix):
            combined_tids = prefix_tids & item_tids
            if len(combined_tids) < min_count:
                continue

            combined = prefix + (item,)
            rows.append(
                {
                    "support": len(combined_tids) / transaction_count,
                    "itemsets": combined,
                }
            )
            extend(combined, combined_tids, suffix[index + 1 :])

    for index, (item, tids) in enumerate(frequent):
        extend((item,), tids, frequent[index + 1 :])

    return normalize_itemsets(pd.DataFrame(rows))


def itemset_key(itemset: tuple[str, ...]) -> str:
    return " ".join(itemset)


def load_article_names(articles_path: Path) -> dict[str, str]:
    if not articles_path.exists():
        return {}

    articles = pd.read_csv(articles_path, usecols=["article_id", "prod_name"], dtype={"article_id": "string"})
    articles["article_id"] = articles["article_id"].astype("string").str.zfill(10)
    return dict(zip(articles["article_id"], articles["prod_name"]))


def write_itemsets(
    itemsets: pd.DataFrame,
    output_path: Path,
    article_names: dict[str, str],
) -> None:
    output = itemsets.copy()
    output["itemsets"] = output["itemsets"].map(itemset_key)
    output["product_names"] = output["itemsets"].map(
        lambda value: " | ".join(article_names.get(item, item) for item in value.split())
    )
    output.to_csv(output_path, index=False)


def compare_results(results: dict[str, pd.DataFrame]) -> dict[str, object]:
    itemset_sets = {
        name: set(itemsets["itemsets"].map(itemset_key))
        for name, itemsets in results.items()
    }
    if not itemset_sets:
        return {"common_itemsets": 0, "pairwise_differences": {}}

    common = set.intersection(*itemset_sets.values())
    pairwise = {}
    names = sorted(itemset_sets)
    for left, right in combinations(names, 2):
        pairwise[f"{left}_vs_{right}"] = {
            f"only_{left}": len(itemset_sets[left] - itemset_sets[right]),
            f"only_{right}": len(itemset_sets[right] - itemset_sets[left]),
        }

    return {
        "common_itemsets": len(common),
        "pairwise_differences": pairwise,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Apriori, FP-growth, and Eclat on the same basket data."
    )
    parser.add_argument("--baskets", default="processed/baskets.csv")
    parser.add_argument("--articles", default="processed/articles_minimal.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--algorithms", nargs="+", choices=DEFAULT_ALGORITHMS, default=DEFAULT_ALGORITHMS)
    parser.add_argument("--max-baskets", type=int, default=50_000, help="Use 0 to read all baskets.")
    parser.add_argument("--top-items", type=int, default=500, help="Use 0 to keep all items.")
    parser.add_argument("--min-items", type=int, default=2)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--max-len", type=int, default=3)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_baskets = args.max_baskets if args.max_baskets > 0 else None
    top_items = args.top_items if args.top_items > 0 else None

    transactions = load_baskets(
        baskets_path=Path(args.baskets),
        max_baskets=max_baskets,
        start_date=args.start_date,
        end_date=args.end_date,
        top_items=top_items,
        min_items=args.min_items,
    )
    if not transactions:
        raise SystemExit("No transactions left after filtering.")

    item_count = len({item for transaction in transactions for item in transaction})
    encoded = None
    if any(name in args.algorithms for name in ["apriori", "fpgrowth"]):
        encoded = encode_transactions(transactions)

    article_names = load_article_names(Path(args.articles))
    results: dict[str, pd.DataFrame] = {}
    summary: dict[str, object] = {
        "input": {
            "baskets": args.baskets,
            "transactions": len(transactions),
            "unique_items": item_count,
            "max_baskets": max_baskets,
            "top_items": top_items,
            "min_items": args.min_items,
            "min_support": args.min_support,
            "max_len": args.max_len,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
        "algorithms": {},
    }

    for name in args.algorithms:
        started = time.perf_counter()
        if name == "eclat":
            itemsets = run_eclat(transactions, args.min_support, args.max_len)
        else:
            if encoded is None:
                raise RuntimeError("Encoded matrix was not prepared.")
            itemsets = run_mlxtend_algorithm(name, encoded, args.min_support, args.max_len)

        elapsed_seconds = time.perf_counter() - started
        results[name] = itemsets
        write_itemsets(itemsets, output_dir / f"{name}_itemsets.csv", article_names)
        summary["algorithms"][name] = {
            "elapsed_seconds": round(elapsed_seconds, 4),
            "frequent_itemsets": len(itemsets),
            "max_itemset_length": int(itemsets["length"].max()) if not itemsets.empty else 0,
        }

    summary["comparison"] = compare_results(results)

    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote results to: {output_dir}")


if __name__ == "__main__":
    main()
