import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_algorithms import (
    DEFAULT_ALGORITHMS,
    compare_results,
    encode_transactions,
    load_baskets,
    run_eclat,
    run_mlxtend_algorithm,
)
import time


def run_algorithms(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
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

    encoded = None
    if any(name in args.algorithms for name in ["apriori", "fpgrowth"]):
        encoded = encode_transactions(transactions)

    results: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, float | int]] = {}
    for name in args.algorithms:
        started = time.perf_counter()
        if name == "eclat":
            itemsets = run_eclat(transactions, args.min_support, args.max_len)
        else:
            if encoded is None:
                raise RuntimeError("Encoded matrix was not prepared.")
            itemsets = run_mlxtend_algorithm(name, encoded, args.min_support, args.max_len)

        elapsed = time.perf_counter() - started
        results[name] = itemsets
        metrics[name] = {
            "elapsed_seconds": elapsed,
            "frequent_itemsets": len(itemsets),
            "max_itemset_length": int(itemsets["length"].max()) if not itemsets.empty else 0,
        }

    summary = {
        "transactions": len(transactions),
        "unique_items": len({item for transaction in transactions for item in transaction}),
        "max_baskets": max_baskets,
        "top_items": top_items,
        "min_support": args.min_support,
        "max_len": args.max_len,
        "metrics": metrics,
        "comparison": compare_results(results),
    }
    return summary, results


def add_bar_labels(axis, values: list[float], fmt: str) -> None:
    max_value = max(values) if values else 0
    offset = max_value * 0.02 if max_value else 0.01
    for index, value in enumerate(values):
        axis.text(index, value + offset, fmt.format(value), ha="center", va="bottom", fontsize=9)


def plot_comparison(summary: dict[str, object], output_path: Path) -> None:
    metrics = summary["metrics"]
    names = list(metrics.keys())
    elapsed = [metrics[name]["elapsed_seconds"] for name in names]
    itemsets = [metrics[name]["frequent_itemsets"] for name in names]
    max_lengths = [metrics[name]["max_itemset_length"] for name in names]

    comparison = summary["comparison"]
    common_itemsets = comparison["common_itemsets"]
    pairwise = comparison["pairwise_differences"]
    pairwise_labels = list(pairwise.keys())
    pairwise_diffs = [
        sum(value for value in pairwise[label].values())
        for label in pairwise_labels
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        (
            "Algorithm Comparison: Apriori vs FP-growth vs Eclat\n"
            f"transactions={summary['transactions']:,}, unique_items={summary['unique_items']:,}, "
            f"min_support={summary['min_support']}, max_len={summary['max_len']}"
        ),
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].bar(names, elapsed, color=["#4c78a8", "#f58518", "#54a24b"][: len(names)])
    axes[0, 0].set_title("Runtime")
    axes[0, 0].set_ylabel("Seconds")
    add_bar_labels(axes[0, 0], elapsed, "{:.4f}s")

    axes[0, 1].bar(names, itemsets, color=["#4c78a8", "#f58518", "#54a24b"][: len(names)])
    axes[0, 1].set_title("Frequent Itemsets")
    axes[0, 1].set_ylabel("Count")
    add_bar_labels(axes[0, 1], itemsets, "{:.0f}")

    axes[1, 0].bar(names, max_lengths, color=["#4c78a8", "#f58518", "#54a24b"][: len(names)])
    axes[1, 0].set_title("Max Itemset Length Found")
    axes[1, 0].set_ylabel("Length")
    axes[1, 0].set_ylim(0, max(max_lengths + [1]) + 1)
    add_bar_labels(axes[1, 0], max_lengths, "{:.0f}")

    labels = ["common"] + pairwise_labels
    values = [common_itemsets] + pairwise_diffs
    axes[1, 1].bar(labels, values, color=["#72b7b2", "#e45756", "#e45756", "#e45756"][: len(labels)])
    axes[1, 1].set_title("Result Agreement")
    axes[1, 1].set_ylabel("Itemset Count")
    axes[1, 1].tick_params(axis="x", labelrotation=20)
    add_bar_labels(axes[1, 1], values, "{:.0f}")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three algorithms and export a comparison chart image."
    )
    parser.add_argument("--baskets", default="processed/baskets.csv")
    parser.add_argument("--output", default="results/algorithm_comparison.png")
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
    summary, _ = run_algorithms(args)
    output_path = Path(args.output)
    plot_comparison(summary, output_path)
    print(f"Wrote chart: {output_path}")
    print(
        "Input: "
        f"transactions={summary['transactions']}, "
        f"unique_items={summary['unique_items']}, "
        f"min_support={summary['min_support']}, "
        f"max_len={summary['max_len']}"
    )
    for name, metrics in summary["metrics"].items():
        print(
            f"{name}: "
            f"{metrics['elapsed_seconds']:.4f}s, "
            f"{metrics['frequent_itemsets']} itemsets, "
            f"max_len_found={metrics['max_itemset_length']}"
        )


if __name__ == "__main__":
    main()
