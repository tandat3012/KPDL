import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from step_03_benchmark_algorithms import DEFAULT_ALGORITHMS
from step_04_plot_algorithm_comparison import run_algorithms


DEFAULT_CONFIGS = [
    {
        "config": "small",
        "max_baskets": 50_000,
        "top_items": 500,
        "min_support": 0.01,
        "max_len": 3,
    },
    {
        "config": "medium",
        "max_baskets": 100_000,
        "top_items": 1_000,
        "min_support": 0.005,
        "max_len": 3,
    },
    {
        "config": "lower_support",
        "max_baskets": 100_000,
        "top_items": 1_000,
        "min_support": 0.003,
        "max_len": 3,
    },
    {
        "config": "more_items",
        "max_baskets": 100_000,
        "top_items": 1_500,
        "min_support": 0.005,
        "max_len": 3,
    },
    {
        "config": "recent_2020",
        "max_baskets": 150_000,
        "top_items": 1_000,
        "min_support": 0.005,
        "max_len": 3,
        "start_date": "2020-01-01",
        "end_date": "2020-09-22",
    },
]


QUICK_CONFIGS = [
    {
        "config": "quick_small",
        "max_baskets": 10_000,
        "top_items": 100,
        "min_support": 0.02,
        "max_len": 3,
    },
    {
        "config": "quick_medium",
        "max_baskets": 20_000,
        "top_items": 200,
        "min_support": 0.015,
        "max_len": 3,
    },
]


def load_configs(config_file: str | None, preset: str) -> list[dict[str, object]]:
    if config_file is None:
        return QUICK_CONFIGS if preset == "quick" else DEFAULT_CONFIGS

    path = Path(config_file)
    with path.open(encoding="utf-8") as file:
        configs = json.load(file)

    if not isinstance(configs, list):
        raise ValueError("Config file must contain a JSON list.")
    return configs


def build_args(base_args: argparse.Namespace, config: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        baskets=base_args.baskets,
        algorithms=base_args.algorithms,
        max_baskets=int(config.get("max_baskets", base_args.max_baskets)),
        top_items=int(config.get("top_items", base_args.top_items)),
        min_items=int(config.get("min_items", base_args.min_items)),
        min_support=float(config.get("min_support", base_args.min_support)),
        max_len=int(config.get("max_len", base_args.max_len)),
        start_date=config.get("start_date", base_args.start_date),
        end_date=config.get("end_date", base_args.end_date),
    )


def flatten_summary(config: dict[str, object], summary: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    config_name = str(config.get("config", "unnamed"))
    comparison = summary.get("comparison", {})
    common_itemsets = comparison.get("common_itemsets", 0)

    for algorithm, metrics in summary["metrics"].items():
        rows.append(
            {
                "config": config_name,
                "algorithm": algorithm,
                "elapsed_seconds": round(metrics["elapsed_seconds"], 6),
                "peak_memory_mb": round(metrics["peak_memory_mb"], 6),
                "frequent_itemsets": metrics["frequent_itemsets"],
                "max_itemset_length": metrics["max_itemset_length"],
                "transactions": summary["transactions"],
                "unique_items": summary["unique_items"],
                "max_baskets": summary["max_baskets"],
                "top_items": summary["top_items"],
                "min_support": summary["min_support"],
                "max_len": summary["max_len"],
                "common_itemsets": common_itemsets,
                "start_date": config.get("start_date"),
                "end_date": config.get("end_date"),
                "status": "ok",
                "error": "",
            }
        )
    return rows


def save_config_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_CONFIGS, indent=2), encoding="utf-8")
    print(f"Wrote config template: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple benchmark configurations and export a CSV report."
    )
    parser.add_argument("--baskets", default="processed/baskets.csv")
    parser.add_argument("--output", default="results/experiments.csv")
    parser.add_argument("--summary-output", default="results/experiments_summary.json")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--preset", choices=["quick", "default"], default="default")
    parser.add_argument("--algorithms", nargs="+", choices=DEFAULT_ALGORITHMS, default=DEFAULT_ALGORITHMS)
    parser.add_argument("--max-baskets", type=int, default=50_000)
    parser.add_argument("--top-items", type=int, default=500)
    parser.add_argument("--min-items", type=int, default=2)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--max-len", type=int, default=3)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--write-config-template", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_config_template:
        save_config_template(Path(args.write_config_template))
        return

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    configs = load_configs(args.config_file, args.preset)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    started_all = time.perf_counter()
    for index, config in enumerate(configs, start=1):
        config_name = str(config.get("config", f"config_{index}"))
        print(f"[{index}/{len(configs)}] Running {config_name}...", flush=True)

        try:
            run_args = build_args(args, config)
            summary, _ = run_algorithms(run_args)
            rows = flatten_summary(config, summary)
            all_rows.extend(rows)
            summaries.append({"config": config_name, "summary": summary})

            for row in rows:
                print(
                    f"  {row['algorithm']}: "
                    f"{row['elapsed_seconds']}s, "
                    f"{row['peak_memory_mb']} MiB, "
                    f"{row['frequent_itemsets']} itemsets",
                    flush=True,
                )
        except Exception as exc:
            error = str(exc)
            print(f"  ERROR: {error}", file=sys.stderr, flush=True)
            all_rows.append(
                {
                    "config": config_name,
                    "algorithm": "",
                    "elapsed_seconds": "",
                    "peak_memory_mb": "",
                    "frequent_itemsets": "",
                    "max_itemset_length": "",
                    "transactions": "",
                    "unique_items": "",
                    "max_baskets": config.get("max_baskets"),
                    "top_items": config.get("top_items"),
                    "min_support": config.get("min_support"),
                    "max_len": config.get("max_len"),
                    "common_itemsets": "",
                    "start_date": config.get("start_date"),
                    "end_date": config.get("end_date"),
                    "status": "error",
                    "error": error,
                }
            )
            if not args.continue_on_error:
                break

        pd.DataFrame(all_rows).to_csv(output_path, index=False)

    elapsed_all = time.perf_counter() - started_all
    summary_payload = {
        "elapsed_seconds": round(elapsed_all, 6),
        "output": str(output_path),
        "config_count": len(configs),
        "algorithms": args.algorithms,
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote CSV: {output_path}")
    print(f"Wrote JSON summary: {summary_path}")
    print(f"Total elapsed: {elapsed_all:.2f}s")


if __name__ == "__main__":
    main()
