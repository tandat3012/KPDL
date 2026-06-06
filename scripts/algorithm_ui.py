import argparse
import csv
import html
import json
import os
import sys
import time
import webbrowser
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
from mlxtend.frequent_patterns import association_rules
from step_03_benchmark_algorithms import (
    encode_transactions,
    load_article_names,
    load_baskets,
    run_eclat,
    run_mlxtend_algorithm,
)
from step_04_plot_algorithm_comparison import DEFAULT_ALGORITHMS, plot_comparison, run_algorithms


DEFAULTS = {
    "baskets": "processed/baskets.csv",
    "basket_dataset": "base_article",
    "output": "",
    "preset": "custom",
    "max_baskets": "50000",
    "top_items": "500",
    "min_items": "2",
    "min_support": "0.01",
    "max_len": "3",
    "start_date": "",
    "end_date": "",
    "date_range": "all",
    "algorithms": DEFAULT_ALGORITHMS,
}

HISTORY_PATH = Path("results/ui_runs.csv")
CHART_DIR = Path("results/ui_charts")
BASKET_MANIFEST_PATH = Path("processed/basket_variants/manifest.json")
MODEL_DIR = Path("hm_lstm-trans_web_files")
MODEL_COMPARISON_PATH = MODEL_DIR / "result_comparison.csv"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
OLD_TO_NEW_MAP_PATH = MODEL_DIR / "old_to_new_item_map.json"
NEW_TO_OLD_MAP_PATH = MODEL_DIR / "new_to_old_item_map.json"
TOP_ITEMS_PATH = MODEL_DIR / "top_items.csv"
MODEL_HISTORY_PATHS = {
    "LSTM + Attention": MODEL_DIR / "history_lstm.csv",
    "Transformer": MODEL_DIR / "history_transformer.csv",
}
MAX_SEQ_LEN = 20

_PRODUCT_CACHE = None
_MODEL_CACHE = {}

PRESETS = {
    "custom": {
        "label": "Tùy chỉnh",
        "max_baskets": "50000",
        "top_items": "500",
        "min_items": "2",
        "min_support": "0.01",
        "max_len": "3",
        "date_range": "all",
    },
    "quick": {
        "label": "Quick - kiểm tra nhanh",
        "max_baskets": "10000",
        "top_items": "100",
        "min_items": "2",
        "min_support": "0.02",
        "max_len": "3",
        "date_range": "all",
    },
    "small": {
        "label": "Small - nhỏ",
        "max_baskets": "50000",
        "top_items": "500",
        "min_items": "2",
        "min_support": "0.01",
        "max_len": "3",
        "date_range": "all",
    },
    "medium": {
        "label": "Medium - vừa",
        "max_baskets": "100000",
        "top_items": "1000",
        "min_items": "2",
        "min_support": "0.005",
        "max_len": "3",
        "date_range": "all",
    },
    "low_support": {
        "label": "Low Support - nhiều itemset",
        "max_baskets": "100000",
        "top_items": "1000",
        "min_items": "2",
        "min_support": "0.003",
        "max_len": "3",
        "date_range": "all",
    },
    "more_items": {
        "label": "More Items - nhiều sản phẩm",
        "max_baskets": "100000",
        "top_items": "1500",
        "min_items": "2",
        "min_support": "0.005",
        "max_len": "3",
        "date_range": "all",
    },
    "recent_2020": {
        "label": "Recent 2020",
        "max_baskets": "100000",
        "top_items": "1000",
        "min_items": "2",
        "min_support": "0.005",
        "max_len": "3",
        "date_range": "recent_2020",
    },
}

SELECT_OPTIONS = {
    "max_baskets": [
        ("10000", "10,000 - chạy thử rất nhanh"),
        ("50000", "50,000 - nhỏ"),
        ("100000", "100,000 - vừa"),
        ("200000", "200,000 - lớn"),
        ("0", "Toàn bộ - rất nặng"),
    ],
    "top_items": [
        ("100", "Top 100 - rất nhanh"),
        ("500", "Top 500 - nhỏ"),
        ("1000", "Top 1,000 - vừa"),
        ("1500", "Top 1,500 - lớn"),
        ("0", "Tất cả item - rất nặng"),
    ],
    "min_items": [
        ("2", "Tối thiểu 2 item"),
        ("3", "Tối thiểu 3 item"),
        ("4", "Tối thiểu 4 item"),
    ],
    "max_len": [
        ("2", "Tối đa 2 item"),
        ("3", "Tối đa 3 item"),
        ("4", "Tối đa 4 item - nặng"),
    ],
    "min_itemset_length": [
        ("1", "Từ 1 item"),
        ("2", "Từ 2 item - combo cơ bản"),
        ("3", "Từ 3 item - combo/phối đồ rõ hơn"),
        ("4", "Từ 4 item - rất hẹp"),
    ],
    "date_range": [
        ("all", "Tất cả thời gian"),
        ("2018", "Năm 2018"),
        ("2019", "Năm 2019"),
        ("2020", "Năm 2020"),
        ("recent_2020", "2020-01-01 đến 2020-09-22"),
    ],
    "top_results": [
        ("10", "Top 10"),
        ("20", "Top 20"),
        ("30", "Top 30"),
        ("50", "Top 50"),
        ("100", "Top 100"),
    ],
    "display_mode": [
        ("rules", "Luật gợi ý mua kèm"),
        ("itemsets", "Combo phổ biến"),
    ],
    "sort_metric": [
        ("lift", "Lift - độ liên quan"),
        ("confidence", "Confidence - xác suất gợi ý"),
        ("support", "Support - độ phổ biến"),
    ],
}

DATE_RANGES = {
    "all": (None, None),
    "2018": ("2018-09-20", "2018-12-31"),
    "2019": ("2019-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "recent_2020": ("2020-01-01", "2020-09-22"),
}

PATTERN_DEFAULTS = {
    "algorithm": "fpgrowth",
    "max_baskets": "50000",
    "top_items": "500",
    "min_items": "2",
    "min_itemset_length": "2",
    "min_support": "0.01",
    "max_len": "3",
    "date_range": "all",
    "top_results": "30",
    "display_mode": "rules",
    "min_confidence": "0.3",
    "sort_metric": "lift",
}


def parse_int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer.") from exc


def parse_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number.") from exc
    if field in {"min_support", "min_confidence"} and not 0 < parsed <= 1:
        raise ValueError(f"{field} must be greater than 0 and less than or equal to 1.")
    return parsed


def load_basket_datasets() -> list[dict[str, object]]:
    datasets = [
        {
            "key": "base_article",
            "label": "Basket gốc - article_id",
            "path": "processed/baskets.csv",
            "description": "Basket gốc theo article_id.",
        },
        {
            "key": "custom",
            "label": "Custom path",
            "path": "",
            "description": "Nhập đường dẫn CSV thủ công bên dưới.",
        },
    ]

    if not BASKET_MANIFEST_PATH.exists():
        return datasets

    with BASKET_MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)

    seen = {dataset["key"] for dataset in datasets}
    for dataset in manifest:
        key = dataset.get("key")
        path = dataset.get("path")
        if not key or key in seen or not path:
            continue
        if not Path(path).exists():
            continue
        datasets.insert(-1, dataset)
        seen.add(key)
    return datasets


def resolve_basket_path(query: dict[str, list[str]]) -> tuple[str, str]:
    selected_key = query.get("basket_dataset", [DEFAULTS["basket_dataset"]])[0]
    custom_path = query.get("baskets", [DEFAULTS["baskets"]])[0].strip()
    if selected_key == "custom":
        return selected_key, custom_path or DEFAULTS["baskets"]

    for dataset in load_basket_datasets():
        if dataset["key"] == selected_key and dataset.get("path"):
            return selected_key, str(dataset["path"])

    return "base_article", DEFAULTS["baskets"]


def params_from_query(query: dict[str, list[str]]) -> SimpleNamespace:
    selected_algorithms = query.get("algorithms", DEFAULTS["algorithms"])
    algorithms = [name for name in selected_algorithms if name in DEFAULT_ALGORITHMS]
    if not algorithms:
        raise ValueError("Select at least one algorithm.")

    date_range = query.get("date_range", [DEFAULTS["date_range"]])[0]
    start_date, end_date = DATE_RANGES.get(date_range, DATE_RANGES["all"])
    output = query.get("output", [DEFAULTS["output"]])[0].strip()
    if not output:
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        output = str(CHART_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    basket_dataset, baskets = resolve_basket_path(query)

    return SimpleNamespace(
        baskets=baskets,
        basket_dataset=basket_dataset,
        output=output,
        preset=query.get("preset", [DEFAULTS["preset"]])[0],
        algorithms=algorithms,
        max_baskets=parse_int(query.get("max_baskets", [DEFAULTS["max_baskets"]])[0], "max_baskets"),
        top_items=parse_int(query.get("top_items", [DEFAULTS["top_items"]])[0], "top_items"),
        min_items=parse_int(query.get("min_items", [DEFAULTS["min_items"]])[0], "min_items"),
        min_support=parse_float(query.get("min_support", [DEFAULTS["min_support"]])[0], "min_support"),
        max_len=parse_int(query.get("max_len", [DEFAULTS["max_len"]])[0], "max_len"),
        start_date=start_date,
        end_date=end_date,
        date_range=date_range,
    )


def pattern_params_from_query(query: dict[str, list[str]]) -> SimpleNamespace:
    algorithm = query.get("algorithm", [PATTERN_DEFAULTS["algorithm"]])[0]
    if algorithm not in DEFAULT_ALGORITHMS:
        raise ValueError("Invalid algorithm.")

    date_range = query.get("date_range", [PATTERN_DEFAULTS["date_range"]])[0]
    start_date, end_date = DATE_RANGES.get(date_range, DATE_RANGES["all"])
    basket_dataset, baskets = resolve_basket_path(query)

    min_itemset_length = parse_int(
        query.get("min_itemset_length", [PATTERN_DEFAULTS["min_itemset_length"]])[0],
        "min_itemset_length",
    )
    max_len = parse_int(query.get("max_len", [PATTERN_DEFAULTS["max_len"]])[0], "max_len")
    if min_itemset_length > max_len:
        raise ValueError("Độ dài combo tối thiểu không được lớn hơn Max Length.")
    display_mode = query.get("display_mode", [PATTERN_DEFAULTS["display_mode"]])[0]
    if display_mode not in {"itemsets", "rules"}:
        raise ValueError("Invalid display mode.")
    sort_metric = query.get("sort_metric", [PATTERN_DEFAULTS["sort_metric"]])[0]
    if sort_metric not in {"support", "confidence", "lift"}:
        raise ValueError("Invalid sort metric.")

    return SimpleNamespace(
        baskets=baskets,
        basket_dataset=basket_dataset,
        algorithm=algorithm,
        max_baskets=parse_int(query.get("max_baskets", [PATTERN_DEFAULTS["max_baskets"]])[0], "max_baskets"),
        top_items=parse_int(query.get("top_items", [PATTERN_DEFAULTS["top_items"]])[0], "top_items"),
        min_items=parse_int(query.get("min_items", [PATTERN_DEFAULTS["min_items"]])[0], "min_items"),
        min_itemset_length=min_itemset_length,
        min_support=parse_float(query.get("min_support", [PATTERN_DEFAULTS["min_support"]])[0], "min_support"),
        min_confidence=parse_float(
            query.get("min_confidence", [PATTERN_DEFAULTS["min_confidence"]])[0],
            "min_confidence",
        ),
        max_len=max_len,
        top_results=parse_int(query.get("top_results", [PATTERN_DEFAULTS["top_results"]])[0], "top_results"),
        display_mode=display_mode,
        sort_metric=sort_metric,
        start_date=start_date,
        end_date=end_date,
        date_range=date_range,
    )


def values_from_args(args: SimpleNamespace | None) -> dict[str, object]:
    if args is None:
        return DEFAULTS.copy()

    return {
        "baskets": args.baskets,
        "basket_dataset": getattr(args, "basket_dataset", "custom"),
        "output": args.output,
        "preset": getattr(args, "preset", "custom"),
        "max_baskets": str(args.max_baskets),
        "top_items": str(args.top_items),
        "min_items": str(args.min_items),
        "min_support": str(args.min_support),
        "max_len": str(args.max_len),
        "start_date": args.start_date or "",
        "end_date": args.end_date or "",
        "date_range": getattr(args, "date_range", "all"),
        "algorithms": args.algorithms,
    }


def pattern_values_from_args(args: SimpleNamespace | None) -> dict[str, object]:
    if args is None:
        values = DEFAULTS.copy()
        values.update(PATTERN_DEFAULTS)
        return values

    return {
        "baskets": args.baskets,
        "basket_dataset": getattr(args, "basket_dataset", "custom"),
        "algorithm": args.algorithm,
        "max_baskets": str(args.max_baskets),
        "top_items": str(args.top_items),
        "min_items": str(args.min_items),
        "min_itemset_length": str(args.min_itemset_length),
        "min_support": str(args.min_support),
        "min_confidence": str(args.min_confidence),
        "max_len": str(args.max_len),
        "top_results": str(args.top_results),
        "display_mode": args.display_mode,
        "sort_metric": args.sort_metric,
        "date_range": getattr(args, "date_range", "all"),
    }


def load_history(limit: int = 30) -> list[dict[str, str]]:
    if not HISTORY_PATH.exists():
        return []

    with HISTORY_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[-limit:][::-1]


def clear_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def append_history(args: SimpleNamespace, summary: dict[str, object]) -> str:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for algorithm, metrics in summary["metrics"].items():
        rows.append(
            {
                "run_id": run_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "preset": args.preset,
                "basket_dataset": getattr(args, "basket_dataset", "custom"),
                "algorithm": algorithm,
                "elapsed_seconds": f"{metrics['elapsed_seconds']:.6f}",
                "peak_memory_mb": f"{metrics['peak_memory_mb']:.6f}",
                "frequent_itemsets": str(metrics["frequent_itemsets"]),
                "max_itemset_length": str(metrics["max_itemset_length"]),
                "transactions": str(summary["transactions"]),
                "unique_items": str(summary["unique_items"]),
                "max_baskets": str(summary["max_baskets"]),
                "top_items": str(summary["top_items"]),
                "min_items": str(args.min_items),
                "min_support": str(summary["min_support"]),
                "max_len": str(summary["max_len"]),
                "date_range": args.date_range,
                "start_date": args.start_date or "",
                "end_date": args.end_date or "",
                "output": args.output,
            }
        )

    write_header = not HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return run_id


def render_metrics_table(summary: dict[str, object]) -> str:
    rows = []
    for algorithm, metrics in summary["metrics"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(algorithm)}</td>"
            f"<td>{metrics['elapsed_seconds']:.4f}s</td>"
            f"<td>{metrics['peak_memory_mb']:.2f} MiB</td>"
            f"<td>{metrics['frequent_itemsets']}</td>"
            f"<td>{metrics['max_itemset_length']}</td>"
            "</tr>"
        )

    return """
    <h2>Kết quả lần chạy hiện tại</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Thuật toán</th>
            <th>Thời gian</th>
            <th>Bộ nhớ đỉnh</th>
            <th>Frequent itemsets</th>
            <th>Max length</th>
          </tr>
        </thead>
        <tbody>
          %s
        </tbody>
      </table>
    </div>
    """ % "\n".join(rows)


def run_pattern_mining(args: SimpleNamespace) -> tuple[dict[str, object], object, object]:
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
        raise ValueError("No transactions left after filtering.")

    started = time.perf_counter()
    if args.algorithm == "eclat":
        itemsets = run_eclat(transactions, args.min_support, args.max_len)
    else:
        encoded = encode_transactions(transactions)
        itemsets = run_mlxtend_algorithm(args.algorithm, encoded, args.min_support, args.max_len)

    total_frequent_itemsets = len(itemsets)
    filtered_itemsets = itemsets[itemsets["length"] >= args.min_itemset_length]
    filtered_itemsets = filtered_itemsets.sort_values(["support", "length", "itemsets"], ascending=[False, False, True])
    rules = build_association_rules(itemsets, len(transactions), args)
    summary = {
        "basket_dataset": args.basket_dataset,
        "baskets": args.baskets,
        "algorithm": args.algorithm,
        "transactions": len(transactions),
        "unique_items": len({item for transaction in transactions for item in transaction}),
        "frequent_itemsets": len(filtered_itemsets),
        "total_frequent_itemsets": total_frequent_itemsets,
        "rules": len(rules),
        "elapsed_seconds": time.perf_counter() - started,
        "max_baskets": max_baskets,
        "top_items": top_items,
        "min_support": args.min_support,
        "min_confidence": args.min_confidence,
        "min_itemset_length": args.min_itemset_length,
        "max_len": args.max_len,
        "display_mode": args.display_mode,
        "sort_metric": args.sort_metric,
        "date_range": args.date_range,
    }
    return summary, filtered_itemsets.head(args.top_results), rules.head(args.top_results)


def build_association_rules(itemsets, transaction_count: int, args: SimpleNamespace):
    if itemsets.empty or itemsets["length"].max() < 2:
        return itemsets.iloc[0:0].copy()

    try:
        rules = association_rules(
            itemsets,
            num_itemsets=transaction_count,
            metric="confidence",
            min_threshold=args.min_confidence,
        )
    except ValueError:
        return itemsets.iloc[0:0].copy()

    if rules.empty:
        return rules

    rules = rules.copy()
    rules["antecedents"] = rules["antecedents"].map(lambda items: tuple(sorted(items)))
    rules["consequents"] = rules["consequents"].map(lambda items: tuple(sorted(items)))
    rules["rule_length"] = rules["antecedents"].map(len) + rules["consequents"].map(len)
    rules = rules[rules["rule_length"] >= args.min_itemset_length]
    rules = rules.sort_values(
        [args.sort_metric, "confidence", "support"],
        ascending=[False, False, False],
    )
    return rules.reset_index(drop=True)


def readable_item(item: str, article_names: dict[str, str]) -> str:
    if item in article_names:
        return f"{article_names[item]} ({item})"

    prefixes = [
        "product_group_",
        "garment_group_",
        "section_",
        "outfit_",
    ]
    for prefix in prefixes:
        if item.startswith(prefix):
            return item.removeprefix(prefix).replace("_", " ").title()

    return item.replace("_", " ").title()


def readable_items(items: tuple[str, ...], article_names: dict[str, str]) -> str:
    return " + ".join(readable_item(item, article_names) for item in items)


def render_patterns_table(itemsets, basket_dataset: str) -> str:
    article_names = load_article_names(Path("processed/articles_minimal.csv"))
    rows = []
    for row in itemsets.itertuples(index=False):
        raw_items = tuple(row.itemsets)
        display_items = [readable_item(item, article_names) for item in raw_items]
        rows.append(
            "<tr>"
            f"<td>{row.support:.4f}</td>"
            f"<td>{len(raw_items)}</td>"
            f"<td>{html.escape(' + '.join(display_items))}</td>"
            f"<td>{html.escape(' '.join(raw_items))}</td>"
            "</tr>"
        )

    if not rows:
        return "<p>Không tìm thấy frequent itemsets với cấu hình hiện tại.</p>"

    heading = "Top frequent itemsets"
    if basket_dataset.startswith("age_"):
        heading = "Pattern mua hàng theo nhóm tuổi"
    elif basket_dataset.startswith("year_"):
        heading = "Pattern mua hàng theo năm"
    elif basket_dataset in {"product_group", "garment_group", "section", "outfit_category"}:
        heading = "Pattern theo nhóm đồ/category"

    return f"""
    <h2>{heading}</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Support</th>
            <th>Độ dài</th>
            <th>Diễn giải dễ đọc</th>
            <th>Item gốc</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def render_recommendation_cards(itemsets, basket_dataset: str, transactions: int) -> str:
    article_names = load_article_names(Path("processed/articles_minimal.csv"))
    cards = []

    for index, row in enumerate(itemsets.itertuples(index=False), start=1):
        raw_items = tuple(row.itemsets)
        if len(raw_items) < 2:
            continue

        readable_items = [readable_item(item, article_names) for item in raw_items]
        support_count = round(row.support * transactions)
        title = " + ".join(readable_items)

        if basket_dataset in {"product_group", "garment_group", "section", "outfit_category"}:
            message = "Nhóm đồ này thường xuất hiện cùng nhau trong cùng một giỏ hàng."
            action = "Có thể dùng làm gợi ý phối đồ hoặc combo nhóm sản phẩm."
        elif basket_dataset.startswith("age_"):
            message = "Combo này nổi bật trong phân khúc tuổi đã chọn."
            action = "Có thể dùng để giải thích khác biệt hành vi mua theo nhóm tuổi."
        elif basket_dataset.startswith("year_"):
            message = "Combo này nổi bật trong giai đoạn/năm đã chọn."
            action = "Có thể dùng để phân tích xu hướng mua hàng theo thời gian."
        else:
            message = "Các sản phẩm này thường được mua cùng nhau."
            action = "Có thể dùng làm gợi ý mua kèm ở cấp sản phẩm."

        cards.append(
            f"""
            <article class="combo-card">
              <div class="combo-rank">#{index}</div>
              <h3>{html.escape(title)}</h3>
              <p>{html.escape(message)}</p>
              <p>{html.escape(action)}</p>
              <div class="combo-meta">
                <span>Support: {row.support:.2%}</span>
                <span>Khoảng {support_count:,}/{transactions:,} giỏ</span>
              </div>
            </article>
            """
        )

        if len(cards) >= 6:
            break

    if not cards:
        return """
        <div class="recommendations">
          <h2>Gợi ý dễ đọc</h2>
          <p>Chưa có combo từ 2 item trở lên. Hãy giảm min support hoặc tăng số basket/top items.</p>
        </div>
        """

    return f"""
    <div class="recommendations">
      <h2>Gợi ý dễ đọc từ frequent itemsets</h2>
      <p>Các thẻ dưới đây chuyển kết quả thuật toán thành gợi ý combo dễ giải thích hơn.</p>
      <div class="combo-grid">{''.join(cards)}</div>
    </div>
    """


def render_rule_cards(rules, basket_dataset: str, transactions: int) -> str:
    article_names = load_article_names(Path("processed/articles_minimal.csv"))
    cards = []

    for index, row in enumerate(rules.itertuples(index=False), start=1):
        antecedents = tuple(row.antecedents)
        consequents = tuple(row.consequents)
        support_count = round(row.support * transactions)
        antecedent_text = readable_items(antecedents, article_names)
        consequent_text = readable_items(consequents, article_names)

        if basket_dataset in {"product_group", "garment_group", "section", "outfit_category"}:
            action = "Luật này có thể dùng làm gợi ý phối đồ hoặc combo nhóm sản phẩm."
        elif basket_dataset.startswith("age_"):
            action = "Luật này giúp giải thích xu hướng mua kèm trong phân khúc tuổi đã chọn."
        elif basket_dataset.startswith("year_"):
            action = "Luật này giúp mô tả xu hướng mua kèm trong giai đoạn/năm đã chọn."
        else:
            action = "Luật này có thể dùng làm gợi ý mua kèm ở cấp sản phẩm."

        cards.append(
            f"""
            <article class="combo-card">
              <div class="combo-rank">Rule #{index}</div>
              <h3>Nếu mua {html.escape(antecedent_text)}</h3>
              <p>Gợi ý thêm: <strong>{html.escape(consequent_text)}</strong></p>
              <p>{html.escape(action)}</p>
              <div class="combo-meta">
                <span>Confidence: {row.confidence:.2%}</span>
                <span>Lift: {row.lift:.2f}</span>
                <span>Support: {row.support:.2%}</span>
                <span>Khoảng {support_count:,}/{transactions:,} giỏ</span>
              </div>
            </article>
            """
        )

        if len(cards) >= 6:
            break

    if not cards:
        return """
        <div class="recommendations">
          <h2>Luật gợi ý mua kèm</h2>
          <p>Không có rule nào thỏa điều kiện. Hãy giảm Min Support, giảm Min Confidence hoặc tăng số basket/top items.</p>
        </div>
        """

    return f"""
    <div class="recommendations">
      <h2>Luật gợi ý mua kèm</h2>
      <p>Các thẻ dưới đây diễn giải luật dạng: nếu khách mua nhóm A thì nên gợi ý nhóm B.</p>
      <div class="combo-grid">{''.join(cards)}</div>
    </div>
    """


def render_rules_table(rules) -> str:
    article_names = load_article_names(Path("processed/articles_minimal.csv"))
    if rules.empty:
        return "<p>Không có association rules với cấu hình hiện tại.</p>"

    rows = []
    for row in rules.itertuples(index=False):
        antecedents = tuple(row.antecedents)
        consequents = tuple(row.consequents)
        rows.append(
            "<tr>"
            f"<td>{html.escape(readable_items(antecedents, article_names))}</td>"
            f"<td>{html.escape(readable_items(consequents, article_names))}</td>"
            f"<td>{row.support:.4f}</td>"
            f"<td>{row.confidence:.4f}</td>"
            f"<td>{row.lift:.4f}</td>"
            f"<td>{row.rule_length}</td>"
            "</tr>"
        )

    return f"""
    <h2>Bảng association rules</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Nếu mua</th>
            <th>Gợi ý thêm</th>
            <th>Support</th>
            <th>Confidence</th>
            <th>Lift</th>
            <th>Độ dài rule</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def render_rule_insights(rules, args: SimpleNamespace) -> str:
    if rules.empty:
        return """
        <div class="insights">
          <h2>Nhận xét luật gợi ý</h2>
          <ul>
            <li>Không có rule nào thỏa điều kiện hiện tại.</li>
            <li>Thử giảm Min Confidence hoặc Min Support để lấy thêm luật.</li>
          </ul>
        </div>
        """

    best_confidence = rules.sort_values("confidence", ascending=False).iloc[0]
    best_lift = rules.sort_values("lift", ascending=False).iloc[0]
    notes = [
        f"Đang sắp xếp rule theo {args.sort_metric}.",
        f"Rule confidence cao nhất đạt {best_confidence['confidence']:.2%}.",
        f"Rule lift cao nhất đạt {best_lift['lift']:.2f}. Lift > 1 nghĩa là hai phía xuất hiện cùng nhau nhiều hơn ngẫu nhiên.",
    ]
    if best_lift["lift"] <= 1.05:
        notes.append("Lift cao nhất khá gần 1, quan hệ mua kèm chưa thật sự mạnh.")
    if args.min_confidence >= 0.7:
        notes.append("Min Confidence đang cao, số rule có thể ít nhưng thường đáng tin hơn.")

    items = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    return f"""
    <div class="insights">
      <h2>Nhận xét luật gợi ý</h2>
      <ul>{items}</ul>
    </div>
    """


def render_pattern_result(args: SimpleNamespace, summary: dict[str, object], itemsets, rules) -> str:
    main_result = (
        render_rule_insights(rules, args)
        + render_rule_cards(rules, args.basket_dataset, summary["transactions"])
        + render_rules_table(rules)
        if args.display_mode == "rules"
        else render_recommendation_cards(itemsets, args.basket_dataset, summary["transactions"])
        + render_patterns_table(itemsets, args.basket_dataset)
    )
    return f"""
    <div class="insights">
      <h2>Tóm tắt lần phân tích</h2>
      <ul>
        <li>Loại basket: {html.escape(args.basket_dataset)}</li>
        <li>Thuật toán: {html.escape(args.algorithm)}</li>
        <li>Transactions dùng để phân tích: {summary['transactions']:,}</li>
        <li>Số item khác nhau sau lọc: {summary['unique_items']:,}</li>
        <li>Số frequent itemsets sau khi lọc độ dài combo: {summary['frequent_itemsets']:,}</li>
        <li>Tổng frequent itemsets trước khi lọc độ dài combo: {summary['total_frequent_itemsets']:,}</li>
        <li>Số association rules thỏa điều kiện: {summary['rules']:,}</li>
        <li>Chế độ hiển thị: {html.escape(args.display_mode)}</li>
        <li>Min Confidence: {args.min_confidence:.2%}</li>
        <li>Cỡ basket đầu vào tối thiểu: {args.min_items} item</li>
        <li>Độ dài combo tối thiểu hiển thị: {args.min_itemset_length} item</li>
        <li>Thời gian chạy: {summary['elapsed_seconds']:.4f}s</li>
      </ul>
    </div>
    {main_result}
    """


def render_insights(args: SimpleNamespace, summary: dict[str, object]) -> str:
    metrics = summary["metrics"]
    if not metrics:
        return ""

    fastest = min(metrics, key=lambda name: metrics[name]["elapsed_seconds"])
    lowest_memory = min(metrics, key=lambda name: metrics[name]["peak_memory_mb"])
    highest_memory = max(metrics, key=lambda name: metrics[name]["peak_memory_mb"])
    itemset_counts = {
        int(values["frequent_itemsets"])
        for values in metrics.values()
    }

    insights = [
        (
            f"{fastest} chạy nhanh nhất trong cấu hình này "
            f"({metrics[fastest]['elapsed_seconds']:.4f}s)."
        ),
        (
            f"{lowest_memory} dùng ít bộ nhớ đỉnh nhất "
            f"({metrics[lowest_memory]['peak_memory_mb']:.2f} MiB)."
        ),
    ]

    if highest_memory != lowest_memory:
        insights.append(
            f"{highest_memory} dùng nhiều bộ nhớ đỉnh nhất "
            f"({metrics[highest_memory]['peak_memory_mb']:.2f} MiB)."
        )

    if len(itemset_counts) == 1:
        count = next(iter(itemset_counts))
        insights.append(
            f"Các thuật toán tìm cùng {count} frequent itemsets; khác biệt chính nằm ở hiệu năng."
        )
    else:
        insights.append(
            "Số frequent itemsets giữa các thuật toán không giống nhau; cần kiểm tra lại input hoặc tham số."
        )

    if args.min_support <= 0.003:
        insights.append(
            "Min support đang thấp, số candidate/itemset dễ tăng mạnh; Apriori thường bị ảnh hưởng rõ nhất."
        )
    if args.top_items == 0:
        insights.append(
            "Đang giữ tất cả item trong phần dữ liệu đã chọn; cấu hình này có thể rất nặng nếu tăng số basket."
        )
    elif args.top_items >= 1000:
        insights.append(
            "Top items khá cao, số tổ hợp sản phẩm có thể tăng nhanh khi giảm support."
        )
    if args.max_baskets == 0:
        insights.append(
            "Đang đọc toàn bộ basket; nên tránh kết hợp với top_items=tất cả, min_support thấp hoặc max_len cao."
        )
    elif args.max_baskets >= 200000:
        insights.append(
            "Số basket lớn, kết quả đáng tin hơn nhưng runtime và memory sẽ tăng."
        )
    if args.max_len >= 4:
        insights.append(
            "Max length từ 4 trở lên làm không gian tổ hợp lớn hơn nhiều; chỉ nên dùng khi dữ liệu đã lọc nhỏ."
        )

    items = "\n".join(f"<li>{html.escape(text)}</li>" for text in insights)
    return f"""
    <div class="insights">
      <h2>Nhận xét tự động</h2>
      <ul>{items}</ul>
    </div>
    """


def render_history_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return """
        <div>
          <h2>Lịch sử chạy</h2>
          <p>Chưa có lịch sử. Mỗi lần bấm chạy sẽ lưu vào results/ui_runs.csv.</p>
        </div>
        """

    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row.get('created_at', ''))}</td>"
            f"<td>{html.escape(row.get('preset', ''))}</td>"
            f"<td>{html.escape(row.get('algorithm', ''))}</td>"
            f"<td>{html.escape(row.get('elapsed_seconds', ''))}</td>"
            f"<td>{html.escape(row.get('peak_memory_mb', ''))}</td>"
            f"<td>{html.escape(row.get('frequent_itemsets', ''))}</td>"
            f"<td>{html.escape(row.get('max_baskets', ''))}</td>"
            f"<td>{html.escape(row.get('top_items', ''))}</td>"
            f"<td>{html.escape(row.get('min_support', ''))}</td>"
            f"<td>{html.escape(row.get('max_len', ''))}</td>"
            f"<td>{html.escape(row.get('date_range', ''))}</td>"
            "</tr>"
        )

    return """
    <div>
      <div class="section-heading">
        <h2>Lịch sử chạy gần đây</h2>
        <form action="/clear-history" method="post" class="inline-form">
          <button type="submit" class="danger" onclick="return confirm('Xóa toàn bộ lịch sử chạy?')">Xóa lịch sử</button>
        </form>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Thời điểm</th>
              <th>Preset</th>
              <th>Thuật toán</th>
              <th>Giây</th>
              <th>MiB</th>
              <th>Itemsets</th>
              <th>Baskets</th>
              <th>Top items</th>
              <th>Support</th>
              <th>Max len</th>
              <th>Thời gian</th>
            </tr>
          </thead>
          <tbody>
            %s
          </tbody>
        </table>
      </div>
    </div>
    """ % "\n".join(body)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def model_file_size(path: Path) -> str:
    if not path.exists():
        return "Không tìm thấy"
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.2f} MiB"


def load_model_comparison() -> list[dict[str, object]]:
    if not MODEL_COMPARISON_PATH.exists():
        return []

    rows = []
    data = pd.read_csv(MODEL_COMPARISON_PATH)
    for row in data.to_dict("records"):
        model_label = str(row["Model"])
        if "baseline" in model_label.lower():
            continue
        top10 = row.get("Top-10 Accuracy")
        top20 = row.get("Top-20 Accuracy")
        rows.append(
            {
                "model": model_label,
                "loss": float(row["Loss"]) if pd.notna(row.get("Loss")) else None,
                "top1": float(row["Accuracy (Top-1)"]) if pd.notna(row.get("Accuracy (Top-1)")) else None,
                "top3": float(row["Top-3 Accuracy"]) if pd.notna(row.get("Top-3 Accuracy")) else None,
                "top5": float(row["Top-5 Accuracy"]) if pd.notna(row.get("Top-5 Accuracy")) else None,
                "top10": float(top10) if pd.notna(top10) else None,
                "top20": float(top20) if pd.notna(top20) else None,
            }
        )
    return rows


def metric_or_na(value: object, formatter=format_percent) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return formatter(float(value))


def load_model_config() -> dict[str, object]:
    if not MODEL_CONFIG_PATH.exists():
        return {"max_seq_len": MAX_SEQ_LEN, "catalog_size": None, "num_items": None, "models": {}}

    with MODEL_CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def configured_model_path(model_name: str) -> Path:
    config = load_model_config()
    models = config.get("models", {})
    if isinstance(models, dict) and model_name in models:
        return MODEL_DIR / str(models[model_name])
    fallback = {
        "lstm": MODEL_DIR / "best_lstm_attention_model.keras",
        "transformer": MODEL_DIR / "best_transformer_model.keras",
    }
    return fallback[model_name]


def layer_detail(layer: dict[str, object]) -> str:
    config = layer.get("config", {})
    if not isinstance(config, dict):
        return ""

    for key in ("units", "output_dim", "embed_dim", "num_heads", "rate", "activation"):
        value = config.get(key)
        if value is not None:
            return f"{key}: {value}"

    nested_layer = config.get("layer")
    if isinstance(nested_layer, dict):
        nested_config = nested_layer.get("config", {})
        if isinstance(nested_config, dict):
            units = nested_config.get("units")
            if units is not None:
                return f"units: {units}"

    return ""


def inspect_keras_model(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": path, "exists": False, "layers": []}

    try:
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("config.json"))
            metadata = json.loads(archive.read("metadata.json"))
    except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        return {"path": path, "exists": True, "error": str(exc), "layers": []}

    model_config = config.get("config", {})
    layers = []
    if isinstance(model_config, dict):
        for layer in model_config.get("layers", []):
            if not isinstance(layer, dict):
                continue
            layer_config = layer.get("config", {})
            name = layer_config.get("name", "") if isinstance(layer_config, dict) else ""
            layers.append(
                {
                    "class_name": str(layer.get("class_name", "")),
                    "name": str(name),
                    "detail": layer_detail(layer),
                }
            )

    return {
        "path": path,
        "exists": True,
        "model_name": model_config.get("name", "") if isinstance(model_config, dict) else "",
        "keras_version": metadata.get("keras_version", ""),
        "date_saved": metadata.get("date_saved", ""),
        "layers": layers,
    }


def render_model_cards(comparison: list[dict[str, object]]) -> str:
    if not comparison:
        return """
        <div class="error">
          <strong>Thiếu dữ liệu:</strong> Không tìm thấy file so sánh LSTM/Transformer.
        </div>
        """

    top20_rows = [row for row in comparison if row.get("top20") is not None]
    loss_rows = [row for row in comparison if row.get("loss") is not None]
    best_top20 = max(top20_rows, key=lambda row: row["top20"]) if top20_rows else None
    best_loss = min(loss_rows, key=lambda row: row["loss"]) if loss_rows else None
    cards = []
    for row in comparison:
        cards.append(
            f"""
            <article class="metric-card">
              <h3>{html.escape(str(row["model"]))}</h3>
              <p>Đánh giá mô hình dự đoán sản phẩm tiếp theo từ chuỗi giao dịch.</p>
              <div class="metric-value">{metric_or_na(row.get("top20"))}</div>
              <p>Top-20 Accuracy</p>
              <dl>
                <dt>Loss</dt><dd>{metric_or_na(row.get("loss"), lambda value: f"{value:.4f}")}</dd>
                <dt>Top-1</dt><dd>{metric_or_na(row.get("top1"))}</dd>
                <dt>Top-3</dt><dd>{metric_or_na(row.get("top3"))}</dd>
                <dt>Top-5</dt><dd>{metric_or_na(row.get("top5"))}</dd>
                <dt>Top-10</dt><dd>{metric_or_na(row.get("top10"))}</dd>
              </dl>
            </article>
            """
        )

    top20_summary = (
        f"<li>Mô hình Top-20 tốt nhất: {html.escape(str(best_top20['model']))} ({format_percent(float(best_top20['top20']))}).</li>"
        if best_top20
        else "<li>File metric hiện chưa có Top-20 Accuracy hợp lệ để so sánh.</li>"
    )
    loss_summary = (
        f"<li>Mô hình có loss thấp nhất: {html.escape(str(best_loss['model']))} ({float(best_loss['loss']):.4f}).</li>"
        if best_loss
        else "<li>File metric hiện chưa có loss hợp lệ để so sánh.</li>"
    )
    return f"""
    <div class="insights">
      <h2>Tóm tắt nhanh</h2>
      <ul>
        {top20_summary}
        {loss_summary}
        <li>Phần này bổ sung góc nhìn sequence modeling bên cạnh frequent itemset mining của bài chính.</li>
      </ul>
    </div>
    <div class="metric-grid">{''.join(cards)}</div>
    """


def render_metric_bars(comparison: list[dict[str, object]]) -> str:
    metrics = [
        ("Top-1 Accuracy", "top1"),
        ("Top-3 Accuracy", "top3"),
        ("Top-5 Accuracy", "top5"),
        ("Top-10 Accuracy", "top10"),
        ("Top-20 Accuracy", "top20"),
    ]
    rows = []
    for label, key in metrics:
        for row in comparison:
            if row.get(key) is None:
                continue
            value = float(row[key])
            class_name = "transformer" if str(row["model"]).lower().startswith("transformer") else ""
            rows.append(
                f"""
                <div class="bar-row">
                  <span>{html.escape(label)} - {html.escape(str(row["model"]))}</span>
                  <div class="bar-track"><div class="bar-fill {class_name}" style="width: {value * 100:.2f}%"></div></div>
                  <strong>{format_percent(value)}</strong>
                </div>
                """
            )

    return f"""
    <h2>So sánh accuracy</h2>
    <div class="bar-list">{''.join(rows)}</div>
    """


def load_training_histories() -> dict[str, pd.DataFrame]:
    histories = {}
    for label, path in MODEL_HISTORY_PATHS.items():
        if path.exists():
            histories[label] = pd.read_csv(path)
    return histories


def svg_points(values: list[float], width: int, height: int, min_value: float, max_value: float) -> str:
    if not values:
        return ""
    if len(values) == 1:
        x_values = [width / 2]
    else:
        x_values = [index * width / (len(values) - 1) for index in range(len(values))]
    span = max(max_value - min_value, 1e-9)
    points = []
    for x, value in zip(x_values, values):
        y = height - ((value - min_value) / span * height)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def render_line_chart(title: str, histories: dict[str, pd.DataFrame], columns: list[tuple[str, str]]) -> str:
    width = 520
    height = 180
    series = []
    for model_label, frame in histories.items():
        for column, label in columns:
            if column not in frame:
                continue
            values = [float(value) for value in frame[column].dropna().tolist()]
            if values:
                series.append((model_label, label, values))

    if not series:
        return ""

    all_values = [value for _, _, values in series for value in values]
    min_value = min(all_values)
    max_value = max(all_values)
    if max_value <= 1.0:
        min_value = max(0.0, min_value - 0.02)
        max_value = min(1.0, max_value + 0.02)
    else:
        padding = (max_value - min_value) * 0.08
        min_value -= padding
        max_value += padding

    colors = ["#156f6d", "#d27a1f", "#263443", "#8a4b9e"]
    polylines = []
    legends = []
    for index, (model_label, label, values) in enumerate(series):
        color = colors[index % len(colors)]
        dash = "5 5" if "Validation" in label else ""
        polylines.append(
            f'<polyline points="{svg_points(values, width, height, min_value, max_value)}" '
            f'fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" '
            f'stroke-dasharray="{dash}" />'
        )
        legends.append(
            f'<span><i style="background:{color}"></i>{html.escape(model_label)} - {html.escape(label)}</span>'
        )

    return f"""
    <article class="chart-card">
      <h3>{html.escape(title)}</h3>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <line x1="0" y1="{height}" x2="{width}" y2="{height}" />
        <line x1="0" y1="0" x2="0" y2="{height}" />
        {''.join(polylines)}
      </svg>
      <div class="chart-legend">{''.join(legends)}</div>
    </article>
    """


def render_training_charts() -> str:
    histories = load_training_histories()
    if not histories:
        return ""

    return f"""
    <h2>Biểu đồ quá trình train</h2>
    <div class="chart-grid">
      {render_line_chart("Top-10 Accuracy theo epoch", histories, [("top_10_accuracy", "Train"), ("val_top_10_accuracy", "Validation")])}
      {render_line_chart("Loss theo epoch", histories, [("loss", "Train"), ("val_loss", "Validation")])}
    </div>
    """


def render_metric_heatmap(comparison: list[dict[str, object]]) -> str:
    metrics = [
        ("Top-1", "top1"),
        ("Top-3", "top3"),
        ("Top-5", "top5"),
        ("Top-10", "top10"),
        ("Top-20", "top20"),
    ]
    values = [float(row[key]) for row in comparison for _, key in metrics if row.get(key) is not None]
    if not values:
        return ""

    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    rows = []
    for row in comparison:
        cells = []
        for label, key in metrics:
            value = row.get(key)
            if value is None:
                cells.append("<td>N/A</td>")
                continue
            intensity = (float(value) - min_value) / span
            cells.append(
                f'<td style="--heat:{intensity:.3f}"><strong>{format_percent(float(value))}</strong><span>{html.escape(label)}</span></td>'
            )
        rows.append(
            f"<tr><th>{html.escape(str(row['model']))}</th>{''.join(cells)}</tr>"
        )

    return f"""
    <h2>Heatmap Top-K Accuracy</h2>
    <div class="table-wrap heatmap-wrap">
      <table class="heatmap">
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    <p class="note">Chưa hiển thị ma trận nhầm lẫn vì thư mục model hiện không có file nhãn thật/dự đoán từng mẫu. Heatmap này dùng metric đánh giá thật từ Kaggle.</p>
    """


def render_architecture_card(title: str, info: dict[str, object]) -> str:
    if not info.get("exists"):
        return f"""
        <article class="model-card">
          <h3>{html.escape(title)}</h3>
          <p>Không tìm thấy file model.</p>
        </article>
        """

    layers = info.get("layers", [])
    layer_items = []
    for layer in layers[:12]:
        if not isinstance(layer, dict):
            continue
        label = layer.get("class_name", "")
        name = layer.get("name", "")
        detail = layer.get("detail", "")
        right = " / ".join(part for part in (str(name), str(detail)) if part)
        layer_items.append(
            f"<li><span>{html.escape(str(label))}</span><span>{html.escape(right)}</span></li>"
        )

    error = info.get("error")
    error_html = f"<p class='error'>{html.escape(str(error))}</p>" if error else ""
    return f"""
    <article class="model-card">
      <h3>{html.escape(title)}</h3>
      <p>Thông tin đọc trực tiếp từ file `.keras`.</p>
      <dl>
        <dt>File</dt><dd>{html.escape(str(info["path"]))}</dd>
        <dt>Dung lượng</dt><dd>{model_file_size(Path(info["path"]))}</dd>
        <dt>Keras</dt><dd>{html.escape(str(info.get("keras_version", ""))) or "N/A"}</dd>
        <dt>Số layer</dt><dd>{len(layers)}</dd>
      </dl>
      {error_html}
      <ul class="layer-list">{''.join(layer_items)}</ul>
    </article>
    """


def render_model_assets() -> str:
    config = load_model_config()
    catalog_size = config.get("catalog_size")
    num_items = config.get("num_items")
    top_item_count = "N/A"
    mapped_count = "N/A"
    if TOP_ITEMS_PATH.exists():
        top_item_count = f"{len(pd.read_csv(TOP_ITEMS_PATH)):,}"
    if OLD_TO_NEW_MAP_PATH.exists():
        with OLD_TO_NEW_MAP_PATH.open(encoding="utf-8") as file:
            mapped_count = f"{len(json.load(file)):,}"

    return f"""
    <div class="model-grid">
      <article class="asset-card">
        <h3>Dữ liệu sản phẩm</h3>
        <p>Danh sách item phổ biến đã dùng khi train lại model trên Kaggle.</p>
        <dl>
          <dt>Top items</dt><dd>{top_item_count}</dd>
          <dt>Mapping</dt><dd>{mapped_count}</dd>
          <dt>Catalog size</dt><dd>{html.escape(str(catalog_size if catalog_size is not None else "N/A"))}</dd>
          <dt>Thư mục</dt><dd>{html.escape(str(MODEL_DIR))}</dd>
        </dl>
      </article>
      <article class="asset-card">
        <h3>Mục tiêu mô hình</h3>
        <p>Dự đoán item tiếp theo hoặc nhóm item ứng viên từ lịch sử mua hàng, phù hợp để trình bày như phần mở rộng recommender system.</p>
        <dl>
          <dt>Input</dt><dd>Chuỗi old_article_idx</dd>
          <dt>Output</dt><dd>Phân phối xác suất trên {html.escape(str(num_items if num_items is not None else "N/A"))} vị trí</dd>
          <dt>Metric</dt><dd>Top-K Accuracy</dd>
        </dl>
      </article>
    </div>
    """


def get_deep_learning_data() -> dict[str, object]:
    global _PRODUCT_CACHE
    if _PRODUCT_CACHE is not None:
        return _PRODUCT_CACHE

    top_items = pd.read_csv(TOP_ITEMS_PATH)
    top_items["old_article_idx"] = top_items["old_article_idx"].astype(int)
    top_items["new_item_idx"] = top_items["new_item_idx"].astype(int)

    with OLD_TO_NEW_MAP_PATH.open(encoding="utf-8") as file:
        old_to_new = {int(key): int(value) for key, value in json.load(file).items()}
    with NEW_TO_OLD_MAP_PATH.open(encoding="utf-8") as file:
        new_to_old = {int(key): int(value) for key, value in json.load(file).items()}

    item_by_new = top_items.set_index("new_item_idx").to_dict("index")

    _PRODUCT_CACHE = {
        "top_items": top_items,
        "old_to_new": old_to_new,
        "new_to_old": new_to_old,
        "item_by_new": item_by_new,
    }
    return _PRODUCT_CACHE


def get_transformer_custom_objects():
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="Custom")
    class AttentionPooling(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.score_dense = tf.keras.layers.Dense(1)

        def call(self, inputs, mask=None):
            scores = self.score_dense(inputs)
            scores = tf.squeeze(scores, axis=-1)
            weights = tf.nn.softmax(scores, axis=1)
            weights = tf.expand_dims(weights, axis=-1)
            return tf.reduce_sum(inputs * weights, axis=1)

        def get_config(self):
            return super().get_config()

    @tf.keras.utils.register_keras_serializable(package="Custom")
    class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.max_len = max_len
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False)
            self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

        def call(self, x):
            positions = tf.range(start=0, limit=self.max_len, delta=1)
            positions = self.pos_emb(positions)
            x = self.token_emb(x)
            return x + positions

        def get_config(self):
            config = super().get_config()
            config.update({"max_len": self.max_len, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
            return config

    @tf.keras.utils.register_keras_serializable(package="Custom")
    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.dropout_rate = dropout_rate
            self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
            self.ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ])
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
            self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        def call(self, inputs, training=False):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

        def get_config(self):
            config = super().get_config()
            config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate})
            return config

    return {
        "AttentionPooling": AttentionPooling,
        "Custom>AttentionPooling": AttentionPooling,
        "PositionalEmbedding": PositionalEmbedding,
        "Custom>PositionalEmbedding": PositionalEmbedding,
        "TransformerBlock": TransformerBlock,
        "Custom>TransformerBlock": TransformerBlock,
    }


def load_prediction_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    import tensorflow as tf

    if model_name == "lstm":
        model = tf.keras.models.load_model(
            configured_model_path("lstm"),
            custom_objects=get_transformer_custom_objects(),
            compile=False,
        )
    elif model_name == "transformer":
        model = tf.keras.models.load_model(
            configured_model_path("transformer"),
            custom_objects=get_transformer_custom_objects(),
            compile=False,
        )
    else:
        raise ValueError("Model không hợp lệ.")

    _MODEL_CACHE[model_name] = model
    return model


def parse_article_ids(raw_value: str) -> list[str]:
    tokens = raw_value.replace(",", " ").replace("\n", " ").split()
    return [token.strip() for token in tokens if token.strip()]


def pad_sequence(sequence: list[int]) -> list[int]:
    max_seq_len = int(load_model_config().get("max_seq_len", MAX_SEQ_LEN))
    trimmed = sequence[-max_seq_len:]
    return trimmed + [0] * (max_seq_len - len(trimmed))


def recommend_with_prediction_model(model_name: str, article_ids: list[str], top_k: int) -> dict[str, object]:
    import numpy as np

    data = get_deep_learning_data()
    old_to_new = data["old_to_new"]
    new_to_old = data["new_to_old"]
    item_by_new = data["item_by_new"]

    parsed_old_ids = []
    unknown = []
    for article_id in article_ids:
        try:
            old_article_idx = int(article_id)
        except ValueError:
            unknown.append(article_id)
            continue
        if old_article_idx not in old_to_new:
            unknown.append(article_id)
            continue
        parsed_old_ids.append(old_article_idx)

    input_sequence = [old_to_new[old_article_idx] for old_article_idx in parsed_old_ids]
    if not input_sequence:
        raise ValueError("Không có old_article_idx hợp lệ trong input.")

    model = load_prediction_model(model_name)
    input_padded = np.array([pad_sequence(input_sequence)], dtype="int32")
    started = time.perf_counter()
    predictions = model.predict(input_padded, verbose=0)[0]
    elapsed = time.perf_counter() - started

    blocked = set(input_sequence)
    candidate_indices = np.argsort(predictions)[::-1]
    recommendations = []
    for new_item_idx in candidate_indices:
        new_item_idx = int(new_item_idx)
        if new_item_idx == 0 or new_item_idx in blocked or new_item_idx not in new_to_old:
            continue
        item = item_by_new.get(new_item_idx, {})
        old_article_idx = int(new_to_old[new_item_idx])
        recommendations.append(
            {
                "rank": len(recommendations) + 1,
                "new_item_idx": new_item_idx,
                "old_article_idx": old_article_idx,
                "catalog_rank": int(item.get("rank", 0)) if item else None,
                "score": float(predictions[new_item_idx]),
            }
        )
        if len(recommendations) >= top_k:
            break

    return {
        "model": model_name,
        "input_sequence": input_sequence,
        "input_old_article_indices": parsed_old_ids,
        "unknown_article_ids": unknown,
        "elapsed_seconds": elapsed,
        "recommendations": recommendations,
    }


def render_recommendation_result(results: list[dict[str, object]], article_ids: list[str]) -> str:
    sections = []
    for result in results:
        rows = []
        for item in result["recommendations"]:
            rows.append(
                "<tr>"
                f"<td>{item['rank']}</td>"
                f"<td>{item['old_article_idx']}</td>"
                f"<td>{item['new_item_idx']}</td>"
                f"<td>{item['catalog_rank'] or 'N/A'}</td>"
                f"<td>{item['score']:.6f}</td>"
                "</tr>"
            )

        unknown_html = ""
        if result["unknown_article_ids"]:
            unknown_html = (
                "<p>Không tìm thấy trong mapping: "
                f"<code>{html.escape(', '.join(result['unknown_article_ids']))}</code></p>"
            )

        sections.append(
            f"""
            <div class="recommendations">
              <h2>{html.escape(str(result["model"]).upper())} - Top sản phẩm gợi ý</h2>
              <p>Thời gian predict: {result['elapsed_seconds']:.4f}s. Input hợp lệ: {len(result['input_sequence'])} item.</p>
              {unknown_html}
              <div class="table-wrap recommendation-table">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>old_article_idx</th>
                      <th>new_item_idx</th>
                      <th>Rank trong top_items</th>
                      <th>Score</th>
                    </tr>
                  </thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
              </div>
            </div>
            """
        )

    return f"""
    <div class="insights">
      <h2>Kết quả chạy model thật</h2>
      <ul>
        <li>Chuỗi đầu vào: {html.escape(', '.join(article_ids))}</li>
        <li>Backend đã chuyển old_article_idx sang new_item_idx, padding theo <code>model_config.json</code>, rồi gọi trực tiếp <code>model.predict()</code>.</li>
      </ul>
    </div>
    {''.join(sections)}
    """


def render_model_demo_form(values: dict[str, str], result_html: str = "") -> str:
    data = get_deep_learning_data()
    top_items = data["top_items"].head(6)
    sample_items = []
    for row in top_items.itertuples(index=False):
        sample_items.append(
            f"""
            <div class="sample-product">
              <strong>old_article_idx {html.escape(str(row.old_article_idx))}</strong>
              <span>new_item_idx {html.escape(str(row.new_item_idx))} - rank {html.escape(str(row.rank))}</span>
              <button type="button" data-article-id="{html.escape(str(row.old_article_idx))}">Thêm vào input</button>
            </div>
            """
        )

    model_options = [
        ("both", "Chạy cả LSTM và Transformer"),
        ("lstm", "Chỉ LSTM"),
        ("transformer", "Chỉ Transformer"),
    ]
    top_k_options = [("5", "Top 5"), ("10", "Top 10"), ("20", "Top 20")]
    model_select = "".join(
        f'<option value="{value}" {"selected" if values["model"] == value else ""}>{label}</option>'
        for value, label in model_options
    )
    top_k_select = "".join(
        f'<option value="{value}" {"selected" if values["top_k"] == value else ""}>{label}</option>'
        for value, label in top_k_options
    )

    return f"""
    <div class="model-demo">
      <form action="/models/run" method="get">
        <label>
          <span>Chuỗi old_article_idx đã mua/đã xem</span>
          <textarea name="article_ids" id="article_ids" placeholder="Ví dụ: 507, 712, 290">{html.escape(values["article_ids"])}</textarea>
          <small>Nhập nhiều old_article_idx, cách nhau bằng dấu phẩy, khoảng trắng hoặc xuống dòng. Model chỉ dùng tối đa {int(load_model_config().get("max_seq_len", MAX_SEQ_LEN))} item cuối.</small>
        </label>
        <label>
          <span>Model</span>
          <select name="model">{model_select}</select>
        </label>
        <label>
          <span>Số lượng gợi ý</span>
          <select name="top_k">{top_k_select}</select>
        </label>
        <div class="actions">
          <button type="submit">Chạy model</button>
          <a class="button secondary" href="/models">Làm mới</a>
        </div>
        <div class="sample-products">
          {''.join(sample_items)}
        </div>
      </form>
      <div>
        {result_html or '<div class="empty-state"><p>Nhập chuỗi old_article_idx rồi chạy LSTM/Transformer để xem gợi ý sản phẩm tiếp theo.</p></div>'}
      </div>
    </div>
    """


def model_values_from_query(query: dict[str, list[str]] | None) -> dict[str, str]:
    if query is None:
        return {"article_ids": "", "model": "both", "top_k": "10"}
    return {
        "article_ids": query.get("article_ids", [""])[0],
        "model": query.get("model", ["both"])[0],
        "top_k": query.get("top_k", ["10"])[0],
    }


def run_model_query(query: dict[str, list[str]]) -> tuple[dict[str, str], str]:
    values = model_values_from_query(query)
    article_ids = parse_article_ids(values["article_ids"])
    if not article_ids:
        raise ValueError("Bạn cần nhập ít nhất một article_id.")

    top_k = parse_int(values["top_k"], "top_k")
    top_k = min(max(top_k, 1), 20)
    model_choice = values["model"]
    if model_choice == "both":
        model_names = ["lstm", "transformer"]
    elif model_choice in {"lstm", "transformer"}:
        model_names = [model_choice]
    else:
        raise ValueError("Model không hợp lệ.")

    results = [
        recommend_with_prediction_model(model_name, article_ids, top_k)
        for model_name in model_names
    ]
    return values, render_recommendation_result(results, article_ids)


def render_models_page(result_html: str = "", values: dict[str, str] | None = None) -> str:
    comparison = load_model_comparison()
    lstm_info = inspect_keras_model(configured_model_path("lstm"))
    transformer_info = inspect_keras_model(configured_model_path("transformer"))
    if values is None:
        values = model_values_from_query(None)
    comparison_table = ""
    if comparison:
        table_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{metric_or_na(row.get('loss'), lambda value: f'{value:.4f}')}</td>"
            f"<td>{metric_or_na(row.get('top1'))}</td>"
            f"<td>{metric_or_na(row.get('top3'))}</td>"
            f"<td>{metric_or_na(row.get('top5'))}</td>"
            f"<td>{metric_or_na(row.get('top10'))}</td>"
            f"<td>{metric_or_na(row.get('top20'))}</td>"
            "</tr>"
            for row in comparison
        )
        comparison_table = f"""
        <h2>Bảng metric</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Loss</th>
                <th>Top-1</th>
                <th>Top-3</th>
                <th>Top-5</th>
                <th>Top-10</th>
                <th>Top-20</th>
              </tr>
            </thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Biểu diễn mô hình</title>
  <style>
{APP_STYLES}
  </style>
</head>
<body>
  <main>
    <header class="app-header">
      <div>
        <p class="eyebrow">Model showcase</p>
        <h1>Biểu diễn mô hình LSTM và Transformer</h1>
        <p class="subtitle">Trang bổ sung để trình bày hướng sequence modeling/recommender system bên cạnh phần khai phá frequent itemsets.</p>
      </div>
      <nav class="nav-pills" aria-label="Điều hướng">
        <a href="/">Benchmark</a>
        <a href="/patterns">Gợi ý combo</a>
        <a class="active" href="/models">Mô hình</a>
      </nav>
    </header>
    <section class="result">
      <h2>Demo gợi ý sản phẩm bằng model thật</h2>
      {render_model_demo_form(values, result_html)}
      {render_model_cards(comparison)}
      {render_metric_bars(comparison) if comparison else ""}
      {render_metric_heatmap(comparison) if comparison else ""}
      {render_training_charts()}
      {comparison_table}
      <h2>Kiến trúc model</h2>
      <div class="model-grid">
        {render_architecture_card("LSTM", lstm_info)}
        {render_architecture_card("Transformer", transformer_info)}
      </div>
      <h2>Dữ liệu và cách trình bày</h2>
      {render_model_assets()}
    </section>
  </main>
  <script>
    const articleInput = document.querySelector("#article_ids");
    document.querySelectorAll("[data-article-id]").forEach((button) => {{
      button.addEventListener("click", () => {{
        const current = articleInput.value.trim();
        articleInput.value = current ? `${{current}}, ${{button.dataset.articleId}}` : button.dataset.articleId;
        articleInput.focus();
      }});
    }});
  </script>
</body>
</html>"""


APP_STYLES = """
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-soft: #f9fafb;
      --ink: #17202a;
      --muted: #687386;
      --line: #d9dee7;
      --line-strong: #c5ccd8;
      --accent: #156f6d;
      --accent-dark: #0f5655;
      --accent-soft: #e7f4f2;
      --amber: #c96f19;
      --amber-soft: #fff3e5;
      --danger: #b53c32;
      --shadow: 0 18px 45px rgba(23, 32, 42, 0.08);
      --radius: 8px;
    }
    * { box-sizing: border-box; }
    html { min-height: 100%; }
    body {
      margin: 0;
      min-height: 100%;
      background:
        linear-gradient(180deg, rgba(21, 111, 109, 0.08), rgba(21, 111, 109, 0) 320px),
        var(--bg);
      color: var(--ink);
      font-family: Inter, "Segoe UI", Tahoma, Arial, sans-serif;
      font-size: 15px;
      line-height: 1.5;
    }
    main {
      width: min(1240px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }
    h1, h2, h3, p { letter-spacing: 0; }
    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.15;
      font-weight: 800;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 19px;
      line-height: 1.25;
      font-weight: 750;
    }
    h3 {
      margin: 0 0 8px;
      font-size: 16px;
      line-height: 1.35;
    }
    p {
      margin: 0 0 16px;
      color: var(--muted);
    }
    a { color: var(--accent-dark); font-weight: 700; }
    code {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface-soft);
      padding: 2px 6px;
      font-size: 13px;
    }
    .app-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 22px;
      padding: 20px 0 4px;
    }
    .eyebrow {
      margin: 0 0 8px;
      color: var(--accent-dark);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .subtitle {
      max-width: 760px;
      margin: 8px 0 0;
      font-size: 15px;
    }
    .nav-pills {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }
    .nav-pills a {
      display: inline-flex;
      align-items: center;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
      padding: 8px 13px;
      text-decoration: none;
      box-shadow: 0 8px 22px rgba(23, 32, 42, 0.05);
    }
    .nav-pills a.active {
      border-color: rgba(21, 111, 109, 0.25);
      background: var(--accent-soft);
      color: var(--accent-dark);
    }
    .layout {
      display: grid;
      grid-template-columns: 380px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    form, .result, .history-panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    form {
      padding: 18px;
      position: sticky;
      top: 18px;
    }
    label {
      display: block;
      margin-bottom: 13px;
    }
    label span {
      display: block;
      margin-bottom: 5px;
      color: #243041;
      font-size: 13px;
      font-weight: 750;
    }
    input[type="text"], input[type="number"], select {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      outline: none;
      transition: border-color .15s ease, box-shadow .15s ease;
    }
    input[type="text"]:focus, input[type="number"]:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(21, 111, 109, 0.14);
    }
    small {
      display: block;
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .checks {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin: 4px 0 16px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 10px;
      background: var(--surface-soft);
    }
    .check input { accent-color: var(--accent); }
    .check span {
      display: inline;
      margin: 0;
      font-weight: 750;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding-top: 3px;
    }
    button, a.button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      border: 0;
      border-radius: 7px;
      background: var(--accent);
      color: #fff;
      padding: 10px 14px;
      font: inherit;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
      box-shadow: 0 10px 22px rgba(21, 111, 109, 0.2);
    }
    button:hover, a.button:hover { background: var(--accent-dark); }
    a.button.secondary {
      border: 1px solid var(--line-strong);
      background: #fff;
      color: var(--ink);
      box-shadow: none;
    }
    a.button.secondary:hover { background: var(--surface-soft); }
    button.danger {
      background: var(--danger);
      box-shadow: none;
    }
    .inline-form {
      padding: 0;
      position: static;
      background: transparent;
      border: 0;
      box-shadow: none;
    }
    .result {
      min-height: 420px;
      padding: 20px;
      overflow: hidden;
    }
    .empty-state {
      display: grid;
      place-items: center;
      min-height: 340px;
      border: 1px dashed var(--line-strong);
      border-radius: var(--radius);
      background: var(--surface-soft);
      padding: 28px;
      text-align: center;
      color: var(--muted);
    }
    .result img {
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: white;
    }
    pre {
      overflow: auto;
      max-height: 360px;
      border-radius: var(--radius);
      background: #17202a;
      color: #eff6f5;
      padding: 14px;
      line-height: 1.45;
      font-size: 13px;
    }
    .error {
      border: 1px solid #f0c7c1;
      border-left: 4px solid var(--danger);
      border-radius: var(--radius);
      background: #fff5f3;
      padding: 12px;
      color: #7b251d;
    }
    .insights {
      margin: 16px 0;
      border: 1px solid #cfe3df;
      border-radius: var(--radius);
      background: var(--accent-soft);
      padding: 14px;
    }
    .insights ul {
      margin: 0;
      padding-left: 20px;
      color: var(--ink);
      line-height: 1.58;
    }
    .insights li { margin: 5px 0; }
    .tables {
      display: grid;
      gap: 18px;
      margin-top: 18px;
    }
    .tables > div {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 18px;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
    }
    th, td {
      padding: 10px 11px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }
    th {
      position: sticky;
      top: 0;
      background: #f1f5f6;
      color: #253142;
      font-weight: 800;
      white-space: nowrap;
    }
    tbody tr:nth-child(even) td { background: #fbfcfd; }
    tbody tr:hover td { background: #f5fbfa; }
    .section-heading {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }
    .section-heading h2 { margin: 0; }
    .recommendations { margin-bottom: 18px; }
    .recommendations > p { margin-bottom: 12px; }
    .combo-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .combo-card {
      border: 1px solid #ead8c2;
      border-radius: var(--radius);
      background: var(--amber-soft);
      padding: 14px;
    }
    .combo-rank {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      background: #fff;
      color: var(--amber);
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 850;
      margin-bottom: 9px;
    }
    .combo-card p { margin: 0 0 8px; }
    .combo-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .combo-meta span {
      border: 1px solid #cfe3df;
      border-radius: 999px;
      background: #fff;
      color: #263443;
      padding: 5px 8px;
      font-size: 12px;
      font-weight: 750;
    }
    .model-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .model-card, .metric-card, .asset-card {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface-soft);
      padding: 14px;
    }
    .model-card h3, .metric-card h3, .asset-card h3 { margin-bottom: 4px; }
    .model-card dl, .metric-card dl, .asset-card dl {
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr);
      gap: 7px 10px;
      margin: 12px 0 0;
      font-size: 13px;
    }
    .model-card dt, .metric-card dt, .asset-card dt {
      color: var(--muted);
      font-weight: 750;
    }
    .model-card dd, .metric-card dd, .asset-card dd {
      margin: 0;
      min-width: 0;
      word-break: break-word;
    }
    .layer-list {
      display: grid;
      gap: 6px;
      margin: 12px 0 0;
      padding: 0;
      list-style: none;
    }
    .layer-list li {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      padding: 7px 9px;
      font-size: 13px;
    }
    .layer-list span:last-child {
      color: var(--muted);
      text-align: right;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin: 12px 0 18px;
    }
    .metric-value {
      margin: 8px 0 10px;
      color: var(--ink);
      font-size: 26px;
      line-height: 1;
      font-weight: 850;
    }
    .bar-list {
      display: grid;
      gap: 11px;
      margin: 14px 0 0;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 130px minmax(0, 1fr) 70px;
      align-items: center;
      gap: 10px;
      font-size: 13px;
    }
    .bar-track {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #e7ecf1;
    }
    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
    }
    .bar-fill.transformer { background: var(--amber); }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin: 12px 0 18px;
    }
    .chart-card {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fff;
      padding: 14px;
    }
    .chart-card h3 { margin-bottom: 10px; }
    .chart-card svg {
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }
    .chart-card line {
      stroke: #d8e0e6;
      stroke-width: 1;
    }
    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
    }
    .chart-legend span {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .chart-legend i {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }
    .heatmap-wrap { margin: 12px 0 8px; }
    .heatmap th {
      text-align: left;
      white-space: nowrap;
      background: #f7f9fb;
    }
    .heatmap td {
      background: color-mix(in srgb, var(--accent) calc(22% + var(--heat) * 58%), #ffffff);
      min-width: 110px;
    }
    .heatmap td strong {
      display: block;
      color: var(--ink);
      font-size: 15px;
    }
    .heatmap td span {
      display: block;
      margin-top: 2px;
      color: #41505f;
      font-size: 12px;
      font-weight: 700;
    }
    .note {
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 13px;
    }
    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      outline: none;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(21, 111, 109, 0.14);
    }
    .model-demo {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
      margin-bottom: 20px;
    }
    .model-demo form { position: static; box-shadow: none; }
    .recommendation-table td:nth-child(3) {
      min-width: 220px;
      white-space: normal;
    }
    .sample-products {
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }
    .sample-product {
      display: grid;
      gap: 3px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      padding: 8px 9px;
      font-size: 12px;
    }
    .sample-product strong { color: var(--ink); }
    .sample-product button {
      justify-self: start;
      min-height: 30px;
      padding: 5px 9px;
      box-shadow: none;
      font-size: 12px;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .model-demo { grid-template-columns: 1fr; }
      form { position: static; }
      .app-header { grid-template-columns: 1fr; }
      .nav-pills { justify-content: flex-start; }
    }
    @media (max-width: 640px) {
      main { width: min(100vw - 20px, 1240px); margin-top: 14px; }
      h1 { font-size: 24px; }
      form, .result, .tables > div { padding: 14px; }
      .checks, .combo-grid, .model-grid, .metric-grid, .chart-grid { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 1fr; }
      .actions > * { width: 100%; }
      th, td { padding: 9px; }
    }
"""


def render_form(values: dict[str, object], result_html: str = "") -> str:
    selected = set(values["algorithms"])

    def input_field(label: str, name: str, help_text: str = "", input_type: str = "text") -> str:
        value = html.escape(str(values[name]))
        return f"""
        <label>
          <span>{html.escape(label)}</span>
          <input type="{input_type}" name="{name}" value="{value}">
          <small>{html.escape(help_text)}</small>
        </label>
        """

    def select_field(label: str, name: str, help_text: str = "") -> str:
        current = str(values[name])
        options = "\n".join(
            f'<option value="{html.escape(value)}" {"selected" if value == current else ""}>'
            f'{html.escape(text)}</option>'
            for value, text in SELECT_OPTIONS[name]
        )
        return f"""
        <label>
          <span>{html.escape(label)}</span>
          <select name="{name}">{options}</select>
          <small>{html.escape(help_text)}</small>
        </label>
        """

    def preset_field() -> str:
        current = str(values["preset"])
        options = "\n".join(
            f'<option value="{html.escape(key)}" {"selected" if key == current else ""}>'
            f'{html.escape(config["label"])}</option>'
            for key, config in PRESETS.items()
        )
        return f"""
        <label>
          <span>Preset cấu hình</span>
          <select name="preset" id="preset">{options}</select>
          <small>Chọn nhanh bộ tham số thường dùng. Có thể chỉnh lại từng dropdown sau đó.</small>
        </label>
        """

    def basket_dataset_field() -> str:
        current = str(values["basket_dataset"])
        options = "\n".join(
            f'<option value="{html.escape(str(dataset["key"]))}" '
            f'{"selected" if str(dataset["key"]) == current else ""}>'
            f'{html.escape(str(dataset["label"]))}</option>'
            for dataset in load_basket_datasets()
        )
        return f"""
        <label>
          <span>Loại basket</span>
          <select name="basket_dataset" id="basket_dataset">{options}</select>
          <small>Chọn basket gốc hoặc các biến thể đã sinh trong processed/basket_variants.</small>
        </label>
        """

    algorithm_checks = "\n".join(
        f"""
        <label class="check">
          <input type="checkbox" name="algorithms" value="{name}" {"checked" if name in selected else ""}>
          <span>{name}</span>
        </label>
        """
        for name in DEFAULT_ALGORITHMS
    )

    history_html = render_history_table(load_history())
    presets_json = json.dumps(PRESETS)
    basket_datasets_json = json.dumps(load_basket_datasets(), ensure_ascii=False)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Giao diện so sánh thuật toán</title>
  <style>
{APP_STYLES}
  </style>
</head>
<body>
  <main>
    <header class="app-header">
      <div>
        <p class="eyebrow">Frequent itemset mining</p>
        <h1>So sánh thuật toán</h1>
        <p class="subtitle">Chạy Apriori, FP-growth và Eclat trên các basket dataset, rồi theo dõi thời gian chạy, bộ nhớ đỉnh và số itemset tìm được.</p>
      </div>
      <nav class="nav-pills" aria-label="Điều hướng">
        <a class="active" href="/">Benchmark</a>
        <a href="/patterns">Gợi ý combo</a>
        <a href="/models">Mô hình</a>
      </nav>
    </header>
    <div class="layout">
      <form action="/run" method="get">
        {basket_dataset_field()}
        {input_field("File CSV giỏ hàng", "baskets", "Chỉ dùng khi chọn Loại basket = Custom path.")}
        {input_field("Ảnh đầu ra", "output", "Để trống để tự tạo tên ảnh trong results/ui_charts/.")}
        {preset_field()}
        <label>
          <span>Thuật toán</span>
          <div class="checks">{algorithm_checks}</div>
        </label>
        {select_field("Số giỏ hàng tối đa", "max_baskets", "Chọn lượng basket dùng để benchmark.")}
        {select_field("Số mặt hàng lấy top", "top_items", "Giới hạn số sản phẩm phổ biến nhất để tránh chạy quá nặng.")}
        {select_field("Cỡ basket đầu vào tối thiểu", "min_items", "Lọc dữ liệu đầu vào: chỉ giữ basket có ít nhất số item này.")}
        {input_field("Độ hỗ trợ tối thiểu (Min Support)", "min_support", "Nhập số trong khoảng 0-1. Ví dụ: 0.5 = 50%, 0.05 = 5%, 0.005 = 0.5%.")}
        {select_field("Độ dài tập tối đa (Max Length)", "max_len", "Giới hạn độ dài itemset cần tìm.")}
        {select_field("Khoảng thời gian", "date_range", "Chọn sẵn khoảng ngày để lọc dữ liệu.")}
        <div class="actions">
          <button type="submit">Chạy và Vẽ biểu đồ</button>
          <a class="button secondary" href="/">Làm mới</a>
        </div>
      </form>
      <section class="result">
        {result_html or '<div class="empty-state"><p>Chọn cấu hình ở bên trái rồi chạy benchmark để tạo biểu đồ so sánh.</p></div>'}
      </section>
    </div>
    <section class="tables">
      {history_html}
    </section>
  </main>
  <script>
    const presets = {presets_json};
    const basketDatasets = {basket_datasets_json};
    const presetSelect = document.querySelector("#preset");
    const basketDatasetSelect = document.querySelector("#basket_dataset");
    const basketsInput = document.querySelector('[name="baskets"]');
    function syncBasketPath() {{
      const selected = basketDatasets.find((dataset) => dataset.key === basketDatasetSelect.value);
      if (selected && selected.key !== "custom" && selected.path) basketsInput.value = selected.path;
    }}
    basketDatasetSelect.addEventListener("change", syncBasketPath);
    syncBasketPath();
    presetSelect.addEventListener("change", () => {{
      const selected = presets[presetSelect.value];
      if (!selected) return;
      for (const field of ["max_baskets", "top_items", "min_items", "min_support", "max_len", "date_range"]) {{
        const input = document.querySelector(`[name="${{field}}"]`);
        if (input && selected[field] !== undefined) input.value = selected[field];
      }}
    }});
  </script>
</body>
</html>"""


def render_result(args: SimpleNamespace, summary: dict[str, object], run_id: str) -> str:
    image_path = Path(args.output)
    image_url = f"/image?path={html.escape(str(image_path))}&t={int(time.time())}"
    escaped_summary = html.escape(json.dumps(summary, indent=2))
    return f"""
      <h2>Biểu đồ</h2>
      <p>Run ID: <code>{html.escape(run_id)}</code>. Đã lưu ảnh: <code>{html.escape(str(image_path))}</code></p>
      <img src="{image_url}" alt="Biểu đồ so sánh thuật toán">
      {render_insights(args, summary)}
      {render_metrics_table(summary)}
      <h2>JSON tóm tắt</h2>
      <pre>{escaped_summary}</pre>
    """


def render_patterns_page(values: dict[str, object], result_html: str = "") -> str:
    def input_field(label: str, name: str, help_text: str = "") -> str:
        return f"""
        <label>
          <span>{html.escape(label)}</span>
          <input type="text" name="{name}" value="{html.escape(str(values[name]))}">
          <small>{html.escape(help_text)}</small>
        </label>
        """

    def select_from_options(label: str, name: str, options: list[tuple[str, str]], help_text: str = "") -> str:
        current = str(values[name])
        rendered_options = "\n".join(
            f'<option value="{html.escape(value)}" {"selected" if value == current else ""}>'
            f"{html.escape(text)}</option>"
            for value, text in options
        )
        return f"""
        <label>
          <span>{html.escape(label)}</span>
          <select name="{name}">{rendered_options}</select>
          <small>{html.escape(help_text)}</small>
        </label>
        """

    basket_options = [
        (str(dataset["key"]), str(dataset["label"]))
        for dataset in load_basket_datasets()
    ]
    algorithm_options = [(name, name) for name in DEFAULT_ALGORITHMS]
    basket_datasets_json = json.dumps(load_basket_datasets(), ensure_ascii=False)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Xem frequent patterns</title>
  <style>
{APP_STYLES}
  </style>
</head>
<body>
  <main>
    <header class="app-header">
      <div>
        <p class="eyebrow">Pattern explorer</p>
        <h1>Gợi ý combo quần áo</h1>
        <p class="subtitle">Biến frequent itemsets và association rules thành các combo, nhóm đồ và gợi ý mua kèm dễ đọc hơn.</p>
      </div>
      <nav class="nav-pills" aria-label="Điều hướng">
        <a href="/">Benchmark</a>
        <a class="active" href="/patterns">Gợi ý combo</a>
        <a href="/models">Mô hình</a>
      </nav>
    </header>
    <div class="layout">
      <form action="/patterns/run" method="get">
        {select_from_options("Loại basket", "basket_dataset", basket_options, "Chọn basket gốc, theo tuổi, theo năm hoặc theo category.")}
        {input_field("File CSV giỏ hàng", "baskets", "Chỉ dùng khi chọn Loại basket = Custom path.")}
        {select_from_options("Chế độ hiển thị", "display_mode", SELECT_OPTIONS["display_mode"], "Luật gợi ý dùng confidence/lift; Combo phổ biến dùng support.")}
        {select_from_options("Thuật toán", "algorithm", algorithm_options, "Dùng FP-growth mặc định để xem pattern nhanh.")}
        {select_from_options("Số giỏ hàng tối đa", "max_baskets", SELECT_OPTIONS["max_baskets"])}
        {select_from_options("Số mặt hàng lấy top", "top_items", SELECT_OPTIONS["top_items"])}
        {select_from_options("Cỡ basket đầu vào tối thiểu", "min_items", SELECT_OPTIONS["min_items"], "Lọc dữ liệu đầu vào, không phải độ dài combo kết quả.")}
        {select_from_options("Độ dài combo tối thiểu", "min_itemset_length", SELECT_OPTIONS["min_itemset_length"], "Lọc kết quả hiển thị. Chọn 3 thì không hiện combo 2 item.")}
        {input_field("Min Support", "min_support", "Nhập số trong khoảng 0-1. Ví dụ: 0.5 = 50%, 0.05 = 5%, 0.005 = 0.5%.")}
        {input_field("Min Confidence", "min_confidence", "Chỉ dùng cho Luật gợi ý. Ví dụ: 0.3 = 30%, 0.7 = 70%.")}
        {select_from_options("Sắp xếp luật theo", "sort_metric", SELECT_OPTIONS["sort_metric"], "Lift đo độ liên quan, confidence đo xác suất gợi ý.")}
        {select_from_options("Max Length", "max_len", SELECT_OPTIONS["max_len"])}
        {select_from_options("Khoảng thời gian", "date_range", SELECT_OPTIONS["date_range"])}
        {select_from_options("Số dòng kết quả", "top_results", SELECT_OPTIONS["top_results"])}
        <div class="actions">
          <button type="submit">Xem gợi ý</button>
          <a class="button secondary" href="/patterns">Làm mới</a>
        </div>
      </form>
      <section class="result">
        {result_html or '<div class="empty-state"><p>Chọn loại basket và chạy phân tích để hiển thị các combo dễ đọc.</p></div>'}
      </section>
    </div>
  </main>
  <script>
    const basketDatasets = {basket_datasets_json};
    const basketDatasetSelect = document.querySelector("#basket_dataset, [name='basket_dataset']");
    const basketsInput = document.querySelector('[name="baskets"]');
    function syncBasketPath() {{
      const selected = basketDatasets.find((dataset) => dataset.key === basketDatasetSelect.value);
      if (selected && selected.key !== "custom" && selected.path) basketsInput.value = selected.path;
    }}
    basketDatasetSelect.addEventListener("change", syncBasketPath);
    syncBasketPath();
  </script>
</body>
</html>"""


class AlgorithmUIHandler(BaseHTTPRequestHandler):
    def send_html(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/":
            self.send_html(render_form(values_from_args(None)))
            return

        if parsed.path == "/patterns":
            self.send_html(render_patterns_page(pattern_values_from_args(None)))
            return

        if parsed.path == "/models":
            self.send_html(render_models_page())
            return

        if parsed.path == "/models/run":
            values = model_values_from_query(query)
            try:
                values, result_html = run_model_query(query)
                self.send_html(render_models_page(result_html, values))
            except Exception as exc:
                message = f"<div class='error'><strong>Lỗi:</strong> {html.escape(str(exc))}</div>"
                self.send_html(render_models_page(message, values), status=400)
            return

        if parsed.path == "/run":
            args = None
            try:
                args = params_from_query(query)
                summary, _ = run_algorithms(args)
                plot_comparison(summary, Path(args.output))
                run_id = append_history(args, summary)
                self.send_html(render_form(values_from_args(args), render_result(args, summary, run_id)))
            except Exception as exc:
                values = values_from_args(args) if args is not None else values_from_args(None)
                message = f"<div class='error'><strong>Lỗi:</strong> {html.escape(str(exc))}</div>"
                self.send_html(render_form(values, message), status=400)
            return

        if parsed.path == "/patterns/run":
            args = None
            try:
                args = pattern_params_from_query(query)
                summary, itemsets, rules = run_pattern_mining(args)
                self.send_html(
                    render_patterns_page(
                        pattern_values_from_args(args),
                        render_pattern_result(args, summary, itemsets, rules),
                    )
                )
            except Exception as exc:
                values = pattern_values_from_args(args) if args is not None else pattern_values_from_args(None)
                message = f"<div class='error'><strong>Lỗi:</strong> {html.escape(str(exc))}</div>"
                self.send_html(render_patterns_page(values, message), status=400)
            return

        if parsed.path == "/image":
            image_path = Path(query.get("path", [""])[0])
            if not image_path.exists() or image_path.suffix.lower() != ".png":
                self.send_error(404)
                return

            data = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/clear-history":
            clear_history()
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            return

        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a browser UI for algorithm comparison.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open the UI in the default browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AlgorithmUIHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Open {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
