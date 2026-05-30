import argparse
import csv
import html
import json
import sys
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

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
    :root {{
      --bg: #eef2f0;
      --panel: #ffffff;
      --ink: #17211d;
      --muted: #66736d;
      --line: #cfd8d3;
      --accent: #2f6f64;
      --accent-2: #d96f32;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #eef2f0 0%, #dde8e4 55%, #f2e8dd 100%);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 20px;
      color: var(--muted);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 390px 1fr;
      gap: 18px;
      align-items: start;
    }}
    form, .result {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(23, 33, 29, 0.08);
    }}
    form {{
      padding: 18px;
      position: sticky;
      top: 18px;
    }}
    label {{
      display: block;
      margin-bottom: 14px;
    }}
    label span {{
      display: block;
      font-weight: 650;
      margin-bottom: 5px;
    }}
    input[type="text"], input[type="number"], select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      background: #fbfcfb;
    }}
    small {{
      display: block;
      color: var(--muted);
      margin-top: 4px;
      line-height: 1.35;
    }}
    .checks {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      margin: 4px 0 16px;
    }}
    .check {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      display: flex;
      align-items: center;
      gap: 8px;
      background: #fbfcfb;
    }}
    .check span {{
      display: inline;
      margin: 0;
      font-weight: 600;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    button, a.button {{
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
      font: inherit;
    }}
    a.button.secondary {{
      background: var(--accent-2);
    }}
    button.danger {{
      background: #b94033;
    }}
    .inline-form {{
      padding: 0;
      position: static;
      background: transparent;
      border: 0;
      box-shadow: none;
    }}
    .result {{
      padding: 18px;
      min-height: 420px;
    }}
    .result img {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
    }}
    pre {{
      overflow: auto;
      background: #17211d;
      color: #eff8f4;
      padding: 14px;
      border-radius: 6px;
      line-height: 1.45;
    }}
    .error {{
      border-left: 4px solid #c63d2f;
      background: #fff4f1;
      padding: 12px;
      border-radius: 6px;
      color: #7b251d;
    }}
    .insights {{
      background: #f4faf7;
      border: 1px solid #cfe0d8;
      border-radius: 8px;
      padding: 14px;
      margin: 16px 0;
    }}
    .insights ul {{
      margin: 0;
      padding-left: 20px;
      color: var(--ink);
      line-height: 1.55;
    }}
    .insights li {{
      margin: 6px 0;
    }}
    .tables {{
      margin-top: 18px;
      display: grid;
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(23, 33, 29, 0.06);
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 13px;
      white-space: nowrap;
    }}
    th {{
      background: #eef5f2;
      font-weight: 750;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 20px;
    }}
    .section-heading {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .section-heading h2 {{
      margin: 0;
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      form {{ position: static; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Giao diện so sánh thuật toán</h1>
    <p>Điều chỉnh tham số, chạy Apriori, FP-growth, và Eclat, sau đó so sánh thời gian chạy và bộ nhớ đỉnh.</p>
    <p><a href="/patterns">Mở trang xem pattern dễ đọc theo loại basket</a></p>
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
        {result_html or "<p>Chạy một cấu hình để tạo biểu đồ so sánh.</p>"}
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
    :root {{
      --bg: #eef2f0;
      --panel: #ffffff;
      --ink: #17211d;
      --muted: #66736d;
      --line: #cfd8d3;
      --accent: #2f6f64;
      --accent-2: #d96f32;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #eef2f0 0%, #dde8e4 55%, #f2e8dd 100%);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 28px auto; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    p {{ margin: 0 0 18px; color: var(--muted); }}
    .layout {{ display: grid; grid-template-columns: 390px 1fr; gap: 18px; align-items: start; }}
    form, .result {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 10px 30px rgba(23,33,29,.08); }}
    form {{ padding: 18px; position: sticky; top: 18px; }}
    label {{ display: block; margin-bottom: 14px; }}
    label span {{ display: block; font-weight: 650; margin-bottom: 5px; }}
    input[type="text"], select {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 10px 11px; font: inherit; background: #fbfcfb; }}
    small {{ display: block; color: var(--muted); margin-top: 4px; line-height: 1.35; }}
    button, a.button {{ border: 0; border-radius: 6px; padding: 11px 14px; background: var(--accent); color: white; font-weight: 700; text-decoration: none; cursor: pointer; font: inherit; }}
    a.button.secondary {{ background: var(--accent-2); }}
    .actions {{ display: flex; gap: 10px; align-items: center; }}
    .result {{ padding: 18px; min-height: 420px; }}
    .insights {{ background: #f4faf7; border: 1px solid #cfe0d8; border-radius: 8px; padding: 14px; margin-bottom: 16px; }}
    .insights ul {{ margin: 0; padding-left: 20px; line-height: 1.55; }}
    .recommendations {{ margin-bottom: 18px; }}
    .recommendations > p {{ margin-bottom: 12px; }}
    .combo-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .combo-card {{ background: #fffdf8; border: 1px solid #e5d8c5; border-radius: 8px; padding: 14px; }}
    .combo-rank {{ color: var(--accent-2); font-weight: 800; margin-bottom: 6px; }}
    .combo-card h3 {{ margin: 0 0 8px; font-size: 17px; line-height: 1.35; }}
    .combo-card p {{ margin: 0 0 8px; color: var(--muted); }}
    .combo-meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .combo-meta span {{ background: #eef5f2; border: 1px solid #cfe0d8; border-radius: 999px; padding: 5px 8px; font-size: 12px; font-weight: 700; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #eef5f2; font-weight: 750; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    .error {{ border-left: 4px solid #c63d2f; background: #fff4f1; padding: 12px; border-radius: 6px; color: #7b251d; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} form {{ position: static; }} .combo-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Trang gợi ý combo quần áo</h1>
    <p>Chọn loại basket để biến frequent itemsets thành các gợi ý mua kèm, nhóm đồ hoặc phối đồ dễ giải thích.</p>
    <p><a href="/">Quay lại trang so sánh thuật toán</a></p>
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
        {result_html or "<p>Chọn loại basket và bấm Xem gợi ý để hiển thị các combo dễ đọc.</p>"}
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
