import argparse
import html
import json
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from plot_algorithm_comparison import DEFAULT_ALGORITHMS, plot_comparison, run_algorithms


DEFAULTS = {
    "baskets": "processed/baskets.csv",
    "output": "results/ui_algorithm_comparison.png",
    "max_baskets": "50000",
    "top_items": "500",
    "min_items": "2",
    "min_support": "0.01",
    "max_len": "3",
    "start_date": "",
    "end_date": "",
    "algorithms": DEFAULT_ALGORITHMS,
}


def parse_int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer.") from exc


def parse_float(value: str, field: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number.") from exc


def params_from_query(query: dict[str, list[str]]) -> SimpleNamespace:
    selected_algorithms = query.get("algorithms", DEFAULTS["algorithms"])
    algorithms = [name for name in selected_algorithms if name in DEFAULT_ALGORITHMS]
    if not algorithms:
        raise ValueError("Select at least one algorithm.")

    return SimpleNamespace(
        baskets=query.get("baskets", [DEFAULTS["baskets"]])[0].strip() or DEFAULTS["baskets"],
        output=query.get("output", [DEFAULTS["output"]])[0].strip() or DEFAULTS["output"],
        algorithms=algorithms,
        max_baskets=parse_int(query.get("max_baskets", [DEFAULTS["max_baskets"]])[0], "max_baskets"),
        top_items=parse_int(query.get("top_items", [DEFAULTS["top_items"]])[0], "top_items"),
        min_items=parse_int(query.get("min_items", [DEFAULTS["min_items"]])[0], "min_items"),
        min_support=parse_float(query.get("min_support", [DEFAULTS["min_support"]])[0], "min_support"),
        max_len=parse_int(query.get("max_len", [DEFAULTS["max_len"]])[0], "max_len"),
        start_date=query.get("start_date", [DEFAULTS["start_date"]])[0].strip() or None,
        end_date=query.get("end_date", [DEFAULTS["end_date"]])[0].strip() or None,
    )


def values_from_args(args: SimpleNamespace | None) -> dict[str, object]:
    if args is None:
        return DEFAULTS.copy()

    return {
        "baskets": args.baskets,
        "output": args.output,
        "max_baskets": str(args.max_baskets),
        "top_items": str(args.top_items),
        "min_items": str(args.min_items),
        "min_support": str(args.min_support),
        "max_len": str(args.max_len),
        "start_date": args.start_date or "",
        "end_date": args.end_date or "",
        "algorithms": args.algorithms,
    }


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

    algorithm_checks = "\n".join(
        f"""
        <label class="check">
          <input type="checkbox" name="algorithms" value="{name}" {"checked" if name in selected else ""}>
          <span>{name}</span>
        </label>
        """
        for name in DEFAULT_ALGORITHMS
    )

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
    input[type="text"], input[type="number"] {{
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
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      form {{ position: static; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Giao diện so sánh thuật toán</h1>
    <p>Điều chỉnh tham số, chạy Apriori, FP-growth, và Eclat, sau đó xuất biểu đồ so sánh.</p>
    <div class="layout">
      <form action="/run" method="get">
        {input_field("File CSV giỏ hàng", "baskets", "Mặc định: processed/baskets.csv")}
        {input_field("Ảnh đầu ra", "output", "Đường dẫn PNG để lưu. Ví dụ: results/chart_100k.png")}
        <label>
          <span>Thuật toán</span>
          <div class="checks">{algorithm_checks}</div>
        </label>
        {input_field("Số giỏ hàng tối đa", "max_baskets", "Dùng 0 để đọc tất cả.", "number")}
        {input_field("Số mặt hàng lấy top", "top_items", "Dùng 0 để giữ tất cả.", "number")}
        {input_field("Cỡ giỏ hàng tối thiểu", "min_items", "Thường là 2.", "number")}
        {input_field("Độ hỗ trợ tối thiểu (Min Support)", "min_support", "Ví dụ: 0.01 nghĩa là 1%.")}
        {input_field("Độ dài tập tối đa (Max Length)", "max_len", "Dùng 2 hoặc 3 với dữ liệu lớn.", "number")}
        {input_field("Ngày bắt đầu", "start_date", "Tùy chọn, định dạng YYYY-MM-DD.")}
        {input_field("Ngày kết thúc", "end_date", "Tùy chọn, định dạng YYYY-MM-DD.")}
        <div class="actions">
          <button type="submit">Chạy và Vẽ biểu đồ</button>
          <a class="button secondary" href="/">Làm mới</a>
        </div>
      </form>
      <section class="result">
        {result_html or "<p>Chạy một cấu hình để tạo biểu đồ so sánh.</p>"}
      </section>
    </div>
  </main>
</body>
</html>"""


def render_result(args: SimpleNamespace, summary: dict[str, object]) -> str:
    image_path = Path(args.output)
    image_url = f"/image?path={html.escape(str(image_path))}&t={int(time.time())}"
    escaped_summary = html.escape(json.dumps(summary, indent=2))
    return f"""
      <h2>Biểu đồ</h2>
      <p>Đã lưu: <code>{html.escape(str(image_path))}</code></p>
      <img src="{image_url}" alt="Biểu đồ so sánh thuật toán">
      <h2>Tóm tắt</h2>
      <pre>{escaped_summary}</pre>
    """


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

        if parsed.path == "/run":
            args = None
            try:
                args = params_from_query(query)
                summary, _ = run_algorithms(args)
                plot_comparison(summary, Path(args.output))
                self.send_html(render_form(values_from_args(args), render_result(args, summary)))
            except Exception as exc:
                values = values_from_args(args) if args is not None else values_from_args(None)
                message = f"<div class='error'><strong>Lỗi:</strong> {html.escape(str(exc))}</div>"
                self.send_html(render_form(values, message), status=400)
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
