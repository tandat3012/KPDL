import pandas as pd
from pathlib import Path

RAW_DIR = Path("orginal_data_csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

files = {
    "transactions": RAW_DIR / "transactions_train.csv",
    "articles": RAW_DIR / "articles.csv",
    "customers": RAW_DIR / "customers.csv",
}

report_lines = []

for name, path in files.items():
    df = pd.read_csv(path)

    report_lines.append(f"\n===== {name.upper()} =====")
    report_lines.append(f"Shape: {df.shape}")
    report_lines.append("\nColumns:")
    report_lines.append(str(list(df.columns)))

    report_lines.append("\nMissing values:")
    report_lines.append(str(df.isnull().sum()))

    report_lines.append("\nUnique values:")
    for col in df.columns:
        report_lines.append(f"{col}: {df[col].nunique()}")

    if name == "transactions":
        report_lines.append(f"\nDate range: {df['t_dat'].min()} -> {df['t_dat'].max()}")
        report_lines.append("\nTop 10 articles:")
        report_lines.append(str(df["article_id"].value_counts().head(10)))

output_path = REPORT_DIR / "data_overview.txt"
output_path.write_text("\n".join(report_lines), encoding="utf-8")

print(f"Report saved to {output_path}")