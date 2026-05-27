# KPDL Frequent Itemset Mining

## Thu tu chay

Chay cac lenh tu root cua project:

```bash
KPDL_env/bin/python scripts/step_01_data_report.py
KPDL_env/bin/python scripts/step_02_preprocess_csv.py
KPDL_env/bin/python scripts/step_03_benchmark_algorithms.py
KPDL_env/bin/python scripts/step_04_plot_algorithm_comparison.py
```

Neu muon chay nhieu cau hinh benchmark va xuat bang tong hop:

```bash
KPDL_env/bin/python scripts/step_05_run_experiments.py
```

## Vai tro tung file

- `step_01_data_report.py`: doc nhanh cac file CSV goc trong `orginal_data_csv/`.
- `step_02_preprocess_csv.py`: tao `processed/baskets.csv` va `processed/articles_minimal.csv`.
- `step_03_benchmark_algorithms.py`: chay Apriori, FP-growth, Eclat va xuat ket qua vao `results/`.
- `step_04_plot_algorithm_comparison.py`: tao bieu do so sanh thoi gian chay va bo nho.
- `step_05_run_experiments.py`: chay nhieu cau hinh benchmark de so sanh.

## File khong nam trong thu tu chay chinh

- `algorithm_ui.py`: giao dien browser de chay tuong tac, khong bat buoc khi push code cho team.
- `build_basket_variants.py`: tao basket bien the cho UI/thi nghiem phu.
- `legacy_clean_csv_data.py` va `legacy_build_test_basket.py`: script cu, giu lai de tham khao.

## Du lieu va ket qua

Khong commit CSV goc hoac file sinh ra. Cac file trong `orginal_data_csv/`, `processed/`, `reports/`, `results/` se duoc tao lai khi chay pipeline va da duoc dua vao `.gitignore`.
