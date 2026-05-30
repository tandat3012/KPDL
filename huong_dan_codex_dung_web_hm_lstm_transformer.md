# YEU CAU XAY DUNG WEB GOI Y SAN PHAM H&M

## 1. Muc tieu cua web

Xay dung web demo cho de tai: Goi y san pham mua kem / san pham tiep theo dua tren bo du lieu H&M Personalized Fashion Recommendations.

Web can co 2 nhom chuc nang chinh:

1. Goi y san pham mua kem bang cac thuat toan khai pha luat ket hop:
   - Apriori
   - FP-Growth
   - ECLAT

2. Goi y san pham tiep theo bang mo hinh hoc sau:
   - LSTM
   - Transformer

Yeu cau quan trong: Web phai chay model that, tuc la backend Flask load truc tiep file `.keras` va goi `model.predict()` khi nguoi dung nhap/chon san pham.

---

## 2. Cong nghe de xuat

Backend:
- Python
- Flask
- TensorFlow
- Pandas
- NumPy

Frontend:
- HTML
- CSS
- JavaScript

Thu vien can cai:

```bash
pip install flask pandas numpy tensorflow
```

Neu tao file `requirements.txt` thi noi dung nen gom:

```txt
flask
pandas
numpy
tensorflow
```

---

## 3. Cac file can dung

Can dat cac file sau vao project web:

```txt
models/
    lstm_hm_model.keras
    transformer_hm_model.keras

data/
    article_id_map.csv
    articles_filtered.csv
    lstm_vs_transformer_comparison.csv
```

Giai thich:

- `lstm_hm_model.keras`: model LSTM da train tren Kaggle.
- `transformer_hm_model.keras`: model Transformer da train tren Kaggle, dung de so sanh baseline voi LSTM.
- `article_id_map.csv`: dung de chuyen `article_id` goc sang `article_idx` ma model hieu.
- `articles_filtered.csv`: dung de lay thong tin san pham de hien thi len web, vi du ten san pham, loai san pham, mau sac, nhom san pham.
- `lstm_vs_transformer_comparison.csv`: dung de hien thi bang so sanh ket qua LSTM va Transformer tren web. File nay khong dung de du doan, chi dung de bao cao/trinh bay.

---

## 4. Cau truc thu muc de xuat

```txt
hm_recommendation_web/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── lstm_hm_model.keras
│   └── transformer_hm_model.keras
│
├── data/
│   ├── article_id_map.csv
│   ├── articles_filtered.csv
│   ├── lstm_vs_transformer_comparison.csv
│   └── cac_file_csv_apriori_fp_growth_eclat/
│
├── templates/
│   ├── index.html
│   ├── association.html
│   ├── deep_learning.html
│   └── comparison.html
│
└── static/
    ├── style.css
    └── script.js
```

---

## 5. Y tuong giao dien web

Web nen co menu gom 4 trang:

### Trang 1: Trang chu

Hien thi thong tin tom tat:

- Ten de tai: Goi y san pham mua kem tren bo du lieu H&M.
- Bo du lieu: H&M Personalized Fashion Recommendations.
- Thuat toan su dung:
  - Apriori
  - FP-Growth
  - ECLAT
  - LSTM
  - Transformer
- Mo ta ngan:
  - Apriori, FP-Growth, ECLAT dung de khai pha luat ket hop.
  - LSTM, Transformer dung de du doan san pham tiep theo dua tren chuoi san pham nguoi dung da mua/xem.

---

### Trang 2: Goi y bang luat ket hop

Trang nay dung cac file CSV hien co cua web, da chia theo mua, gioi tinh, mau sac.

Input nguoi dung:

- Chon mua: Xuan / Ha / Thu / Dong
- Chon gioi tinh: Nam / Nu / Unisex
- Chon mau sac
- Chon thuat toan:
  - Apriori
  - FP-Growth
  - ECLAT
- Chon san pham dau vao hoac nhap san pham dau vao

Output:

- Danh sach san pham mua kem
- Bang luat ket hop gom:
  - antecedents
  - consequents
  - support
  - confidence
  - lift

Muc dich:
- Cho thay cac thuat toan khai pha du lieu tim duoc moi quan he mua kem giua cac san pham.

---

### Trang 3: Goi y bang LSTM / Transformer

Trang nay la phan quan trong de web chay model that.

Input nguoi dung:

- Chon mot hoac nhieu san pham da mua/da xem.
- Cac san pham duoc chon se tao thanh mot chuoi dau vao.
- Chon model:
  - LSTM
  - Transformer
- Chon so luong ket qua goi y: Top 5 / Top 10 / Top 20

Xu ly backend:

1. Nhan danh sach `article_id` tu frontend.
2. Dung `article_id_map.csv` de chuyen `article_id` thanh `article_idx`.
3. Padding chuoi ve do dai `MAX_SEQ_LEN = 20`.
4. Dua chuoi vao model da chon:
   - Neu nguoi dung chon LSTM: dung `lstm_hm_model.keras`
   - Neu nguoi dung chon Transformer: dung `transformer_hm_model.keras`
5. Goi `model.predict()`.
6. Lay Top K san pham co xac suat cao nhat.
7. Dung `articles_filtered.csv` de lay thong tin san pham.
8. Tra ket qua ve frontend.

Output hien thi:

Moi san pham goi y nen hien:

- Thu tu xep hang
- `article_id`
- `article_idx`
- Ten san pham
- Loai san pham
- Nhom san pham
- Mau sac
- Diem du doan / score

Vi du:

```txt
Chuoi dau vao:
Ao thun trang -> Quan jeans xanh -> Ao khoac den

Model: LSTM

Top 10 san pham goi y:
1. Slim Jeans - Blue - score: 0.034
2. Cotton T-shirt - White - score: 0.029
3. Hoodie - Black - score: 0.026
...
```

---

### Trang 4: So sanh mo hinh

Trang nay doc file:

```txt
data/lstm_vs_transformer_comparison.csv
```

Va hien thi bang so sanh:

- Model
- Loss
- Accuracy
- Top-5 Accuracy
- Top-10 Accuracy
- Top-20 Accuracy

Co the ve bieu do cot neu duoc.

Muc dich:
- Chung minh ca LSTM va Transformer deu da duoc train.
- Cho thay ket qua so sanh baseline.
- Neu LSTM tot hon Transformer, ghi chu: LSTM duoc chon lam model chinh, Transformer dung lam baseline so sanh.

---

## 6. Flow hoat dong cua chuc nang LSTM / Transformer

Flow tong quat:

```txt
Nguoi dung chon san pham
        ↓
Frontend gui danh sach article_id len Flask API
        ↓
Backend doi article_id sang article_idx
        ↓
Padding chuoi ve do dai 20
        ↓
Load model LSTM hoac Transformer
        ↓
model.predict()
        ↓
Lay Top K article_idx co score cao nhat
        ↓
Doi article_idx ve thong tin san pham
        ↓
Tra JSON ve frontend
        ↓
Frontend hien thi danh sach goi y
```

---

## 7. API de xuat

### API 1: Lay danh sach san pham de chon

Endpoint:

```txt
GET /api/products
```

Tra ve danh sach san pham tu `articles_filtered.csv`.

Nen tra ve cac cot:

```txt
article_id
article_idx
prod_name
product_type_name
product_group_name
colour_group_name
```

---

### API 2: Goi y bang LSTM hoac Transformer

Endpoint:

```txt
POST /api/recommend/deep-learning
```

JSON gui len:

```json
{
  "model": "lstm",
  "article_ids": ["0884319008", "0715624001", "0751471001"],
  "top_k": 10
}
```

Hoac neu frontend dung truc tiep `article_idx`:

```json
{
  "model": "lstm",
  "article_indices": [2397, 2908, 507],
  "top_k": 10
}
```

JSON tra ve:

```json
{
  "status": "success",
  "model": "lstm",
  "input_sequence": [2397, 2908, 507],
  "recommendations": [
    {
      "rank": 1,
      "article_idx": 636,
      "article_id": "xxxxxxx",
      "prod_name": "Product name",
      "product_type_name": "Trousers",
      "product_group_name": "Garment Lower body",
      "colour_group_name": "Black",
      "score": 0.034
    }
  ]
}
```

---

### API 3: Lay bang so sanh mo hinh

Endpoint:

```txt
GET /api/model-comparison
```

Doc file:

```txt
data/lstm_vs_transformer_comparison.csv
```

Tra ve JSON de hien thi bang so sanh tren frontend.

---

## 8. Code xu ly model trong Flask - y tuong

Trong `app.py` can load model mot lan khi khoi dong app, khong nen load model moi lan request.

Vi du:

```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

MAX_SEQ_LEN = 20

lstm_model = tf.keras.models.load_model("models/lstm_hm_model.keras")
transformer_model = tf.keras.models.load_model("models/transformer_hm_model.keras")

articles_df = pd.read_csv("data/articles_filtered.csv")
article_map_df = pd.read_csv("data/article_id_map.csv")

article_id_to_idx = dict(zip(article_map_df["article_id"].astype(str), article_map_df["article_idx"]))
article_idx_to_id = dict(zip(article_map_df["article_idx"], article_map_df["article_id"].astype(str)))

def recommend_with_model(model_name, input_sequence, top_k=10):
    if model_name == "lstm":
        model = lstm_model
    elif model_name == "transformer":
        model = transformer_model
    else:
        raise ValueError("Model khong hop le")

    input_padded = pad_sequences(
        [input_sequence],
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="pre"
    )

    preds = model.predict(input_padded, verbose=0)[0]

    top_indices = np.argsort(preds)[-top_k:][::-1]

    result = articles_df[articles_df["article_idx"].isin(top_indices)].copy()
    result["score"] = result["article_idx"].apply(lambda x: float(preds[int(x)]))
    result = result.sort_values("score", ascending=False)

    return result.to_dict(orient="records")
```

---

## 9. Luu y quan trong ve model Transformer

Neu model Transformer co custom layer, khi load co the can khai bao custom_objects.

Cac class custom co the gom:

- `TokenAndPositionEmbedding`
- `TransformerBlock`
- `LastNonZeroPooling`

Neu load Transformer bi loi, can dua lai cac class custom nay vao `app.py` truoc khi goi:

```python
transformer_model = tf.keras.models.load_model(
    "models/transformer_hm_model.keras",
    custom_objects={
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,
        "LastNonZeroPooling": LastNonZeroPooling
    }
)
```

LSTM thuong khong can custom_objects.

---

## 10. Luu y ve article_id va article_idx

Model khong hieu `article_id` goc. Model chi hieu `article_idx`.

Vay frontend co 2 cach:

### Cach 1: Frontend gui article_idx

De lam nhanh, khi hien danh sach san pham cho nguoi dung chon, nen gan san pham bang `article_idx`.

Khi nguoi dung chon san pham, gui luon `article_idx` len backend.

### Cach 2: Frontend gui article_id

Neu frontend gui `article_id`, backend phai dung `article_id_map.csv` de doi sang `article_idx`.

Nen ho tro ca hai cach neu co the.

---

## 11. Noi dung hien thi tren web de de bao cao

Nen co cac phan chu thich ngan:

### Voi Apriori / FP-Growth / ECLAT

```txt
Cac thuat toan khai pha luat ket hop duoc dung de tim cac san pham thuong duoc mua cung nhau. Ket qua duoc danh gia bang support, confidence va lift.
```

### Voi LSTM / Transformer

```txt
LSTM va Transformer duoc dung de du doan san pham tiep theo dua tren chuoi san pham khach hang da mua hoac da xem. Ket qua tra ve la Top K san pham co xac suat cao nhat.
```

### Voi so sanh mo hinh

```txt
Do bai toan co 3000 lop san pham, Accuracy thong thuong co the khong cao. Vi vay he thong bo sung Top-5, Top-10 va Top-20 Accuracy. Trong bai toan goi y san pham, Top-10 Accuracy phu hop hon vi he thong thuong goi y nhieu san pham cung luc.
```

---

## 12. Yeu cau hoan thien cho Codex

Codex hay giup xay dung web Flask theo y tuong tren.

Yeu cau:

1. Cai moi truong can thiet, dac biet la TensorFlow.
2. Tao cau truc thu muc nhu tren.
3. Viet `app.py` de:
   - Load LSTM model.
   - Load Transformer model.
   - Load `article_id_map.csv`.
   - Load `articles_filtered.csv`.
   - Tao API goi y bang LSTM/Transformer.
   - Tao API doc bang so sanh model.
4. Tao giao dien HTML/CSS/JS gom:
   - Trang chu.
   - Trang goi y bang luat ket hop.
   - Trang goi y bang LSTM/Transformer.
   - Trang so sanh mo hinh.
5. Giao dien can don gian, de demo, de giai thich voi giang vien.
6. Mac dinh model nen chon LSTM, vi LSTM la model chinh. Transformer dung de so sanh baseline.
7. Khi nguoi dung chon cung mot chuoi san pham, web co the cho chay LSTM va Transformer de so sanh ket qua Top 10.
8. Neu model Transformer load bi loi custom layer, hay them lai cac class custom layer vao `app.py`.

---

## 13. Ket luan y tuong

Web se co 2 huong demo:

1. Khai pha luat ket hop:
   - Input: mua, gioi tinh, mau sac, thuat toan, san pham.
   - Output: san pham mua kem + support/confidence/lift.

2. Hoc sau LSTM/Transformer:
   - Input: chuoi san pham da mua/da xem.
   - Output: Top K san pham goi y.
   - Co the so sanh ket qua LSTM va Transformer tren cung mot input.

Trang so sanh model se dung `lstm_vs_transformer_comparison.csv` de hien thi bang ket qua train/evaluate.
