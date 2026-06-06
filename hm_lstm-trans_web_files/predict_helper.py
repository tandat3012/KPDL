import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


def load_recommender(export_dir, model_type="lstm"):
    with open(os.path.join(export_dir, "model_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(export_dir, "old_to_new_item_map.json"), "r", encoding="utf-8") as f:
        old_to_new = {int(k): int(v) for k, v in json.load(f).items()}
    with open(os.path.join(export_dir, "new_to_old_item_map.json"), "r", encoding="utf-8") as f:
        new_to_old = {int(k): int(v) for k, v in json.load(f).items()}

    model_file = config["models"][model_type]
    model_path = os.path.join(export_dir, model_file)
    model = tf.keras.models.load_model(model_path)
    return model, config, old_to_new, new_to_old


def recommend_next_products(model, config, old_to_new, new_to_old, article_sequence, top_n=10):
    max_seq_len = int(config["max_seq_len"])
    remap_seq = [old_to_new.get(int(x), 0) for x in article_sequence]
    x = pad_sequences([remap_seq], maxlen=max_seq_len, padding="post", truncating="pre")
    prob = model.predict(x, verbose=0)[0]
    prob[0] = -1
    top_ids = np.argsort(prob)[::-1][:top_n]

    results = []
    for rank, new_id in enumerate(top_ids, start=1):
        results.append({
            "rank": int(rank),
            "new_item_idx": int(new_id),
            "old_article_idx": int(new_to_old.get(int(new_id), -1)),
            "score": float(prob[new_id]),
        })
    return results

# Vi du su dung trong web:
# model, config, old_to_new, new_to_old = load_recommender("./export", model_type="lstm")
# results = recommend_next_products(model, config, old_to_new, new_to_old, [123, 456, 789], top_n=10)
