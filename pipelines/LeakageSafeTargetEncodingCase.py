# Leakage-Safe Target Encoding Pipeline (Skrub + Provenance)
# Goal: predict late delivery, with a train-only aggregated feature (no data leakage).
#
# Run:
#   python -m pipelines.LeakageSafeTargetEncodingCase
#   python -m pipelines.LeakageSafeTargetEncodingCase --track-provenance

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import skrub

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier


# -------------------------------------------------
# CLI arguments
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--track-provenance", action="store_true")
args = parser.parse_args()

if args.track_provenance:
    print("Provenance is enabled")
    from src.rdepro_skrub_provenance.monkey_patching_v02_data_provenance import enable_why_data_provenance
    enable_why_data_provenance()
else:
    print("Provenance is disabled")


# -------------------------------------------------
# 1) LOAD DATA
# -------------------------------------------------
orders = skrub.var("orders", pd.read_csv("./src/datasets/olist_orders_dataset.csv"))
order_items = skrub.var("order_items", pd.read_csv("./src/datasets/olist_order_items_dataset.csv"))
products = skrub.var("products", pd.read_csv("./src/datasets/olist_products_dataset.csv"))
cat_tr = skrub.var("cat_tr", pd.read_csv("./src/datasets/product_category_name_translation.csv"))

print("Datasets loaded")


# -------------------------------------------------
# 2) JOIN + TARGET (DataOps-friendly) ثم materialize once
# -------------------------------------------------
products_en = (
    products
    .merge(cat_tr, on="product_category_name", how="left")
    .rename(columns={"product_category_name_english": "product_category_en"})
)

df = (
    orders
    .merge(order_items, on="order_id", how="left")
    .merge(products_en.skb.select(["product_id", "product_category_en"]), on="product_id", how="left")
)

# Lazy datetime parsing
df = df.assign(
    order_delivered_customer_date=lambda d: pd.to_datetime(d["order_delivered_customer_date"], errors="coerce"),
    order_estimated_delivery_date=lambda d: pd.to_datetime(d["order_estimated_delivery_date"], errors="coerce"),
)

df = df.assign(
    is_late=lambda d: (d["order_delivered_customer_date"] > d["order_estimated_delivery_date"]).fillna(False).astype(int)
)

df = df.assign(
    price=lambda d: d["price"].fillna(0),
    freight_value=lambda d: d["freight_value"].fillna(0),
    product_category_en=lambda d: d["product_category_en"].fillna("unknown"),
)

# Materialize once (so we can split without fitting a dummy model on strings)
print("Building a concrete pandas dataframe (preview)...")
df_pd = df.skb.preview()
print("Preview built:", df_pd.shape)


# -------------------------------------------------
# 3) SPLIT FIRST (no leakage)
# -------------------------------------------------
X_pd = df_pd[["product_category_en", "price", "freight_value"]].copy()
y_pd = df_pd["is_late"].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_pd, y_pd, test_size=0.2, random_state=0, stratify=y_pd
)

print("Split done:", X_train.shape, X_test.shape)


# -------------------------------------------------
# 4) TRAIN-ONLY AGGREGATION (target encoding)
# -------------------------------------------------
train_with_y = X_train.copy()
train_with_y["is_late"] = y_train.values

cat_stats = (
    train_with_y
    .groupby("product_category_en", dropna=False)
    .agg(cat_late_rate=("is_late", "mean"))
    .reset_index()
)

global_rate = float(train_with_y["is_late"].mean())

X_train2 = X_train.merge(cat_stats, on="product_category_en", how="left")
X_test2 = X_test.merge(cat_stats, on="product_category_en", how="left")

X_train2["cat_late_rate"] = X_train2["cat_late_rate"].fillna(global_rate)
X_test2["cat_late_rate"] = X_test2["cat_late_rate"].fillna(global_rate)


# -------------------------------------------------
# 5) PREPROCESS + MODEL (numeric only after encoding)
# -------------------------------------------------
feature_cols = ["price", "freight_value", "cat_late_rate"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), feature_cols),
    ],
    remainder="drop",
)

X_train_final = preprocessor.fit_transform(X_train2[feature_cols])
X_test_final = preprocessor.transform(X_test2[feature_cols])

model = HistGradientBoostingClassifier(random_state=0)
model.fit(X_train_final, y_train)

score = model.score(X_test_final, y_test)
print(f"Test accuracy: {score}")
